"""Vector search over the CloudServe documentation. (AC-A4, REQ-F03)

Chunking configuration is a measured design decision, not a default — see
docs/retrieval_experiment.md and scripts/tune_chunking.py.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence

# Chroma reads this at import time; no outbound calls from the vector store.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings, get_settings
from src.schemas import Document, RetrievedPassage

log = logging.getLogger(__name__)

_CHROMA_SETTINGS = chromadb.config.Settings(anonymized_telemetry=False)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)


def _client(path: Path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(path), settings=_CHROMA_SETTINGS)


def build_embedding_function(settings: Settings | None = None) -> EmbeddingFunction:
    """all-MiniLM-L6-v2 either way; ONNX avoids pulling torch into a clean install."""
    settings = settings or get_settings()
    backend = settings.embedding_backend.lower()
    if backend in {"sentence-transformers", "sentence_transformers", "torch"}:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
    return embedding_functions.ONNXMiniLM_L6_V2()


def load_documents(path: str | Path | None = None, settings: Settings | None = None) -> list[Document]:
    settings = settings or get_settings()
    path = Path(path) if path else settings.documentation_file
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("documents", raw.get("docs", []))
    docs: list[Document] = []
    for item in raw:
        # The Setup Guide sample reads doc["id"]; the dataset field is doc_id.
        item.setdefault("doc_id", item.get("id", ""))
        docs.append(Document.model_validate(item))
    return docs


def chunk_documents(
    documents: Sequence[Document], chunk_size: int, chunk_overlap: int
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Prefix each chunk with its document title so short chunks stay self-describing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []
    for doc in documents:
        for position, chunk in enumerate(splitter.split_text(doc.content)):
            chunk = chunk.strip()
            if not chunk:
                continue
            chunk_id = f"{doc.doc_id}#{position:03d}"
            ids.append(chunk_id)
            texts.append(f"{doc.title}\n\n{chunk}")
            metadatas.append(
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "category": doc.category,
                    "applies_to": doc.applies_to,
                    "chunk_index": position,
                }
            )
    return texts, metadatas, ids


def build_index(
    settings: Settings | None = None,
    documents: Sequence[Document] | None = None,
    reset: bool = True,
) -> int:
    """Create or replace the persistent Chroma collection. Idempotent."""
    settings = settings or get_settings()
    documents = documents if documents is not None else load_documents(settings=settings)

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = _client(settings.chroma_dir)
    if reset:
        try:
            client.delete_collection(settings.chroma_collection)
        except Exception:  # noqa: BLE001 - absent collection is the normal first run
            pass

    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=build_embedding_function(settings),
        metadata={"hnsw:space": "cosine"},
    )
    texts, metadatas, ids = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    for start in range(0, len(texts), 128):
        stop = start + 128
        collection.add(ids=ids[start:stop], documents=texts[start:stop], metadatas=metadatas[start:stop])
    log.info("indexed %d passages from %d documents", len(texts), len(documents))
    return len(texts)


class Retriever:
    """Returns nothing when nothing is relevant.

    170 of the 580 supplied tickets have no expected documents at all, so forcing
    top-k results would manufacture citations for questions the corpus cannot answer.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            if not self.settings.chroma_dir.exists():
                raise FileNotFoundError(
                    f"No Chroma store at {self.settings.chroma_dir}. "
                    "Run: python -m scripts.build_index"
                )
            client = _client(self.settings.chroma_dir)
            self._collection = client.get_collection(
                name=self.settings.chroma_collection,
                embedding_function=build_embedding_function(self.settings),
            )
        return self._collection

    def search(
        self, query: str, top_k: int | None = None, max_distance: float | None = None
    ) -> list[RetrievedPassage]:
        top_k = top_k or self.settings.retrieval_top_k
        max_distance = self.settings.retrieval_max_distance if max_distance is None else max_distance
        query = (query or "").strip()
        if not query:
            return []

        result = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        passages: list[RetrievedPassage] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for rank, (chunk_id, text, meta, distance) in enumerate(zip(ids, docs, metas, dists), start=1):
            if distance > max_distance:
                continue
            passages.append(
                RetrievedPassage(
                    chunk_id=chunk_id,
                    doc_id=str(meta.get("doc_id", "")),
                    title=str(meta.get("title", "")),
                    category=str(meta.get("category", "")),
                    text=text,
                    distance=float(distance),
                    rank=rank,
                )
            )
        return passages


def unique_doc_ids(passages: Sequence[RetrievedPassage]) -> list[str]:
    seen: list[str] = []
    for p in passages:
        if p.doc_id and p.doc_id not in seen:
            seen.append(p.doc_id)
    return seen
