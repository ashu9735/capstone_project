"""Build the Chroma index. Run once after install: python -m scripts.build_index"""

from __future__ import annotations

import argparse
import logging

from src.config import get_settings
from src.retrieve import build_index, load_documents


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed the documentation into Chroma.")
    parser.add_argument("--docs", default=None, help="Path to documentation.json")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    if args.chunk_size:
        settings.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        settings.chunk_overlap = args.chunk_overlap

    documents = load_documents(args.docs, settings=settings)
    count = build_index(settings=settings, documents=documents)
    print(
        f"Indexed {count} passages from {len(documents)} documents "
        f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}) "
        f"into {settings.chroma_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
