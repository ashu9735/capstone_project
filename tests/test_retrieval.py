"""Citation validation and retrieval behaviour. (AC-A4, AC-A6)"""

from __future__ import annotations

import pytest

from src.generate import extract_citations, format_passages, validate_citations
from src.retrieve import (
    Retriever,
    build_index,
    chunk_documents,
    lexical_overlap,
    load_documents,
    reciprocal_rank_fusion,
    unique_doc_ids,
)


class TestCitations:
    def test_extracts_document_identifiers(self):
        text = "Clear cookies [DOC-AUTH-001]. Then re-enrol the device [DOC-AUTH-002]."
        assert extract_citations(text) == ["DOC-AUTH-001", "DOC-AUTH-002"]

    def test_duplicates_reported_once(self):
        assert extract_citations("[DOC-A-001] x [DOC-A-001]") == ["DOC-A-001"]

    def test_citations_in_the_retrieved_set_are_kept(self, passages):
        text, kept, dropped = validate_citations("Do the thing [DOC-AUTH-001].", passages)
        assert kept == ["DOC-AUTH-001"]
        assert dropped == []
        assert "[DOC-AUTH-001]" in text

    def test_citations_outside_the_retrieved_set_are_removed(self, passages):
        text, kept, dropped = validate_citations(
            "Do the thing [DOC-AUTH-001] and also this [DOC-FAKE-999].", passages
        )
        assert kept == ["DOC-AUTH-001"]
        assert dropped == ["DOC-FAKE-999"]
        assert "DOC-FAKE-999" not in text

    def test_every_kept_citation_resolves_to_a_passage(self, passages):
        allowed = {p.doc_id for p in passages}
        _, kept, _ = validate_citations("[DOC-AUTH-001] [DOC-AUTH-002] [DOC-NOPE-001]", passages)
        assert set(kept).issubset(allowed)

    def test_passages_are_formatted_with_their_identifier(self, passages):
        block = format_passages(passages)
        for p in passages:
            assert f"[{p.doc_id}]" in block


class TestChunking:
    def test_chunks_carry_resolvable_document_ids(self):
        documents = load_documents()
        texts, metadatas, ids = chunk_documents(documents, 800, 120)
        known = {d.doc_id for d in documents}
        assert len(texts) == len(metadatas) == len(ids)
        assert texts
        assert all(m["doc_id"] in known for m in metadatas)
        assert len(set(ids)) == len(ids)

    def test_smaller_chunks_produce_more_passages(self):
        documents = load_documents()
        few, _, _ = chunk_documents(documents, 1200, 120)
        many, _, _ = chunk_documents(documents, 400, 60)
        assert len(many) > len(few)


class TestRetrieval:
    @pytest.fixture(scope="class")
    def retriever(self, settings):
        build_index(settings=settings)
        return Retriever(settings)

    def test_a_documented_question_retrieves_a_real_passage(self, retriever):
        results = retriever.search("I cannot log in, my account says invalid credentials")
        assert results
        known = {d.doc_id for d in load_documents()}
        assert all(r.doc_id in known for r in results)
        assert all(r.text.strip() for r in results)

    def test_irrelevant_query_returns_nothing(self, retriever):
        results = retriever.search("the migratory patterns of arctic terns", max_distance=0.3)
        assert results == []

    def test_empty_query_returns_nothing(self, retriever):
        assert retriever.search("   ") == []

    def test_results_are_ordered_by_distance(self, retriever):
        results = retriever.search("multi-factor authentication code rejected")
        assert results == sorted(results, key=lambda r: r.distance)

    def test_unique_doc_ids_preserves_rank_order(self, passages):
        assert unique_doc_ids(passages + passages) == ["DOC-AUTH-001", "DOC-AUTH-002"]

    def test_lexical_overlap_rewards_shared_terms(self):
        relevant = lexical_overlap("invalid credentials login", "Login fails with invalid credentials")
        unrelated = lexical_overlap("invalid credentials login", "Billing invoices and refunds")
        assert relevant > unrelated

    def test_reciprocal_rank_fusion_deduplicates_and_reranks(self, passages):
        fused = reciprocal_rank_fusion([passages, list(reversed(passages))])
        assert [p.chunk_id for p in fused] == ["DOC-AUTH-001#000", "DOC-AUTH-002#000"]
        assert [p.rank for p in fused] == [1, 2]
