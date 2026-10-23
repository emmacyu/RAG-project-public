"""Offline tests for hybrid retrieval (no LLM calls, no Docker services needed)."""
import os
import sys
import uuid
from pathlib import Path

import pytest
from langchain_core.documents import Document

backend_dir = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(backend_dir))

from app.services.retrieval import (AdaptiveCutoff, BM25Index, CrossEncoderReranker, HybridRetriever,
                                    bm25_tokenize, rrf_fuse)

PDF = backend_dir / "tests" / "data" / "guideline_e23_model_risk_management_2027.pdf"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
needs_pdf = pytest.mark.skipif(not PDF.exists(), reason=f"sample PDF not found at {PDF}")


def doc(text, source="s"):
    return Document(page_content=text, metadata={"source": source})


# --- Pure functions ---

def test_tokenizer_keeps_guideline_codes():
    tokens = bm25_tokenize("See OSFI Guidelines B-13 and E-21, section A.4.")
    assert {"b-13", "e-21", "a.4"} <= set(tokens)
    assert {"b", "13", "e", "21"} <= set(tokens)   # parts are indexed too
    assert "and" not in tokens                     # stopwords dropped


def test_bm25_ranks_exact_code_first():
    index = BM25Index([doc("models from vendors are covered by Guideline B-10"), doc("vendors and data sources"),
                       doc("unrelated text about monitoring")])
    assert "B-10" in index.search("Which guideline, B-10?", 3)[0].page_content


def test_bm25_returns_nothing_without_overlap():
    assert BM25Index([doc("alpha beta")]).search("gamma", 5) == []


def scored(*values):
    return [(v, doc(f"d{i}")) for i, v in enumerate(values)]


def test_cutoff_returns_nothing_when_best_is_irrelevant():
    assert AdaptiveCutoff(min_score=-6, margin=6).apply(scored(-7, -9, -10), max_k=8) == []


def test_cutoff_keeps_only_chunks_close_to_the_best():
    kept = AdaptiveCutoff(min_score=-6, margin=6, min_k=1).apply(scored(8, 5, 2.5, 1, -3), max_k=8)
    assert [d.page_content for d in kept] == ["d0", "d1", "d2"]          # 1 and -3 are more than 6 below 8


def test_cutoff_respects_min_and_max_k():
    one_strong = scored(9, -2, -5, -8)
    assert len(AdaptiveCutoff(min_score=-6, margin=3, min_k=2).apply(one_strong, max_k=8)) == 2
    many_close = scored(*[5 - i * 0.1 for i in range(20)])
    assert len(AdaptiveCutoff(min_score=-6, margin=6, min_k=2).apply(many_close, max_k=8)) == 8


def test_rrf_prefers_documents_found_by_both():
    a, b, c = doc("a"), doc("b"), doc("c")
    fused = rrf_fuse([[a, b], [c, b]])
    assert fused[0].page_content == "b"
    assert len(fused) == 3                         # duplicates merged


# --- End to end on the sample guideline ---

@pytest.fixture(scope="module")
def store():
    if not PDF.exists():
        pytest.skip(f"sample PDF not found at {PDF}")
    import chromadb
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from app.services.chunking import chunk_pdf

    docs, ids = chunk_pdf(str(PDF), EMBEDDING_MODEL)
    s = Chroma(client=chromadb.EphemeralClient(), collection_name=f"test-{uuid.uuid4().hex[:8]}",
               embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))
    s.add_documents(docs, ids=ids)
    return s


@needs_pdf
def test_reranked_hybrid_puts_answer_first(store):
    retriever = HybridRetriever(store, reranker=CrossEncoderReranker(RERANK_MODEL))
    results = retriever.search("Which other OSFI guideline applies to models bought from outside vendors?", k=5)
    assert len(results) == 5
    assert "B-10" in results[0].page_content
    assert "rerank_score" in results[0].metadata


@needs_pdf
def test_bm25_can_be_switched_off(store):
    retriever = HybridRetriever(store, reranker=None, use_bm25=False)
    dense = store.similarity_search("model inventory", k=3)
    assert [d.page_content for d in retriever.search("model inventory", k=3)] == [d.page_content for d in dense]


@needs_pdf
def test_keyword_index_is_rebuilt_after_invalidate(store):
    retriever = HybridRetriever(store, reranker=None)
    retriever.search("model owner", k=3)
    before = len(retriever._keyword_index().docs)
    store.add_documents([Document(page_content="zebra-77 appendix", metadata={"source": "extra"})], ids=["extra-1"])
    try:
        assert len(retriever._keyword_index().docs) == before      # cached until told otherwise
        retriever.invalidate()
        assert len(retriever._keyword_index().docs) == before + 1
        # Check the keyword index itself: the final fused order is not a stable test target, because
        # Chroma's approximate (HNSW) search can leave a just-added chunk out of the dense candidates,
        # and then it ties in RRF with the dense top hit.
        assert retriever._keyword_index().search("zebra-77", 1)[0].page_content == "zebra-77 appendix"
        assert "zebra-77 appendix" in [d.page_content for d in retriever.search("zebra-77", k=5)]
    finally:
        store.delete(ids=["extra-1"])


@needs_pdf
def test_production_wiring_uses_settings_and_follows_uploads(monkeypatch, tmp_path):
    import chromadb

    for key, value in {"OPENROUTER_API_KEY": "test", "DATABASE_URL": "postgresql://x", "REDIS_URL": "redis://x"}.items():
        monkeypatch.setenv(key, os.environ.get(key, value))
    import app.services.vector_store as vs
    from app.config import settings

    client = chromadb.EphemeralClient()
    monkeypatch.setattr(vs, "get_chroma_client", lambda: client)
    monkeypatch.setattr(vs, "COLLECTION_NAME", f"wiring-{uuid.uuid4().hex[:8]}")
    monkeypatch.setattr(vs, "vector_store_instance", None)
    monkeypatch.setattr(vs, "retriever_instance", None)

    upload = tmp_path / "e23.pdf"
    upload.write_bytes(PDF.read_bytes())
    vs.process_pdf(str(upload))
    retriever = vs.get_retriever()
    assert retriever.reranker.model_name == settings.RERANK_MODEL
    assert retriever.use_bm25 == settings.USE_BM25
    assert retriever.cutoff.min_score == settings.RERANK_MIN_SCORE
    found = retriever.search("Who approves the use of a model?", k=settings.RETRIEVAL_TOP_K)
    assert settings.RETRIEVAL_MIN_K <= len(found) <= settings.RETRIEVAL_TOP_K

    vs.process_pdf(str(upload))                    # a new upload replaces the store ...
    assert vs.get_retriever() is not retriever     # ... and with it the stale BM25 index


CALIBRATED = AdaptiveCutoff(min_score=-6.0, margin=6.0, min_k=2)   # same values as app/config.py


@needs_pdf
def test_cutoff_on_the_guideline(store):
    retriever = HybridRetriever(store, reranker=CrossEncoderReranker(RERANK_MODEL), cutoff=CALIBRATED)
    assert retriever.search("What will the weather be like in Toronto tomorrow?", k=8) == []
    narrow = retriever.search("Which other OSFI guideline applies to models bought from outside vendors?", k=8)
    assert "B-10" in narrow[0].page_content
    broad = retriever.search("How should AI and machine learning model risk be managed across the model lifecycle?", k=8)
    assert len(broad) > len(narrow)


def test_agent_answers_directly_when_nothing_is_relevant(monkeypatch):
    for key, value in {"OPENROUTER_API_KEY": "test", "DATABASE_URL": "postgresql://x", "REDIS_URL": "redis://x"}.items():
        monkeypatch.setenv(key, os.environ.get(key, value))
    import app.services.agent as agent
    from langchain_core.messages import HumanMessage

    class NoLLM:
        def invoke(self, *args, **kwargs):
            raise AssertionError("the LLM must not be called when retrieval found nothing")

        def bind(self, *args, **kwargs):
            return self

    monkeypatch.setattr(agent, "llm", NoLLM())
    state = {"messages": [HumanMessage(content="What is the fine for breaching E-23?")],
             "intent": "audit", "current_context": "", "retry_count": 0, "audit_trail": []}
    reply = agent.generate_node(state)["messages"][0].content
    assert reply == agent.NO_RELEVANT_CONTENT
    assert agent.auditor_node({**state, "messages": state["messages"] + [reply]}) == {"is_hallucinating": False}
