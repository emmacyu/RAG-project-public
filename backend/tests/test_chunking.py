"""Offline tests for PDF chunking and ingestion (no LLM calls, no Docker services needed)."""
import os
import re
import sys
from pathlib import Path

import pytest

backend_dir = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(backend_dir))

from app.services.chunking import build_sections, chunk_pdf, is_minor_heading, load_pdf_pages, major_heading_level

PDF = backend_dir / "tests" / "data" / "guideline_e23_model_risk_management_2027.pdf"
MODEL = "all-MiniLM-L6-v2"

needs_pdf = pytest.mark.skipif(not PDF.exists(), reason=f"sample PDF not found at {PDF}")


# --- Heading detection (pure functions, no PDF needed) ---

@pytest.mark.parametrize("line, level", [
    ("A. Overview", 1),
    ("A.1 Purpose", 2),
    ("D.2 Components of the model lifecycle", 2),
    ("3.2.1 Data quality", 3),
    ("Appendix 1: Information tracking for models", 1),
])
def test_section_headings_are_recognised(line, level):
    assert major_heading_level(line) == level


@pytest.mark.parametrize("line", [
    "1. data input component that may also include relevant assumptions,",          # list item
    "1. Model risk is well understood and managed across the enterprise.",         # sentence in a list
    "Principle 2.1: Institutions should identify and track all models in use or recently decommissioned.",
    "This guideline applies on a risk-basis, proportional to the institution's:",
])
def test_list_items_and_sentences_are_not_headings(line):
    assert major_heading_level(line) is None


def test_minor_heading_needs_a_paragraph_after_it():
    body = "The unit(s) or individual(s) responsible for coordinating model development and deployment."
    assert is_minor_heading("Model Owner", body)
    assert not is_minor_heading("Model Owner", "size")            # followed by a short list item
    assert not is_minor_heading("strategy", body)                 # lower-case list item
    assert not is_minor_heading("Model design encompasses:", body)


def test_unstructured_document_falls_back():
    pages = [{"page": i, "page_label": str(i), "text": "Plain text without any headings at all, just prose.\nMore prose here."}
             for i in range(1, 4)]
    assert build_sections(pages) is None


# --- Behaviour on the sample guideline ---


@pytest.fixture(scope="module")
def chunks():
    if not PDF.exists():
        pytest.skip(f"sample PDF not found at {PDF}")
    return chunk_pdf(str(PDF), MODEL)


@needs_pdf
def test_pages_are_one_based():
    pages = load_pdf_pages(str(PDF))
    assert pages[0]["page"] == 1
    page_with_definition = next(p for p in pages if "Model risk involves the risk of adverse" in p["text"])
    assert page_with_definition["page"] == 4  # printed "Page 4" in the PDF


@needs_pdf
def test_repeated_page_headers_are_removed():
    for page in load_pdf_pages(str(PDF)):
        assert not re.search(r"Guideline E-23 – Model Risk Management \(2027\)\s*Page \d+", page["text"])


@needs_pdf
def test_wrapped_lines_are_joined():
    # In the PDF this sentence wraps after "should be"; it must come back as one line.
    text = "\n".join(p["text"] for p in load_pdf_pages(str(PDF)))
    assert "no longer fit for purpose should be modified, replaced, or decommissioned" in text


@needs_pdf
def test_chunks_fit_the_embedding_model(chunks):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(f"sentence-transformers/{MODEL}", device="cpu")
    limit = model.max_seq_length - 2
    docs, _ = chunks
    assert max(len(model.tokenizer.tokenize(d.page_content)) for d in docs) <= limit


@needs_pdf
def test_chunk_ids_are_unique_and_deterministic(chunks):
    docs, ids = chunks
    assert len(ids) == len(set(ids)) == len(docs)
    _, ids_again = chunk_pdf(str(PDF), MODEL)
    assert ids_again == ids


@needs_pdf
def test_reupload_replaces_instead_of_duplicating(monkeypatch, tmp_path):
    import chromadb

    # vector_store imports app.config, which refuses to start without these settings
    for key, value in {"OPENROUTER_API_KEY": "test", "DATABASE_URL": "postgresql://x", "REDIS_URL": "redis://x"}.items():
        monkeypatch.setenv(key, os.environ.get(key, value))
    import app.services.vector_store as vs

    client = chromadb.EphemeralClient()
    monkeypatch.setattr(vs, "get_chroma_client", lambda: client)
    monkeypatch.setattr(vs, "COLLECTION_NAME", f"test-{tmp_path.name}")

    upload = tmp_path / "e23.pdf"
    upload.write_bytes(PDF.read_bytes())
    first = vs.process_pdf(str(upload))._collection.count()
    second = vs.process_pdf(str(upload))._collection.count()
    assert first > 0
    assert second == first


@needs_pdf
def test_chunks_follow_sections(chunks):
    docs, _ = chunks
    assert all(d.metadata["chunking"] == "structure" for d in docs)
    sections = {d.metadata["section"] for d in docs}
    # D.1 starts right after a full-width line; it used to be glued onto that line and lost
    assert "D. Model lifecycle management > D.1 Policies, procedures, and controls" in sections
    assert "A. Overview > A.4 Key Terms > Model Owner" in sections
    owner = next(d for d in docs if d.metadata["section"].endswith("Model Owner"))
    assert owner.page_content.startswith("[A. Overview > A.4 Key Terms > Model Owner]\n")
    assert owner.metadata["page"] == 5


@needs_pdf
def test_section_chunks_can_span_pages(chunks):
    docs, _ = chunks
    assert any(d.metadata["page_end"] > d.metadata["page"] for d in docs)


@needs_pdf
def test_page_strategy_still_available():
    docs, _ = chunk_pdf(str(PDF), MODEL, strategy="page")
    assert all(d.metadata["chunking"] == "page" and d.metadata["page"] == d.metadata["page_end"] for d in docs)
