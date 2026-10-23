"""Offline retrieval evaluation for chunking strategies (no LLM calls, no API key, no Docker services).

Builds an in-memory Chroma collection per strategy with the production embedding model, asks every
question in e23_golden.json, and reports where the chunk containing the evidence phrase ranks.

Run from backend/:
    python tests/retrieval_eval/run_eval.py
    python tests/retrieval_eval/run_eval.py --k 15 --json tests/retrieval_eval/results.json
"""
import argparse
import json
import re
import statistics
import sys
import uuid
from pathlib import Path

import chromadb
import numpy as np
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))
from app.services.chunking import _load_model, _token_splitter, chunk_pdf, load_pdf_pages  # noqa: E402

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # same default as app/config.py
GOLDEN = Path(__file__).with_name("e23_golden.json")


def normalise(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def baseline_chunks(pdf: str):
    """The original production pipeline: PyPDFLoader (0-based pages) + 800/150 character splitter."""
    docs = PyPDFLoader(pdf).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", " ", ""])
    return splitter.split_documents(docs), None


def semantic_chunks(pdf: str, percentile: float = 95, cap: bool = False):
    """Semantic chunking (evaluation only): cut where adjacent sentences stop being similar.

    Each sentence is embedded together with its neighbours; a cut is made wherever the cosine distance
    between consecutive windows is above the given percentile. With cap=True, groups longer than the
    embedding model's limit are split again by tokens so they are not truncated.
    """
    model = _load_model(EMBEDDING_MODEL)
    limit = model.max_seq_length - 2
    sentences = []
    for page in load_pdf_pages(pdf):
        for para in page["text"].split("\n"):
            for sent in re.split(r"(?<=[.!?;:])\s+(?=[A-Z(\"'])", para.strip()):
                if sent.strip():
                    sentences.append((sent.strip(), page["page"]))

    windows = [" ".join(t for t, _ in sentences[max(0, i - 1): i + 2]) for i in range(len(sentences))]
    emb = model.encode(windows, normalize_embeddings=True)
    distances = 1 - (emb[:-1] * emb[1:]).sum(axis=1)
    threshold = np.percentile(distances, percentile)

    groups = [[]]
    for i, item in enumerate(sentences):
        groups[-1].append(item)
        if i < len(distances) and distances[i] > threshold:
            groups.append([])

    splitter = _token_splitter(EMBEDDING_MODEL, 200, 40)
    docs = []
    for group in filter(None, groups):
        text = " ".join(t for t, _ in group)
        pages = [pg for _, pg in group]
        too_long = len(model.tokenizer.tokenize(text)) > limit
        for piece in (splitter.split_text(text) if cap and too_long else [text]):
            docs.append(Document(page_content=piece, metadata={"page": pages[0], "page_end": pages[-1]}))
    return docs, None


STRATEGIES = {
    "baseline": baseline_chunks,
    "page_tok200": lambda pdf: chunk_pdf(pdf, EMBEDDING_MODEL, strategy="page"),
    "structure_tok200": lambda pdf: chunk_pdf(pdf, EMBEDDING_MODEL, strategy="structure"),
    "structure_noprefix": lambda pdf: chunk_pdf(pdf, EMBEDDING_MODEL, strategy="structure", heading_prefix=False),
    "semantic_p95": lambda pdf: semantic_chunks(pdf),
    "semantic_p95_capped": lambda pdf: semantic_chunks(pdf, cap=True),
}


def build_store(client, embeddings, docs, ids, ingest_twice=False):
    store = Chroma(client=client, collection_name=f"eval-{uuid.uuid4().hex[:8]}", embedding_function=embeddings)
    for _ in range(2 if ingest_twice else 1):
        store.add_documents(docs, ids=ids)  # ids=None -> random ids, like the original from_documents call
    return store


def evaluate(name, pdf, golden, embeddings, tokenizer, token_limit, k):
    docs, ids = STRATEGIES[name](pdf)
    client = chromadb.EphemeralClient()
    store = build_store(client, embeddings, docs, ids)
    twice = build_store(client, embeddings, docs, ids, ingest_twice=True)

    token_lens = [len(tokenizer.tokenize(d.page_content)) for d in docs]
    ranks, cite_ok, context_chars, rows = [], [], [], []
    for q in golden["questions"]:
        results = store.similarity_search(q["question"], k=k)
        context_chars.append(sum(len(d.page_content) for d in results))
        evidence = normalise(q["evidence"])
        rank = next((i + 1 for i, d in enumerate(results) if evidence in normalise(d.page_content)), None)
        ranks.append(rank)
        if rank is not None:
            meta = results[rank - 1].metadata
            cite_ok.append(meta.get("page", -1) <= q["page"] <= meta.get("page_end", meta.get("page", -1)))
        rows.append({"id": q["id"], "rank": rank})

    hits = lambda n: sum(1 for r in ranks if r is not None and r <= n) / len(ranks)
    return {
        "strategy": name,
        "chunks": len(docs),
        "tokens_mean": round(statistics.mean(token_lens)),
        "tokens_max": max(token_lens),
        "truncated_chunks": sum(1 for t in token_lens if t > token_limit),
        "stored_after_two_uploads": twice._collection.count(),
        "hit@1": round(hits(1), 3),
        "hit@5": round(hits(5), 3),
        f"hit@{k}": round(hits(k), 3),
        f"mrr@{k}": round(sum(1 / r for r in ranks if r) / len(ranks), 3),
        "citation_page_correct": f"{sum(cite_ok)}/{len(cite_ok)}",
        f"context_chars_top{k}": round(statistics.mean(context_chars)),
        "per_question": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=15, help="top-k, production uses 15")
    parser.add_argument("--json", help="write full results to this file")
    parser.add_argument("--pdf", help="override the PDF path in e23_golden.json")
    args = parser.parse_args()

    golden = json.loads(GOLDEN.read_text())
    pdf = args.pdf or str(BACKEND_DIR / golden["document"])

    # Sanity check: every evidence phrase must exist in the document at all.
    full_text = normalise(" ".join(p["text"] for p in load_pdf_pages(pdf)))
    missing = [q["id"] for q in golden["questions"] if normalise(q["evidence"]) not in full_text]
    if missing:
        sys.exit(f"Evidence phrases not found in the PDF: {missing}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    st = SentenceTransformer(f"sentence-transformers/{EMBEDDING_MODEL}", device="cpu")
    token_limit = st.max_seq_length - 2

    results = [evaluate(n, pdf, golden, embeddings, st.tokenizer, token_limit, args.k) for n in STRATEGIES]

    cols = ["strategy", "chunks", "tokens_mean", "tokens_max", "truncated_chunks", "stored_after_two_uploads",
            "hit@1", "hit@5", f"hit@{args.k}", f"mrr@{args.k}", "citation_page_correct", f"context_chars_top{args.k}"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in results:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")

    base = {row["id"]: row["rank"] for row in results[0]["per_question"]}
    print("\nPer-question rank (None = not in top-k):")
    print("  id   " + "  ".join(f"{r['strategy'][:18]:>18}" for r in results))
    for i, qid in enumerate(base):
        print(f"  {qid}  " + "  ".join(f"{str(r['per_question'][i]['rank']):>18}" for r in results))

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
