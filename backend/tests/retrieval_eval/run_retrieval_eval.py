"""Offline evaluation of retrieval methods on top of the production chunking (no LLM calls).

Compares dense-only (the previous production setup), BM25-only, hybrid RRF, and hybrid / dense with a
cross-encoder reranker, and measures the time each query takes on this machine's CPU.

Run from backend/:
    python tests/retrieval_eval/run_retrieval_eval.py
    python tests/retrieval_eval/run_retrieval_eval.py --json tests/retrieval_eval/results_retrieval.json
"""
import argparse
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))
from app.services.chunking import chunk_pdf  # noqa: E402
from app.services.retrieval import BM25Index, CrossEncoderReranker, rrf_fuse  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from run_eval import EMBEDDING_MODEL, GOLDEN, normalise  # noqa: E402

MINILM = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE = "BAAI/bge-reranker-base"
CANDIDATES = 30


def make_methods(store, bm25):
    minilm, bge = CrossEncoderReranker(MINILM), CrossEncoderReranker(BGE)
    dense = lambda q, n=CANDIDATES: store.similarity_search(q, k=n)
    hybrid = lambda q: rrf_fuse([dense(q), bm25.search(q, CANDIDATES)])
    return {
        "dense (old production)": lambda q, k: dense(q, k),
        "bm25": lambda q, k: bm25.search(q, k),
        "hybrid_rrf": lambda q, k: hybrid(q)[:k],
        "dense+rerank_minilm": lambda q, k: minilm.rerank(q, dense(q), k),
        "hybrid+rerank_minilm": lambda q, k: minilm.rerank(q, hybrid(q), k),
        "hybrid+rerank_bge": lambda q, k: bge.rerank(q, hybrid(q), k),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--json")
    args = parser.parse_args()

    golden = json.loads(GOLDEN.read_text())
    pdf = str(BACKEND_DIR / golden["document"])
    docs, ids = chunk_pdf(pdf, EMBEDDING_MODEL)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma(client=chromadb.EphemeralClient(), collection_name=f"eval-{uuid.uuid4().hex[:8]}",
                   embedding_function=embeddings)
    store.add_documents(docs, ids=ids)
    bm25 = BM25Index(docs)

    results = []
    for name, method in make_methods(store, bm25).items():
        method("warm-up query to load models", args.k)
        ranks, timings = [], []
        for q in golden["questions"]:
            started = time.perf_counter()
            found = method(q["question"], args.k)
            timings.append((time.perf_counter() - started) * 1000)
            evidence = normalise(q["evidence"])
            ranks.append(next((i + 1 for i, d in enumerate(found) if evidence in normalise(d.page_content)), None))
        hit = lambda n: round(sum(1 for r in ranks if r and r <= n) / len(ranks), 3)
        results.append({
            "method": name,
            "hit@1": hit(1), "hit@3": hit(3), "hit@5": hit(5), "hit@8": hit(8), f"hit@{args.k}": hit(args.k),
            f"mrr@{args.k}": round(sum(1 / r for r in ranks if r) / len(ranks), 3),
            "ms_median": round(statistics.median(timings)),
            "ms_p95": round(sorted(timings)[int(len(timings) * 0.95) - 1]),
            "ranks": dict(zip([q["id"] for q in golden["questions"]], ranks)),
        })

    cols = ["method", "hit@1", "hit@3", "hit@5", "hit@8", f"hit@{args.k}", f"mrr@{args.k}", "ms_median", "ms_p95"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in results:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")

    print("\nRank per question (None = not in top-k):")
    print("  id   " + "  ".join(f"{r['method'][:20]:>20}" for r in results))
    for qid in results[0]["ranks"]:
        print(f"  {qid}  " + "  ".join(f"{str(r['ranks'][qid]):>20}" for r in results))

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
