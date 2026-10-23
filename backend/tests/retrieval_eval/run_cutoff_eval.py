"""Calibrate the adaptive cutoff on reranker scores (offline, no LLM calls).

Uses three kinds of questions from e23_golden.json:
  questions  - one evidence phrase each; must still be found
  multi      - several evidence phrases in different sections; coverage should not drop
  negatives  - not answered by the document; ideally nothing is returned

Every question is reranked once; the cutoff parameters are then grid-searched over the cached scores.
Run from backend/:
    python tests/retrieval_eval/run_cutoff_eval.py --json tests/retrieval_eval/results_cutoff.json
"""
import argparse
import itertools
import json
import statistics
import sys
import uuid
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))
from app.services.chunking import chunk_pdf  # noqa: E402
from app.services.retrieval import AdaptiveCutoff, BM25Index, CrossEncoderReranker, rrf_fuse  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from run_eval import EMBEDDING_MODEL, GOLDEN, normalise  # noqa: E402

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 30

GRID = {
    "min_score": [x / 2 for x in range(-16, 13)],    # -8.0 .. 6.0
    "margin": [1, 2, 3, 4, 5, 6, 8, 10, 15],
    "min_k": [1, 2, 3],
    "max_k": [5, 8, 10],
}


def covered(docs, evidence):
    return [any(normalise(e) in normalise(d.page_content) for d in docs) for e in evidence]


def evaluate(selector, cached):
    single = [all(covered(selector(s), [q["evidence"]])) for q, s in cached["single"]]
    multi = [statistics.mean(covered(selector(s), q["evidence"])) for q, s in cached["multi"]]
    refused_pos = [len(selector(s)) == 0 for _, s in cached["single"] + cached["multi"]]
    refused_neg = [len(selector(s)) == 0 for _, s in cached["negatives"]]
    return {
        "single_recall": round(statistics.mean(single), 3),
        "multi_evidence_recall": round(statistics.mean(multi), 3),
        "false_refusals": sum(refused_pos),
        "negatives_refused": f"{sum(refused_neg)}/{len(refused_neg)}",
        "neg_refusal_rate": round(statistics.mean(refused_neg), 3),
        "avg_k_single": round(statistics.mean(len(selector(s)) for _, s in cached["single"]), 2),
        "avg_k_multi": round(statistics.mean(len(selector(s)) for _, s in cached["multi"]), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    args = parser.parse_args()

    golden = json.loads(GOLDEN.read_text())
    docs, ids = chunk_pdf(str(BACKEND_DIR / golden["document"]), EMBEDDING_MODEL)
    store = Chroma(client=chromadb.EphemeralClient(), collection_name=f"eval-{uuid.uuid4().hex[:8]}",
                   embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))
    store.add_documents(docs, ids=ids)
    bm25, reranker = BM25Index(docs), CrossEncoderReranker(RERANK_MODEL)

    def scored(question):
        fused = rrf_fuse([store.similarity_search(question, k=CANDIDATES), bm25.search(question, CANDIDATES)])
        return reranker.score(question, fused)

    cached = {kind: [(q, scored(q["question"])) for q in golden[key]]
              for kind, key in (("single", "questions"), ("multi", "multi"), ("negatives", "negatives"))}

    # How well do top scores separate answerable from unanswerable questions?
    pos_top = sorted((s[0][0], q["id"]) for q, s in cached["single"] + cached["multi"])
    neg_top = sorted(((s[0][0], q["id"]) for q, s in cached["negatives"]), reverse=True)
    print("Lowest top scores of answerable questions:", [(round(v, 2), i) for v, i in pos_top[:5]])
    print("Highest top scores of unanswerable ones:  ", [(round(v, 2), i) for v, i in neg_top[:5]])
    print()

    rows = []
    for k in (5, 10):
        rows.append({"config": f"fixed k={k}", **evaluate(lambda s, k=k: [d for _, d in s[:k]], cached)})
    fixed5 = rows[0]

    candidates = []
    for min_score, margin, min_k, max_k in itertools.product(*GRID.values()):
        cut = AdaptiveCutoff(min_score=min_score, margin=margin, min_k=min_k)
        m = evaluate(lambda s, cut=cut, max_k=max_k: cut.apply(s, max_k), cached)
        feasible = (m["false_refusals"] == 0 and m["single_recall"] >= fixed5["single_recall"]
                    and m["multi_evidence_recall"] >= fixed5["multi_evidence_recall"])
        candidates.append(({"min_score": min_score, "margin": margin, "min_k": min_k, "max_k": max_k}, m, feasible))

    feasible = [c for c in candidates if c[2]]
    feasible.sort(key=lambda c: (-c[1]["neg_refusal_rate"], c[1]["avg_k_single"], -c[1]["multi_evidence_recall"]))

    cols = ["config", "single_recall", "multi_evidence_recall", "false_refusals", "negatives_refused",
            "avg_k_single", "avg_k_multi"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows + [{"config": json.dumps(p), **m} for p, m, _ in feasible[:12]]:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    print(f"\n{len(feasible)} of {len(candidates)} configurations meet the constraints.")

    if feasible:
        best = feasible[0][0]
        same = [p["min_score"] for p, m, _ in feasible
                if (p["margin"], p["min_k"], p["max_k"]) == (best["margin"], best["min_k"], best["max_k"])
                and m["neg_refusal_rate"] == feasible[0][1]["neg_refusal_rate"]]
        print(f"For margin={best['margin']}, min_k={best['min_k']}, max_k={best['max_k']}, min_score values "
              f"with the same result: {min(same)} .. {max(same)}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "baselines": rows,
            "feasible_top": [{"params": p, **m} for p, m, _ in feasible[:30]],
            "top_scores": {"answerable": pos_top, "unanswerable": neg_top},
        }, indent=2))


if __name__ == "__main__":
    main()
