"""Hybrid retrieval: dense vectors + BM25 keywords, merged with Reciprocal Rank Fusion, then reranked.

Kept free of app.config and database imports so it can be unit-tested and evaluated offline.

Why each step:
  * Dense search (Chroma) finds paraphrases ("outside vendors" ~ "third-party vendors") but is weak on
    exact identifiers such as "B-10" or "E-21".
  * BM25 is the opposite: exact terms and codes match, synonyms do not.
  * Reciprocal Rank Fusion merges the two ranked lists without having to calibrate their scores.
  * A cross-encoder reads the query and each candidate together, which ranks far more precisely than
    comparing two independently computed vectors, so fewer chunks need to be passed to the LLM.
  * An adaptive cutoff on the cross-encoder scores decides how many chunks to pass: only those close to
    the best one, and none at all when even the best is irrelevant (the question is not covered).
"""
import re
import threading
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# Keep codes like "b-10", "e-23", "a.4" and "3.2.1" as single tokens; their parts are indexed too.
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how in is it its of on or should that the "
    "this to was were what when where which who why will with".split()
)


def bm25_tokenize(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN.findall(text.lower()):
        if token in _STOPWORDS:
            continue
        tokens.append(token)
        if any(sep in token for sep in ".-/"):
            tokens.extend(part for part in re.split(r"[.\-/]", token) if part and part not in _STOPWORDS)
    return tokens


def _doc_key(doc: Document) -> str:
    # Content-based on purpose: depending on the langchain-chroma version, documents returned by
    # similarity_search may or may not carry their id, while those from store.get() always do.
    return f"{doc.metadata.get('source')}|{doc.page_content}"


class BM25Index:
    def __init__(self, docs: list[Document]):
        self.docs = docs
        self._bm25 = BM25Okapi([bm25_tokenize(d.page_content) for d in docs]) if docs else None

    def search(self, query: str, n: int) -> list[Document]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(bm25_tokenize(query))
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        return [self.docs[i] for i in ranked[:n] if scores[i] > 0]


def rrf_fuse(rankings: list[list[Document]], rrf_k: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion: score(doc) = sum over lists of 1 / (rrf_k + rank)."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = _doc_key(doc)
            docs.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    # Imported lazily: loading the model is slow and only needed for the first query.
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu")


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def score(self, query: str, docs: list[Document]) -> list[tuple[float, Document]]:
        """All candidates with their relevance scores, best first. Each document carries `rerank_score`."""
        if not docs:
            return []
        scores = _load_cross_encoder(self.model_name).predict([(query, d.page_content) for d in docs])
        ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)
        return [
            (float(s), Document(page_content=d.page_content, metadata={**d.metadata, "rerank_score": float(s)}, id=d.id))
            for s, d in ranked
        ]

    def rerank(self, query: str, docs: list[Document], k: int) -> list[Document]:
        return [d for _, d in self.score(query, docs)[:k]]


@dataclass(frozen=True)
class AdaptiveCutoff:
    """Choose how many reranked chunks to keep from their scores instead of a fixed k.

    Scores are on the reranker's own scale, so the values must be calibrated per reranker model
    (see tests/retrieval_eval/run_cutoff_eval.py).
      min_score: if even the best chunk scores below this, nothing is returned (question not covered);
                 chunks below it are never kept beyond min_k.
      margin:    keep chunks scoring within this distance of the best one.
      min_k:     always keep at least this many once the best chunk passes min_score.
    """
    min_score: float
    margin: float
    min_k: int = 1

    def apply(self, scored: list[tuple[float, Document]], max_k: int) -> list[Document]:
        if not scored or scored[0][0] < self.min_score:
            return []
        floor = max(self.min_score, scored[0][0] - self.margin)
        keep = [d for s, d in scored if s >= floor][:max_k]
        if len(keep) < self.min_k:
            keep = [d for _, d in scored[: min(self.min_k, max_k)]]
        return keep


class HybridRetriever:
    """Dense + BM25 candidates, fused with RRF, optionally reranked by a cross-encoder."""

    def __init__(self, store, reranker: CrossEncoderReranker | None = None, candidates: int = 30,
                 rrf_k: int = 60, use_bm25: bool = True, cutoff: AdaptiveCutoff | None = None):
        self.store = store
        self.reranker = reranker
        self.cutoff = cutoff
        self.use_bm25 = use_bm25
        self.candidates = candidates
        self.rrf_k = rrf_k
        self._bm25: BM25Index | None = None
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        """Call after documents are added or removed so the keyword index is rebuilt."""
        with self._lock:
            self._bm25 = None

    def _keyword_index(self) -> BM25Index:
        with self._lock:
            if self._bm25 is None:
                data = self.store.get(include=["documents", "metadatas"])
                docs = [
                    Document(page_content=text, metadata=meta or {}, id=doc_id)
                    for doc_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"])
                ]
                self._bm25 = BM25Index(docs)
            return self._bm25

    def search(self, query: str, k: int) -> list[Document]:
        """Up to k chunks. With a reranker and a cutoff the count adapts, and can be 0 when nothing is relevant."""
        dense = self.store.similarity_search(query, k=self.candidates)
        if self.use_bm25:
            fused = rrf_fuse([dense, self._keyword_index().search(query, self.candidates)], self.rrf_k)
        else:
            fused = dense
        if self.reranker and self.cutoff:
            return self.cutoff.apply(self.reranker.score(query, fused), max_k=k)
        if self.reranker:
            return self.reranker.rerank(query, fused, k)
        return fused[:k]
