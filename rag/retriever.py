"""Hybrid retrieval: dense vectors (FAISS) + keyword search (BM25), fused with RRF.

Dense search catches paraphrases ("time off" ~ "leave"); BM25 catches exact terms
(policy numbers, names, acronyms). Reciprocal Rank Fusion merges both rankings
without having to calibrate their very different score scales.
"""

import re
from dataclasses import dataclass

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .chunker import Chunk
from .providers import Embedder

RRF_K = 60  # standard constant from Cormack et al., 2009


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # fused RRF score
    dense_rank: int | None
    keyword_rank: int | None


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        if not chunks:
            raise ValueError("No text could be extracted from the uploaded documents.")
        self.chunks = chunks
        self.embedder = embedder

        vectors = _normalize(embedder.embed_documents([c.text for c in chunks]))
        self.index = faiss.IndexFlatIP(vectors.shape[1])  # inner product on unit vectors = cosine
        self.index.add(vectors)

        self.bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        pool = min(len(self.chunks), max(k * 4, 20))

        q = _normalize(self.embedder.embed_query(query).reshape(1, -1))
        _, dense_ids = self.index.search(q, pool)
        dense_ranking = [int(i) for i in dense_ids[0] if i != -1]

        bm25_scores = self.bm25.get_scores(_tokenize(query))
        keyword_ranking = [int(i) for i in np.argsort(-bm25_scores)[:pool] if bm25_scores[i] > 0]

        fused: dict[int, float] = {}
        for ranking in (dense_ranking, keyword_ranking):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [
            SearchResult(
                chunk=self.chunks[i],
                score=fused[i],
                dense_rank=dense_ranking.index(i) + 1 if i in dense_ranking else None,
                keyword_rank=keyword_ranking.index(i) + 1 if i in keyword_ranking else None,
            )
            for i in top
        ]
