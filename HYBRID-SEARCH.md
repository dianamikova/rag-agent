## 🚀 Hybrid Search with Reciprocal Rank Fusion (Proposed Improvement)

The current pipeline runs **dense search** (semantic embeddings with
`all-mpnet-base-v2` + cosine similarity) at Level 1 and **BM25** keyword search
at a later level. The next improvement is **hybrid search**: run both on the same
query and fuse their results into one ranking.

**Why hybrid wins:** dense search captures meaning and paraphrase, while BM25
captures exact terms, rare words, IDs, and acronyms that embeddings tend to blur.
Combining them typically retrieves better than either method alone.

### Fusion strategy: Reciprocal Rank Fusion (RRF)

RRF is the recommended default because it uses each document's *rank position*
rather than its raw score. BM25 scores and cosine similarities live on completely
different scales, and RRF sidesteps that problem without any score normalization.

Each document is scored as the sum of `1 / (k + rank)` across both methods (k ≈ 60):

```python
def hybrid_search(self, query: str, top_k: int = 4, k_rrf: int = 60,
                  pool: int = 20) -> list[dict]:
    """
    Hybrid retrieval: fuse dense (semantic) + BM25 rankings via
    Reciprocal Rank Fusion. Robust to the two methods' different score scales.
    """
    dense = self.semantic_search(query, top_k=pool)
    sparse = self.bm25_search(query, top_k=pool)

    fused: dict[str, dict] = {}
    for ranked in (dense, sparse):
        for rank, doc in enumerate(ranked):
            entry = fused.setdefault(doc["id"], {**doc, "rrf": 0.0})
            entry["rrf"] += 1.0 / (k_rrf + rank)

    results = sorted(fused.values(), key=lambda d: d["rrf"], reverse=True)
    for r in results:
        r["method"] = "hybrid"
        r["score"] = round(r["rrf"], 6)
    return results[:top_k]
```

### Implementation notes

- **Pull a wide pool first.** Retrieve ca 20 candidates from each method *before*
  fusing, then trim to `top_k` - fusing only the top 4 gives RRF too little to work with.
- **Chain with the existing re-ranker.** Hybrid retrieve → RRF fuse → feed the fused
  top-N into the current cross-encoder `rerank()`. This gives hybrid recall plus
  precise final ordering.
- **Guard the embedding fallback.** `semantic_search` falls back to BM25 when
  embeddings aren't built, which would double-count BM25 inside hybrid. Ensure
  `build_index()` has succeeded before calling `hybrid_search`.
- **Where to slot it:** making Level 1 hybrid is the highest-leverage single change,
  since most queries resolve there.

### Further scaling (optional)

Dense search here is brute-force cosine over all chunks — fine at ~39 chunks, slow at
thousands. At larger scale, swap in a FAISS or hnswlib index for speed, or move to a
vector store like Qdrant or Weaviate that supports hybrid search natively.
