"""
reranker.py — Cross-encoder re-ranking for retrieved chunks.

Why re-ranking?
- Embedding similarity measures how semantically close two texts are
- Cross-encoder measures how well a specific chunk ANSWERS a specific query
- These are different things — re-ranking corrects embedding retrieval mistakes

How it works:
- Takes a query and a list of retrieved chunks
- Scores each (query, chunk) pair using a cross-encoder model
- Returns chunks sorted by answer relevance, not embedding similarity

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Free, runs locally, no API calls
- Trained specifically for passage re-ranking on MS MARCO dataset
- Fast enough for real-time use (< 100ms for top-6 chunks)
"""

from typing import Optional

# Model singleton

_reranker_model = None

def _get_reranker():
    """Load cross-encoder model once and reuse."""
    global _reranker_model
    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except ImportError:
            print("Sentence-transformers not installed — run: pip install sentence-transformers")
        except Exception as e:
            print(f"Failed to load reranker model: {e}")
    return _reranker_model


# Re-ranking

def rerank(query: str, chunks: list[dict], top_k: Optional[int] = None) -> list[dict]:
    """
    Re-rank retrieved chunks by answer relevance using a cross-encoder.
    Args:
        query:   the user's original query
        chunks:  list of chunk dicts from retriever (must have 'text' key)
        top_k:   number of chunks to return (default: all chunks re-ranked)
    Returns:
        chunks sorted by cross-encoder relevance score (highest first)
        each chunk gets a new 'rerank_score' field added
    """
    if not chunks:
        return chunks

    model = _get_reranker()
    if model is None:
        # Graceful fallback — return chunks unchanged if model unavailable
        return chunks

    try:
        # Build (query, chunk_text) pairs for the cross-encoder
        pairs = [(query, c.get("text", "")) for c in chunks]

        # Score all pairs — cross-encoder returns a relevance score per pair
        scores = model.predict(pairs)

        # Attach rerank score to each chunk
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            scored_chunk = dict(chunk)  # don't mutate original
            scored_chunk["rerank_score"] = round(float(score), 4)
            scored_chunks.append(scored_chunk)

        # Sort by rerank score descending
        scored_chunks.sort(key=lambda c: c["rerank_score"], reverse=True)

        return scored_chunks[:top_k] if top_k else scored_chunks

    except Exception as e:
        print(f"Re-ranking failed: {e} — returning original order")
        return chunks