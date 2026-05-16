"""
retriever.py — Document indexing and retrieval (semantic + BM25)

Handles:
- Loading and chunking documents from the docs/ folder (.txt, .md, .pdf)
- Semantic search via local sentence-transformers (all-mpnet-base-v2)
- BM25 keyword search (Level 3)

Shared utilities:
- embed_texts() — used by both Retriever and cache.py
- cosine_similarity() — used by both Retriever and cache.py
"""

import math
import fitz
from pathlib import Path
from typing import Optional
from rank_bm25 import BM25Okapi


# Shared embedding model

_st_model = None

def _get_model():
    """Load sentence-transformers model once and reuse."""
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer('all-mpnet-base-v2')
        except ImportError:
            print("sentence-transformers not installed — run: pip install sentence-transformers")
    return _st_model


def embed_texts(texts: list[str]) -> Optional[list[list[float]]]:
    """
    Embed a list of texts using all-mpnet-base-v2.
    Runs locally — no API calls.
    Shared by Retriever and cache.py.
    """
    try:
        model = _get_model()
        if model is None:
            return None
        embeddings = model.encode(texts)
        return embeddings.tolist()
    except Exception as e:
#        print(f"DEBUG embed error: {e}")
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Shared by Retriever and cache.py.
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


# Document loading

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def load_documents(docs_dir: str = "docs") -> list[dict]:
    """Load all .txt, .md and .pdf files from docs/ folder."""
    docs = []
    path = Path(docs_dir)
    if not path.exists():
        return docs

    for f in path.rglob("*"):
        text = ""
        try:
            if f.suffix in (".txt", ".md"):
                text = f.read_text(encoding="utf-8", errors="ignore")
            elif f.suffix == ".pdf":
                try:
                    doc = fitz.open(str(f))
                    text = " ".join(page.get_text() for page in doc)
                    doc.close()
                except ImportError:
                    print("pymupdf not installed — run: pip install pymupdf")
                except Exception:
                    pass
            if text.strip():
                for i, chunk in enumerate(chunk_text(text)):
                    docs.append({
                        "id": f"{f.name}::{i}",
                        "source": f.name,
                        "text": chunk,
                    })
        except Exception:
            pass

    return docs

# Retriever

class Retriever:
    """
    Dual retriever: semantic (cosine similarity via local embeddings) + BM25 keyword.
    Embeddings use all-mpnet-base-v2 via sentence-transformers — no API calls needed.
    Falls back to BM25 if embeddings are unavailable.
    """

    def __init__(self, docs: list[dict]):
        self.docs = docs
        self._embeddings: Optional[list[list[float]]] = None

        # BM25 index always built, no API cost
        tokenized = [d["text"].lower().split() for d in docs]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def build_index(self) -> bool:
        """Pre-compute embeddings for all chunks. Returns True if successful."""
        if not self.docs:
            return False
        texts = [d["text"] for d in self.docs]
        self._embeddings = embed_texts(texts)
        return self._embeddings is not None

    def semantic_search(self, query: str, top_k: int = 4) -> list[dict]:
        """Semantic search using embeddings. Falls back to BM25 if unavailable."""
        if self._embeddings is None:
            return self.bm25_search(query, top_k)
        q_emb = embed_texts([query])
        if q_emb is None:
            return self.bm25_search(query, top_k)
        scores = [cosine_similarity(q_emb[0], e) for e in self._embeddings]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            {**self.docs[i], "score": round(scores[i], 4), "method": "semantic"}
            for i in ranked[:top_k]
        ]

    def bm25_search(self, query: str, top_k: int = 4) -> list[dict]:
        """BM25 keyword search — no API calls."""
        if self.bm25 is None or not self.docs:
            return []
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            {**self.docs[i], "score": round(float(scores[i]), 4), "method": "bm25"}
            for i in ranked[:top_k]
            if scores[i] > 0
        ]