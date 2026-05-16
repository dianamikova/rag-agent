"""
cache.py — Semantic query cache with pre-computed embeddings.

Performance fix:
- Old approach: re-embed ALL cached queries on every lookup → slow
- New approach: embed query once when saving, store embedding in cache
                On lookup: embed only the new query (1 embedding) and
                compare against stored embeddings → fast

Cache invalidation logic:
- Each document is fingerprinted by name + line count
- New documents added → cache KEPT (old answers still valid)
- Existing document removed → cache CLEARED
- Existing document content changed → cache CLEARED
- Nothing changed → cache KEPT

Semantic matching:
- Incoming query embedded once locally (free)
- Compared against stored embeddings via cosine similarity
- If similarity > SIMILARITY_THRESHOLD → return cached answer instantly
"""

import json
import fitz
import math
from pathlib import Path
from typing import Optional
from retriever import embed_texts as _embed, cosine_similarity as _cosine


CACHE_FILE        = Path("logs/cache.jsonl")
FINGERPRINT_FILE  = Path("logs/doc_fingerprints.json")
SIMILARITY_THRESHOLD = 0.75


# ── Fingerprinting ─────────────────────────────────────────────────────

def compute_doc_fingerprints(docs_dir: str = "docs") -> dict[str, int]:
    """Return {filename: line_count} for all docs."""
    path = Path(docs_dir)
    fingerprints = {}
    if not path.exists():
        return fingerprints
    for f in path.rglob("*"):
        if not f.is_file():
            continue
        try:
            if f.suffix == ".pdf":
                try:
                    doc = fitz.open(str(f))
                    lines = sum(len(page.get_text().splitlines()) for page in doc)
                    doc.close()
                except Exception:
                    lines = 0
            elif f.suffix in (".txt", ".md"):
                lines = len(f.read_text(errors="ignore").splitlines())
            else:
                continue
            fingerprints[f.name] = lines
        except Exception:
            fingerprints[f.name] = 0
    return fingerprints


def load_fingerprints() -> dict[str, int]:
    if not FINGERPRINT_FILE.exists():
        return {}
    try:
        return json.loads(FINGERPRINT_FILE.read_text())
    except Exception:
        return {}


def save_fingerprints(fingerprints: dict[str, int]):
    try:
        Path("logs").mkdir(exist_ok=True)
        FINGERPRINT_FILE.write_text(json.dumps(fingerprints, indent=2))
    except Exception:
        pass


def wipe_cache():
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()


def check_and_invalidate_cache(docs_dir: str = "docs") -> bool:
    """
    Compare current doc fingerprints with stored ones.
    Wipe cache if any doc removed or changed.
    Keep cache if only new docs added.
    """
    Path("logs").mkdir(exist_ok=True)
    current = compute_doc_fingerprints(docs_dir)
    previous = load_fingerprints()

    if not previous:
        save_fingerprints(current)
        return False

    for name, lines in previous.items():
        if name not in current:
            wipe_cache()
            save_fingerprints(current)
            print(f"'{name}' removed — cache cleared.")
            return True
        if current[name] != lines:
            wipe_cache()
            save_fingerprints(current)
            print(f"'{name}' changed — cache cleared.")
            return True

    new_docs = set(current.keys()) - set(previous.keys())
    if new_docs:
        save_fingerprints(current)
        print(f"{len(new_docs)} new doc(s) added — cache kept.")

    return False


# ── Cache read / write ─────────────────────────────────────────────────

def load_cache() -> list[dict]:
    """Load all cached entries including stored embeddings."""
    if not CACHE_FILE.exists():
        return []
    entries = []
    with open(CACHE_FILE) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    return entries


def save_to_cache(query: str, answer: str, confidence: float, level: int):
    """
    Save query/answer to cache WITH pre-computed embedding.
    Embedding computed once here — never again during lookup.
    """
    try:
        Path("logs").mkdir(exist_ok=True)

        # Compute and store embedding at save time
        embedding = _embed([query])
        embedding_list = embedding[0] if embedding else None

        entry = {
            "query": query,
            "embedding": embedding_list,  # stored for fast lookup
            "answer": answer,
            "confidence": confidence,
            "resolved_at_level": level,
        }
        with open(CACHE_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# ── Semantic lookup ────────────────────────────────────────────────────

def find_cached_answer(query: str) -> Optional[dict]:
    """
    Fast semantic cache lookup.

    Steps:
    1. Embed the incoming query ONCE (local, free)
    2. Compare against pre-stored embeddings (dot product only, no re-embedding)
    3. Return best match above SIMILARITY_THRESHOLD

    Cost: 1 embedding regardless of cache size.
    """
    entries = load_cache()
    if not entries:
        return None

    # Embed only the new query — O(1) regardless of cache size
    q_emb = _embed([query])
    if q_emb is None:
        return None
    q_vec = q_emb[0]

    best_score = 0.0
    best_entry = None

    for entry in entries:
        stored_emb = entry.get("embedding")
        if stored_emb is None:
            # Legacy entry without embedding — skip
            continue
        score = _cosine(q_vec, stored_emb)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_score >= SIMILARITY_THRESHOLD:
        result = {k: v for k, v in best_entry.items() if k != "embedding"}
        result["cache_similarity"] = round(best_score, 4)
        return result

    return None