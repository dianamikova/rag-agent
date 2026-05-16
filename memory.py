"""
memory.py — Persistent memory across sessions.

How it works:
- After each successful answer, extract key facts using the LLM
- Store facts in logs/memory.json
- Load facts at the start of each ask() and prepend to system prompt
- Facts accumulate over time, giving the agent context about the user

This is inspired by OpenClaw's memory system but simplified for
a single-user RAG agent focused on IR tasks.
"""

import json
from pathlib import Path
from typing import Optional

MEMORY_FILE = Path("logs/memory.json")
MAX_FACTS   = 20  # cap to avoid memory growing unbounded


# Load / Save

def load_memory() -> list[str]:
    """Load stored facts from memory file."""
    if not MEMORY_FILE.exists():
        return []
    try:
        data = json.loads(MEMORY_FILE.read_text())
        return data.get("facts", [])
    except Exception:
        return []

def save_memory(facts: list[str]):
    """Save facts to memory file."""
    try:
        Path("logs").mkdir(exist_ok=True)
        MEMORY_FILE.write_text(json.dumps({"facts": facts}, indent=2))
    except Exception:
        pass


# Fact extraction

def extract_facts(client, model: str, query: str, answer: str) -> list[str]:
    """
    Ask the LLM to extract memorable facts from this query/answer pair.
    Returns a list of short fact strings.
    """
    prompt = (
        f"A user asked: {query}\n\n"
        f"The answer was: {answer[:300]}\n\n"
        "Extract 1-2 short facts worth remembering about what the user is interested in. "
        "Format as a JSON array of strings. "
        "Example: [\"User is interested in RAG evaluation\", \"User asked about BM25\"]\n"
        "Reply with ONLY the JSON array, nothing else."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        facts = json.loads(raw)
        if isinstance(facts, list):
            return [str(f) for f in facts if isinstance(f, str)]
        return []
    except Exception:
        return []

def update_memory(client, model: str, query: str, answer: str):
    """Extract facts from latest query and append to memory."""
    new_facts = extract_facts(client, model, query, answer)
    if not new_facts:
        return

    existing = load_memory()
    combined = existing + new_facts

    # Keep only the most recent MAX_FACTS
    if len(combined) > MAX_FACTS:
        combined = combined[-MAX_FACTS:]

    save_memory(combined)


def enrich_query_with_memory(query: str) -> str:
    """
    Enrich a query with memory facts for better cache lookup.
    Zero API cost — pure string concatenation + local embedding.

    Example:
        query:    "How does it rank?"
        memory:   ["User researches BM25", "User interested in ranking functions"]
        enriched: "How does it rank? BM25 ranking functions"

    The enriched query is ONLY used for cache similarity matching.
    The original query is used for everything else.
    """
    facts = load_memory()
    if not facts:
        return query

    import re
    keywords = []
    for fact in facts[-3:]:
        words = [
            w for w in re.findall(r'\b[a-zA-Z]{4,}\b', fact)
            if w.lower() not in {
                "user", "asked", "about", "interested", "that", "this",
                "with", "from", "they", "them", "have", "been", "were"
            }
        ]
        keywords.extend(words[:3])

    if not keywords:
        return query

    keyword_str = " ".join(dict.fromkeys(keywords[:8]))  # max 8 keywords total
    enriched = f"{query} {keyword_str}"
    return enriched[:200]  # hard cap to keep embedding fast


# Memory context

def get_memory_context() -> str:
    """
    Return a formatted string of known facts to prepend to system prompts.
    Returns empty string if no memory exists yet.
    """
    facts = load_memory()
    if not facts:
        return ""
    facts_text = "\n".join(f"- {f}" for f in facts)
    return f"Known context about the user:\n{facts_text}\n\n"