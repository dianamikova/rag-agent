"""
websearch.py — Level 4 web search using DuckDuckGo (no API key required)
"""
 
from ddgs import DDGS

def web_search(query: str, max_results: int = 4) -> list[dict]:
    """
    Search the web via DuckDuckGo. Returns list of result dicts:
    { text, source, score, method }
    No API key needed.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "text": r.get("body", ""),
                "source": r.get("href", "web"),
                "score": 1.0,  # web results don't have a relevance score
                "method": "web_search",
                "id": f"web::{i}",
            }
            for i, r in enumerate(results)
            if r.get("body")
        ]
    except Exception:
        return []
 