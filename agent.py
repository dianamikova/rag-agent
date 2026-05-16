"""
agent.py — 3-level escalating RAG agent

Level 0: Semantic cache   — 0 API calls (truly free)
Level 1: Query expansion  — 1 API call (answer generation only)
Level 2: BM25 search      — 1 API call (answer generation only)
Level 3: Web search       — 2 API calls (confidence + answer)

At each level:
- Retrieve chunks using current strategy
- Score confidence (free at L1/L2 via rerank score, LLM call at L3)
- If confidence >= threshold --> generate final answer
- Else --> escalate to next level

Additional features:
- Compaction: removed (no longer needed without HyDE)
- Cache: semantically similar past queries return instantly (zero API cost)
- Memory: persistent facts about the user prepended to every system prompt
- Fallback: if answer looks like a failure, retry web search
- Re-ranking: cross-encoder re-ranking at every level
- Self-correction: NLTK overlap check + LLM correction if needed
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from retriever import Retriever, load_documents
from expander import expand_query
from websearch import web_search
from reranker import rerank
from logger import QueryLog, IterationLog, Timer, save_log
from cache import find_cached_answer, save_to_cache, check_and_invalidate_cache
from memory import update_memory, get_memory_context, enrich_query_with_memory
from self_correction import self_correct

load_dotenv()

CONFIDENCE_THRESHOLD = 0.75  # for LLM-based scoring (Level 3)
RERANK_HIGH          = 0.75  # align with CONFIDENCE_THRESHOLD
RERANK_LOW           = 0.30  # below this → clearly irrelevant → escalate free
MAX_CHUNK_TOKENS     = 500
TOP_K                = 6

FAILURE_PHRASES = [
    "insufficient", "not mentioned", "not found",
    "cannot find", "no information", "does not mention",
    "not provided", "not available", "could not find"
]


# Client

def build_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.berget.ai/v1"),
    )

def get_model() -> str:
    return os.getenv("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

# Formatting

def truncate(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> str:
    words = text.split()
    return " ".join(words[:max_tokens]) + ("..." if len(words) > max_tokens else "")

def format_chunks(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        source = c.get("source", "unknown")
        text = truncate(c.get("text", ""))
        parts.append(f"[{i}] (source: {source})\n{text}")
    return "\n\n".join(parts)


def is_failure(answer: str) -> bool:
    """Check if an answer is a failure message."""
    return any(p in answer.lower() for p in FAILURE_PHRASES)

# LLM calls

def score_confidence(client, model, query, chunks) -> tuple[float, int]:
    """Rate how well retrieved chunks answer the query via LLM. Returns (score, tokens)."""
    if not chunks:
        return 0.0, 0
    context = format_chunks(chunks)
    prompt = (
        f"Query: {query}\n\nRetrieved context:\n{context}\n\n"
        "Rate how well this context answers the query.\n"
        "Reply with ONLY a number between 0.0 and 1.0.\n"
        "0.0 = completely irrelevant. 1.0 = perfectly answers the query."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        score = float(raw.split()[0])
        score = max(0.0, min(1.0, score))
        tokens = resp.usage.total_tokens if resp.usage else 10
        return score, tokens
    except Exception:
        return 0.3, 10


def hybrid_confidence_score(client, model, query, chunks) -> tuple[float, int]:
    """
    Hybrid confidence scoring — free for clear cases, LLM only for uncertain ones.

    rerank_score > RERANK_HIGH  → clearly relevant   → return high score, 0 tokens
    rerank_score < RERANK_LOW   → clearly irrelevant  → return low score, 0 tokens
    rerank_score in between     → uncertain           → ask LLM (costs tokens)

    This minimises token usage while maintaining accuracy on borderline cases.
    """
    import math
    if not chunks:
        return 0.0, 0

    top_score = chunks[0].get("rerank_score", None)

    if top_score is None:
        # No rerank score available — fall back to LLM scoring
        return score_confidence(client, model, query, chunks)

    # Sigmoid normalisation of cross-encoder logit to 0.0-1.0
    normalised = round(1 / (1 + math.exp(-top_score / 3)), 4)

    if normalised >= RERANK_HIGH:
        # Clearly relevant — answer directly, no LLM call
        return normalised, 0

    if normalised <= RERANK_LOW:
        # Clearly irrelevant — escalate, no LLM call
        return normalised, 0

    # Uncertain — ask the LLM to confirm
    return score_confidence(client, model, query, chunks)

def generate_hyde(client, model, query) -> tuple[str, int]:
    """Generate a hypothetical document that would answer the query (HyDE)."""
    prompt = (
        "Write a short passage (3-5 sentences) that directly answers the following question. "
        "Write as if you are the document being searched for.\n\n"
        f"Question: {query}\n\nPassage:"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
        )
        text = resp.choices[0].message.content.strip()
        tokens = resp.usage.total_tokens if resp.usage else 50
        return text, tokens
    except Exception:
        return query, 0

def compact_history(client, model, query, history) -> tuple[str, int]:
    """Summarise past failed attempts to avoid context window bloat."""
    if not history:
        return "", 0
    attempts = "\n".join(
        f"- Level {h['level']} ({h['strategy']}): confidence {h['confidence']:.2f}"
        for h in history
    )
    prompt = (
        f"Original query: {query}\n\n"
        f"Previous retrieval attempts that did not find sufficient context:\n{attempts}\n\n"
        "In one sentence, summarise what information is still missing."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.0,
        )
        summary = resp.choices[0].message.content.strip()
        tokens = resp.usage.total_tokens if resp.usage else 20
        return summary, tokens
    except Exception:
        return "", 0

def generate_answer(client, model, query, chunks, compaction_note="", memory_context="") -> tuple[str, int]:
    """Generate the final answer from the best retrieved context."""
    context = format_chunks(chunks)
    system = (
        "You are a helpful assistant. Answer the question based on the context below. "
        "Be specific and detailed. Use information directly from the context. "
        "If the context does not contain the answer, say: "
        "'Could not find a reliable answer in the available sources.'"
    )
    if memory_context:
        system = memory_context + system
    user = f"Context:\n{context}\n\n"
    if compaction_note:
        user += f"Note: {compaction_note}\n\n"
    user += f"Question: {query}\n\nAnswer:"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=512,
            temperature=0.2,
        )
        answer = resp.choices[0].message.content.strip()
        tokens = resp.usage.total_tokens if resp.usage else 100
        return answer, tokens
    except Exception as e:
        return f"Error generating answer: {e}", 0

# Agent

class RAGAgent:
    def __init__(self):
        self.client = build_client()
        self.model = get_model()
        self.retriever = None
        self.last_query = ""  # last successful query for cache enrichment

    def load_docs(self, docs_dir: str = "docs") -> int:
        """Index documents from docs/ folder. Returns number of chunks indexed."""
        check_and_invalidate_cache(docs_dir)
        docs = load_documents(docs_dir)
        self.retriever = Retriever(docs)
        self.retriever.build_index()
        return len(docs)

    def ask(self, query: str) -> QueryLog:
        """Run the 4-level escalation pipeline. Returns a full QueryLog."""
        log = QueryLog(original_query=query)

        # ── Cache check ─────────────────────────────────────────────
        # Try original query first — exact/near matches always work best
        # Fall back to enriched query for paraphrase matching
        cached = find_cached_answer(query)
        if not cached:
            # Try enriched query with memory + last query context
            enriched_query = enrich_query_with_memory(query)
            if self.last_query:
                enriched_query = f"{enriched_query} {self.last_query[:50]}"
            enriched_query = enriched_query[:300]
            if enriched_query != query:
                cached = find_cached_answer(enriched_query)
        if cached:
            log.final_answer = f"[Cached] {cached['answer']}"
            log.resolved_at_level = cached.get("resolved_at_level", 0)
            similarity = cached.get("cache_similarity", 1.0)
            log.add_iteration(IterationLog(
                level=0,
                strategy=f"cache hit (similarity {similarity:.2f})",
                query_used=query,
                chunks_retrieved=0,
                confidence_before=0.0,
                confidence_after=similarity,  # show similarity not stored confidence
                tokens_used=0,
                time_ms=0.0,
                resolved=True,
            ))
            self.last_query = query
            save_log(log)
            return log

        # Memory context
        memory_context = get_memory_context()

        best_chunks: list[dict] = []
        best_confidence = 0.0
        history: list[dict] = []
        compaction_note = ""

        # Level 1: Zero API cost
        expanded = expand_query(query)
        l1_chunks: list[dict] = []

        with Timer() as t:
            if self.retriever:
                seen_ids = set()
                for eq in expanded:
                    for r in self.retriever.semantic_search(eq, top_k=TOP_K):
                        if r["id"] not in seen_ids:
                            l1_chunks.append(r)
                            seen_ids.add(r["id"])
                l1_chunks = l1_chunks[:TOP_K]
                # re-ranking
                l1_chunks = rerank(query,l1_chunks,top_k=TOP_K)

        confidence, tokens = hybrid_confidence_score(self.client, self.model, query, l1_chunks)

        log.add_iteration(IterationLog(
            level=1,
            strategy=f"query expansion ({len(expanded)} variants)",
            query_used=query,
            chunks_retrieved=len(l1_chunks),
            confidence_before=0.0,
            confidence_after=confidence,
            tokens_used=tokens,
            time_ms=t.elapsed_ms,
            resolved=confidence >= CONFIDENCE_THRESHOLD,
        ))

        if confidence >= CONFIDENCE_THRESHOLD:
            answer, t2 = generate_answer(
                self.client, self.model, query, l1_chunks,
                memory_context=memory_context
            )
            # self-correction
            answer, t3, _= self_correct(self.client, self.model, query, answer, l1_chunks)
            log.final_answer = answer
            log.total_tokens += t2 + t3
            save_log(log)
            if not is_failure(answer) and confidence >= 0.5:
                save_to_cache(query, answer, confidence, 1)
                update_memory(self.client, self.model, query, answer)
            return log

        best_chunks, best_confidence = l1_chunks, confidence
        history.append({"level": 1, "strategy": "query expansion", "confidence": confidence})

        # Level 2: BM25 keyword search (free — no API calls)
        with Timer() as t:
            l2_chunks = []
            if self.retriever:
                l2_chunks = self.retriever.bm25_search(query, top_k=TOP_K)
            l2_chunks = rerank(query, l2_chunks, top_k=TOP_K)

        confidence, tokens = hybrid_confidence_score(self.client, self.model, query, l2_chunks)

        log.add_iteration(IterationLog(
            level=2,
            strategy="BM25 keyword search",
            query_used=query,
            chunks_retrieved=len(l2_chunks),
            confidence_before=best_confidence,
            confidence_after=confidence,
            tokens_used=tokens,
            time_ms=t.elapsed_ms,
            resolved=confidence >= CONFIDENCE_THRESHOLD,
        ))

        if confidence >= CONFIDENCE_THRESHOLD:
            answer, t2 = generate_answer(
                self.client, self.model, query, l2_chunks,
                memory_context=memory_context
            )
            answer, t3, _ = self_correct(self.client, self.model, query, answer, l2_chunks)
            log.final_answer = answer
            log.total_tokens += t2 + t3
            save_log(log)
            if not is_failure(answer) and confidence >= 0.5:
                save_to_cache(query, answer, confidence, 2)
                update_memory(self.client, self.model, query, answer)
            return log

        if confidence > best_confidence:
            best_chunks, best_confidence = l2_chunks, confidence
        history.append({"level": 2, "strategy": "BM25", "confidence": confidence})

        # Level 3: Web search (DuckDuckGo — no API key needed)
        with Timer() as t:
            l3_chunks = web_search(query, max_results=TOP_K)
            l3_chunks = rerank(query, l3_chunks, top_k=TOP_K)

        confidence, tokens = score_confidence(self.client, self.model, query, l3_chunks)

        log.add_iteration(IterationLog(
            level=3,
            strategy="web search (DuckDuckGo)",
            query_used=query,
            chunks_retrieved=len(l3_chunks),
            confidence_before=best_confidence,
            confidence_after=confidence,
            tokens_used=tokens,
            time_ms=t.elapsed_ms,
            resolved=True,
        ))

        final_chunks = l3_chunks if confidence >= best_confidence else best_chunks
        answer, t2 = generate_answer(
            self.client, self.model, query, final_chunks,
            memory_context=memory_context
        )
        # self-correction
        answer, t3, _ = self_correct(self.client, self.model, query, answer, final_chunks)
        log.final_answer = answer
        log.total_tokens += t2 + t3

        # Fallback safety net
        if is_failure(log.final_answer):
            with Timer() as t:
                fallback_chunks = web_search(query, max_results=TOP_K)
                fallback_chunks = rerank(query, fallback_chunks, top_k=TOP_K)
            fallback_confidence, fallback_tokens = score_confidence(
                self.client, self.model, query, fallback_chunks
            )
            fallback_answer, fallback_t2 = generate_answer(
                self.client, self.model, query, fallback_chunks,
                memory_context=memory_context
            )
            fallback_answer, fallback_t3, _ = self_correct(
                self.client, self.model, query, fallback_answer, fallback_chunks
            )
            log.add_iteration(IterationLog(
                level=3,
                strategy="web search (fallback — answer was insufficient)",
                query_used=query,
                chunks_retrieved=len(fallback_chunks),
                confidence_before=confidence,
                confidence_after=fallback_confidence,
                tokens_used=fallback_tokens + fallback_t2 + fallback_t3,
                time_ms=t.elapsed_ms,
                resolved=True,
            ))
            if not is_failure(fallback_answer):
                log.final_answer = fallback_answer
            log.total_tokens += fallback_tokens + fallback_t2 + fallback_t3

        save_log(log)

        # Save to cache and update memory
        # Use max of local and web confidence — web answers have low best_confidence
        final_confidence = max(best_confidence, confidence)
        # Web search (level 3) accepts lower confidence threshold
        min_conf = 0.3 if log.resolved_at_level == 3 else 0.5
        if not is_failure(log.final_answer) and final_confidence >= min_conf:
            save_to_cache(query, log.final_answer, final_confidence, log.resolved_at_level)
            update_memory(self.client, self.model, query, log.final_answer)
            self.last_query = query  # update for next query's cache enrichment

        return log