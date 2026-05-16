"""
self_correction.py: Two-stage self-correction for generated answers.

Stage 1: NLTK token overlap check (free, no API call)
- Measures how many answer tokens appear in the retrieved context
- If overlap >= threshold --> answer is likely grounded --> return as-is
- If overlap < threshold --> answer may be hallucinating --> proceed to Stage 2

Stage 2: LLM verification and correction (one API call)
- Ask the LLM to verify the answer against the context
- If answer is supported --> return as-is
- If not --> generate a corrected answer

Why two stages?
- Most good answers pass Stage 1 instantly at zero cost
- Only suspicious answers spend tokens on Stage 2
- Mirrors the token-efficiency philosophy of the escalation ladder
"""

from nltk.tokenize import word_tokenize

# Minimum fraction of answer tokens that must appear in context
# Below this --> answer is likely hallucinating
OVERLAP_THRESHOLD = 0.4

# Stage 1: Token overlap check

def compute_overlap(answer: str, chunks: list[dict]) -> float:
    """
    Compute fraction of answer tokens that appear in retrieved context.
    Uses NLTK word_tokenize for proper tokenization.
    Returns a score between 0.0 and 1.0.
    """
    try:
        answer_tokens = set(
            t.lower() for t in word_tokenize(answer)
            if t.isalpha() and len(t) > 2
        )
        context_tokens = set(
            t.lower()
            for c in chunks
            for t in word_tokenize(c.get("text", ""))
            if t.isalpha() and len(t) > 2
        )
        if not answer_tokens:
            return 0.0
        return len(answer_tokens & context_tokens) / len(answer_tokens)
    except Exception:
        return 1.0  # assume correct if check fails


# Stage 2: LLM verification and correction

def llm_correct(client, model: str, query: str, answer: str, chunks: list[dict]) -> tuple[str, int]:
    """
    Ask the LLM to verify and correct the answer against the context.
    Returns (corrected_answer, tokens_used).
    """
    from agent import format_chunks  # avoid circular import at module level
    context = format_chunks(chunks)
    prompt = (
        f"Query: {query}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer given: {answer}\n\n"
        "Check if this answer is fully supported by the context above.\n"
        "If the answer is correct and grounded in the context, reply with: CORRECT\n"
        "If the answer contains information NOT in the context, "
        "rewrite it using ONLY the context. Be specific and detailed.\n"
        "Reply with either 'CORRECT' or the corrected answer — nothing else."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        result = resp.choices[0].message.content.strip()
        tokens = resp.usage.total_tokens if resp.usage else 100

        if "CORRECT" in result.upper() and len(result) < 200:
            return answer, tokens  # original answer is fine
        return result, tokens  # return corrected answer

    except Exception:
        return answer, 0  # fallback — return original if correction fails


# Main entry point

def self_correct(
    client,
    model: str,
    query: str,
    answer: str,
    chunks: list[dict],
) -> tuple[str, int, bool]:
    """
    Two-stage self-correction.

    Stage 1: NLTK token overlap — free
    Stage 2: LLM correction — only if Stage 1 flags the answer

    Returns:
        (final_answer, tokens_used, was_corrected)
        was_corrected=True if the answer was changed
    """
    # Stage 1 — cheap overlap check
    overlap = compute_overlap(answer, chunks)

    if overlap >= OVERLAP_THRESHOLD:
        # Answer looks grounded, no correction needed
        return answer, 0, False

    # Stage 2 — LLM correction
    corrected, tokens = llm_correct(client, model, query, answer, chunks)
    was_corrected = corrected != answer

    return corrected, tokens, was_corrected