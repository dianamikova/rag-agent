# Token-Efficient RAG Agent with Escalating Retrieval and Context-Aware Caching

**Course:** Information Retrieval | **GitHub:** [dianamikova/rag-agent]
 
Github link: https://github.com/dianamikova/rag-agent
Video link: https://drive.google.com/file/d/1PUSQvuvORo2YxYNBjWq_Dy5v27DWyZz2/view?usp=sharing
---

## 1. Motivation

Large language models are powerful but suffer from two fundamental limitations: static parametric knowledge and a tendency to hallucinate when operating beyond their training data. Retrieval-Augmented Generation (RAG) addresses both by grounding responses in externally retrieved evidence (Lewis et al., 2020). However, most RAG systems retrieve once and generate once — a fixed pipeline that fails silently when retrieval quality is poor. Recent work such as FAIR-RAG (Asl et al., 2025) and Self-RAG (Asai et al., 2023) introduced iterative and self-reflective retrieval, but these systems share a critical practical limitation: they do not account for the token cost of each retrieval iteration, relying on fixed strategies or expensive multi-model pipelines that are impractical in resource-constrained settings.

This experiment was motivated by a simple question: *Can an agent retrieve reliably while spending as few tokens as possible?* The hypothesis was that most queries can be resolved "cheaply" - either from a semantic cache, from local documents using free methods, or from keyword search - and only the hardest queries should incur the full cost of LLM-based scoring and web search. This cost-aware escalation is the core contribution of this work.

---

## 2. System Design

The agent implements a four-level escalation ladder, stopping as soon as confidence is sufficient:

**Level 0: Semantic cache (0 API calls).** 
Before any retrieval, the query is embedded and compared against pre-stored embeddings of previously answered queries. Embeddings are pre-computed at save time, reducing lookup cost from O(N) to O(1). To improve hit rates, the query is enriched with keywords from persistent memory facts and the previous query context. A 0.10 cosine similarity improvement threshold prevents enriched variants from returning unrelated cached answers. Cache is invalidated by document fingerprinting — wiped only when a document is removed, modified or via UI request (streamlit). The similarity threshold is set to 0.75.

**Level 1: Semantic search with query expansion (1 API call).** The query is expanded using WordNet synonym substitution, PorterStemmer morphological variants, and POS-tag-based compound decomposition — all free local NLTK operations. Retrieved chunks are re-ranked using a cross-encoder (`ms-marco-MiniLM-L-6-v2`, local, no API cost). Confidence is assessed using a **hybrid scoring** approach: if the top re-ranked chunk scores above 0.75 (normalised cross-encoder logit via sigmoid), confidence is inferred for free; only borderline scores (0.30–0.75) trigger an LLM confidence call. This eliminates API costs for clearly relevant or clearly irrelevant retrievals, spending tokens only on uncertain cases.

**Level 2: BM25 keyword search (1 API call).** If semantic search fails, the agent switches to BM25 — a complementary method that succeeds where semantic search fails. The same hybrid confidence scoring applies, keeping the cost at zero for clear cases. BM25 and semantic search have complementary failure modes: semantic search struggles with domain-specific acronyms while BM25 struggles with synonyms, making the two levels genuinely additive.

**Level 3: Web search (2 API calls).** When local documents fail to provide a sufficient answer, the agent falls back to DuckDuckGo web search, which requires no API key. In this case, LLM confidence scoring is always used since web result quality is unpredictable. A fallback safety net detects failure phrases in the generated answer (*"could not find"*, *"not mentioned"*) and retries web search if the primary response is empty.

Following answer generation, a **two-stage self-correction** module verifies groundedness: Stage 1 computes NLTK token overlap between the answer and retrieved chunks (free); only if overlap falls below 0.4 does Stage 2 invoke the LLM to verify and correct the answer.

---

## 3. Token Efficiency

The primary engineering concern throughout was minimising token cost without sacrificing answer quality. Using local sentence-transformers (`all-mpnet-base-v2`) for embeddings eliminated OpenAI embedding API calls entirely. Pre-computing and storing cache embeddings at save time reduced lookup cost from O(N) embeddings per query to O(1). The hybrid confidence scorer eliminated LLM calls for the majority of Level 1 and Level 2 queries. At Berget.AI pricing (€0.90/M tokens, Llama 3.3 70B), a typical query costs approximately €0.00054 — less than a fraction of a cent — with cache hits costing nothing.

---

## 4. Memory and Cache

**Persistent memory** stores 1–2 LLM-extracted facts after each successful answer, capped at 20 facts. These facts are prepended to the system prompt to personalize responses across sessions. Memory also contributes to cache enrichment, improving hit rates for paraphrased follow-up questions. **Cache invalidation** is handled by document fingerprinting: each document's line count is stored.

The system tracks the previous query in session to support cache enrichment — appending its text to the current query before lookup, which helps resolve follow-up questions referencing earlier topics. Persistent memory contributes similarly, supplying topic keywords extracted from past interactions. A more ambitious approach — pronoun resolution, where *"How does it work?"* would be rewritten to *"How does BM25 work?"* using the previous query's subject — was implemented and tested but ultimately removed. Reliable subject extraction from arbitrary natural language queries proved inconsistent in practice, causing unrelated cached answers to be returned.

---

## 5. Future Work

Several improvements remain. The hybrid confidence threshold (0.30–0.75) was set heuristically and would benefit from calibration against a labelled query set. Cache enrichment uses keyword extraction from memory facts, which works well for topically related follow-ups but struggles with cross-topic pronoun references. A learned query expansion model fine-tuned on the target domain would likely outperform the generic WordNet synonyms used here. The system also currently supports a single user and a single document collection; extending to multi-user scenarios would require partitioned caching and memory namespacing.

Beyond retrieval quality, the system has several operational gaps. Too much responsibility sits inside agent.py, which mixes control flow, prompt logic, thresholds, cache behaviour, and fallback rules — a cleaner separation of concerns would improve maintainability. The most notable production gap could be the absence of tests and evaluation infrastructure; in a RAG system, most failures come from retrieval and decision quality, making automated evaluation essential for any further development.

---

## 6. Reflection on AI-Assisted Development

This project was developed in collaboration with Claude (Anthropic). The AI was effective at generating modular, well-documented code and connecting design decisions to the academic literature. However, it required consistent human oversight, critical evaluation and code logic correction. Several suggested implementations introduced subtle bugs; misaligned confidence thresholds, double-counted tokens, incorrect cache similarity display, unnecessarily hardcoded parts in code. The AI also made outright incorrect claims; it repeatedly asserted that certain levels would cost zero tokens, when in practice confidence scoring required LLM calls at every level, producing poor results when removed. It occasionally suggested overcomplicated solutions where simpler ones existed, such as architecting a custom stemmer instead of using the NLTK PorterStemmer that was already available. It also referenced models not available through the Berget.AI provider. Perhaps most notably, the AI consistently anchored its ambition to the minimum required rather than pushing toward a stronger system — the more interesting design decisions came from questioning its suggestions rather than accepting them. The key lesson was that AI tools accelerate implementation but do not replace the need to challenge every design decision.

---

## References

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS. arXiv:2005.11401
- Asai, A. et al. (2023). *Self-RAG*. ICLR 2024. arXiv:2310.11511
- Jiang, Z. et al. (2023). *FLARE*. EMNLP 2023. arXiv:2305.06983
- Gao, L. et al. (2023). *HyDE*. ACL 2023. arXiv:2212.10496
- Es, S. et al. (2024). *RAGAs*. EACL 2024.
- Asl, M. A. et al. (2025). *FAIR-RAG*. arXiv:2510.22344