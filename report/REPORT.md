# Token-Efficient RAG Agent with Escalating Retrieval and Context-Aware Caching

**Course:** Information Retrieval | **GitHub:** [your-username/rag-agent]

---

## 1. Motivation

Large language models are powerful but suffer from two fundamental limitations: static parametric knowledge and a tendency to hallucinate when operating beyond their training data. Retrieval-Augmented Generation (RAG) addresses both by grounding responses in externally retrieved evidence (Lewis et al., 2020). However, most RAG systems retrieve once and generate once — a fixed pipeline that fails silently when retrieval quality is poor. Recent work such as FAIR-RAG (Asl et al., 2025) and Self-RAG (Asai et al., 2023) introduced iterative and self-reflective retrieval, but these systems share a critical practical limitation: they do not account for the token cost of each retrieval iteration, relying on fixed strategies or expensive multi-model pipelines that are impractical in resource-constrained settings.

This experiment was motivated by a simple question: *Can an agent retrieve reliably while spending as few tokens as possible?* The hypothesis was that most queries can be resolved "cheaply" - either from a semantic cache, from local documents using free methods, or from keyword search - and only the hardest queries should incur the full cost of LLM-based scoring and web search. This cost-aware escalation is the core contribution of this work.

---

## 2. System Design

The agent implements a four-level escalation ladder, stopping as soon as confidence is sufficient:

**Level 0: Semantic cache (0 API calls).** 
Before any retrieval, the agent embeds the incoming query and compares it against pre-stored embeddings of previously answered queries using cosine similarity. To improve hit rate, the query is enriched with two sources of context at zero cost: keywords extracted from persistent memory facts, and the full text of the previous query for pronoun resolution (e.g., *"How does it work?"* after *"What is BM25?"* correctly resolves to the BM25 context). The cache threshold is 0.75 cosine similarity.

**Level 1: Semantic search with query expansion (1 API call).** The query is expanded using WordNet synonym substitution, PorterStemmer morphological variants, and POS-tag-based compound decomposition — all free local NLTK operations. Retrieved chunks are re-ranked using a cross-encoder (`ms-marco-MiniLM-L-6-v2`, local, no API cost). Confidence is assessed using a **hybrid scoring** approach: if the top re-ranked chunk scores above 0.75 (normalised cross-encoder logit via sigmoid), confidence is inferred for free; only borderline scores (0.30–0.75) trigger an LLM confidence call. This eliminates API costs for clearly relevant or clearly irrelevant retrievals, spending tokens only on uncertain cases.

**Level 2: BM25 keyword search (1 API call).** If semantic search fails, the agent switches to BM25 — a complementary method that succeeds where semantic search fails, namely on exact technical terminology. The same hybrid confidence scoring applies, keeping the cost at zero for clear cases. BM25 and semantic search have complementary failure modes: semantic search struggles with domain-specific acronyms while BM25 struggles with synonyms, making the two levels genuinely additive.

**Level 3: Web search (2 API calls).** Only when local documents cannot answer the query does the agent fall back to DuckDuckGo web search (no API key required). Here LLM confidence scoring is always used since web result quality is unpredictable. A fallback safety net detects failure phrases in the generated answer (*"could not find"*, *"not mentioned"*) and retries web search if the primary response is empty.

Following answer generation, a **two-stage self-correction** module verifies groundedness: Stage 1 computes NLTK token overlap between the answer and retrieved chunks (free); only if overlap falls below 0.4 does Stage 2 invoke the LLM to verify and correct the answer.

---

## 3. Token Efficiency

The primary engineering concern throughout was token cost. Every design decision was evaluated against this constraint. Using local sentence-transformers (`all-mpnet-base-v2`) for embeddings eliminated OpenAI embedding API calls entirely. Pre-computing and storing cache embeddings at save time reduced lookup cost from O(N) embeddings per query to O(1). The hybrid confidence scorer eliminated LLM calls for the majority of Level 1 and Level 2 queries. At Berget.AI pricing (€0.90/M tokens, Llama 3.3 70B), a typical query costs approximately €0.00054 — less than a fraction of a cent — with cache hits costing nothing.

---

## 4. Memory and Cache

**Persistent memory** stores 1–2 LLM-extracted facts after each successful answer, capped at 20 facts. These facts are prepended to the system prompt to personalise responses across sessions. Memory also contributes to cache enrichment, improving hit rates for paraphrased follow-up questions. **Cache invalidation** is handled by document fingerprinting: each document's line count is stored, and the cache is wiped only when a document is removed or modified — not when new documents are added, since old answers remain valid.

One approach that was designed but ultimately not implemented as a standalone feature was **multi-turn conversation history** for retrieval context. The pronoun-aware enrichment partially addresses this use case — detecting pronouns in the current query and prepending the previous query to the cache lookup. Full conversational memory was considered but judged to add token overhead without proportional benefit for the single-user, document-focused use case.

---

## 5. Future Work

Several improvements remain for future work. The hybrid confidence threshold (0.30–0.75) was set heuristically and would benefit from calibration against a labelled query set. Cache enrichment currently uses keyword extraction from memory facts, which works well for topically related follow-ups but struggles with cross-topic pronoun references where vocabulary overlap is low. A learned query expansion model fine-tuned on the target document domain would likely outperform the generic WordNet synonyms used here. Finally, the system currently supports a single user and a single document collection; extending to multi-user or multi-collection scenarios would require partitioned caching and memory namespacing.

---

## 6. Reflection on AI-Assisted Development

This project was developed in close collaboration with Claude (Anthropic). The AI was used for code generation, architecture design, literature review, and debugging. The experience highlighted both the strengths and limitations of AI-assisted development. The AI was effective at generating modular, well-documented code and at connecting design decisions to the academic literature. However, it required consistent human oversight: several suggested implementations introduced subtle bugs (misaligned confidence thresholds, double-counting of tokens, incorrect cache similarity display), and the initial architecture included components (HyDE, compaction) that were later removed after reasoning through their actual cost-benefit tradeoffs. The key lesson was that AI tools accelerate implementation but do not replace the need to understand every design decision — particularly when the contribution is the architecture itself.

---

## References

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS. arXiv:2005.11401
- Asai, A. et al. (2023). *Self-RAG*. ICLR 2024. arXiv:2310.11511
- Jiang, Z. et al. (2023). *FLARE*. EMNLP 2023. arXiv:2305.06983
- Gao, L. et al. (2023). *HyDE*. ACL 2023. arXiv:2212.10496
- Es, S. et al. (2024). *RAGAs*. EACL 2024.
- Asl, M. A. et al. (2025). *FAIR-RAG*. arXiv:2510.22344