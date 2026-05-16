# RAG Agent with Escalating Retrieval

A token-efficient RAG agent that escalates through retrieval strategies in order of cost, stopping as soon as confidence is sufficient. Built for Information Retrieval research and practical question answering over local documents.

---

## How it works

The agent follows a cost-aware escalation ladder — it starts with free local strategies and only calls the LLM when cheaper methods fail to find a confident answer.

| Level | Strategy | API calls | When used |
|-------|----------|-----------|-----------|
| 0 | Semantic cache | 0 | Similar question asked before |
| 1 | Query expansion + semantic search | 1 | First attempt on local docs |
| 2 | BM25 keyword search | 1 | When semantic search fails |
| 3 | Web search (DuckDuckGo) | 2 | When local docs have no answer |

**Additional features:**
- Cross-encoder re-ranking at every level
- Two-stage self-correction (NLTK overlap + LLM verification)
- Persistent memory across sessions
- Cache invalidation by document fingerprinting
- Pronoun-aware cache enrichment

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/dianamikova/rag-agent
cd rag-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and add your API key:

```bash
LLM_API_KEY=your-key-here
LLM_BASE_URL=https://api.berget.ai/v1
LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct
```

> **Free credits:** Sign up at [berget.ai](https://berget.ai) for free API credits.

### 4. Add your documents

Drop `.txt`, `.md`, or `.pdf` files into the `docs/` folder. The agent indexes them on startup.

### 5. Run

**Terminal (interactive mode):**
```bash
python main.py
```

**Single query:**
```bash
python main.py "What is retrieval augmented generation?"
```

**Streamlit demo UI:**
```bash
streamlit run app.py
```

**Demo video:**

https://drive.google.com/file/d/1PUSQvuvORo2YxYNBjWq_Dy5v27DWyZz2/view?usp=sharing

---

## Provider configuration

The agent works with any OpenAI-compatible API. Change three lines in `.env`:

| Provider | LLM_BASE_URL | LLM_MODEL |
|----------|-------------|-----------|
| Berget.AI | `https://api.berget.ai/v1` | `meta-llama/Llama-3.3-70B-Instruct` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4.1-nano` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Ollama (local) | `http://localhost:11434/v1` | `llama3` |

---

## Project structure

```
rag-agent/
├── agent.py           # Core escalation pipeline
├── retriever.py       # Document indexing, semantic + BM25 search
├── expander.py        # Zero-cost query expansion (Level 1)
├── reranker.py        # Cross-encoder re-ranking
├── websearch.py       # DuckDuckGo web search (Level 3)
├── cache.py           # Semantic cache with pre-computed embeddings
├── memory.py          # Persistent memory across sessions
├── self_correction.py # Two-stage answer self-correction
├── logger.py          # Structured per-query logging
├── main.py            # CLI entry point
├── app.py             # Streamlit demo UI
├── docs/              # Drop your documents here
├── logs/              # Auto-generated (cache, memory, query history)
├── requirements.txt
└── .env.example
```

---

## Token cost per level

| Level | Retrieval | Confidence scoring | Answer generation |
|-------|-----------|-------------------|-------------------|
| 0 Cache | Free | Free | Free |
| 1 Semantic | Free (local) | Free (rerank score) | ~600 tokens |
| 2 BM25 | Free (local) | Free (rerank score) | ~600 tokens |
| 3 Web search | Free (DuckDuckGo) | ~50 tokens | ~600 tokens |

At Berget.AI with Llama 3.3 70B (€0.90/M tokens), a typical query costs **€0.00054** — less than a fraction of a cent.

---

## Supported document formats

- `.txt` — plain text (summaries recommended)
- `.md` — Markdown
- `.pdf` — PDF (requires `pymupdf`, included in requirements)

---

## Logs

Every query is logged to `logs/queries.jsonl` with full escalation trace, confidence scores, tokens used, and timing. The Streamlit UI displays a live confidence timeline across all queries.

---

## References

- Asai et al. (2023). *Self-RAG*. ICLR 2024. [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
- Jiang et al. (2023). *FLARE*. EMNLP 2023. [arXiv:2305.06983](https://arxiv.org/abs/2305.06983)
- Gao et al. (2023). *HyDE*. ACL 2023. [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)
- Es et al. (2024). *RAGAs*. EACL 2024.
- Asl et al. (2025). *FAIR-RAG*. [arXiv:2510.22344](https://arxiv.org/abs/2510.22344)