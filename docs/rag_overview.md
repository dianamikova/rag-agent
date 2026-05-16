# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with large language model generation. Instead of relying solely on the model's parametric knowledge, RAG retrieves relevant documents from an external knowledge base and uses them as context for generating answers.

## How RAG Works

1. The user submits a query.
2. The system retrieves the most relevant documents or chunks from a knowledge base using semantic search or keyword search.
3. The retrieved documents are passed as context to the language model.
4. The language model generates an answer grounded in the retrieved context.

## Advantages of RAG

- Reduces hallucination by grounding responses in real documents.
- Allows the model to access up-to-date information not present in training data.
- More cost-effective than fine-tuning for domain adaptation.
- Provides citations and traceability for answers.

## Limitations of RAG

- Retrieval quality directly affects answer quality. If the wrong documents are retrieved, the answer will be poor.
- Single-pass retrieval can fail on ambiguous or complex queries.
- Context window limits how much retrieved content can be used.

## Advanced RAG Techniques

### Self-RAG
Self-RAG trains a model to use reflection tokens to decide when to retrieve, and to critique its own retrieved passages for relevance and support.

### HyDE (Hypothetical Document Embeddings)
HyDE generates a hypothetical answer document and uses its embedding to search for real documents. This bridges the semantic gap between questions and answer-shaped documents.

### FLARE
FLARE (Forward-Looking Active Retrieval) iteratively retrieves new documents during generation when the model's confidence in its next sentence is low.

### Iterative Retrieval
Instead of retrieving once, iterative systems refine their queries based on what was already retrieved, filling gaps in the evidence.

## Evaluation of RAG Systems

RAG systems are typically evaluated on:
- **Faithfulness**: Does the answer stay grounded in the retrieved context?
- **Answer relevancy**: Does the answer actually address the question?
- **Context precision**: Were the retrieved chunks relevant?
- **Context recall**: Were all necessary chunks retrieved?

The RAGAs framework (Es et al., 2024) provides automated metrics for all four dimensions without requiring human annotations.
