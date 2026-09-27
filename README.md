# rag-pipeline

End-to-end Retrieval-Augmented Generation (RAG) pipeline in Python —
from raw documents to grounded answers, with retrieval evaluation.

## How it works

```
  docs/ ──▶  LOAD & CHUNK  ──▶  EMBED  ──▶  VECTOR INDEX
                                                  │
  "What is chunking?" ──▶  RETRIEVE top-k  ◀──────┘
                                │
                                ▼
                        GENERATE answer (LLM or extractive demo)
                                │
                                ▼
                        EVALUATE (recall@k, MRR)
```

1. **Ingest** — load `.txt` / `.md` / `.pdf` documents, split into
   overlapping chunks (configurable size/overlap).
2. **Embed** — encode chunks with dense embeddings
   (`sentence-transformers`, `all-MiniLM-L6-v2`); vectors are L2-normalized
   for cosine search. A lightweight TF-IDF backend (`embeddings.backend:
   tfidf`) is included for environments without the model download.
3. **Index** — a NumPy vector store (cosine similarity) persisted to disk,
   so you build the index once and query many times.
4. **Retrieve** — top-k semantic search over the index.
5. **Generate** — answer with an LLM backend, or with the built-in
   extractive mode (no API key needed).
6. **Evaluate** — recall@k and MRR against a small labeled QA set.
