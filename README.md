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
