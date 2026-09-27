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

## Quickstart (no API keys needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Build the index from the sample knowledge base
python scripts/ingest.py --config config/config.yaml

# 2. Ask questions
python scripts/ask.py --config config/config.yaml --question "What is chunking and why does it matter?"

# 3. Evaluate retrieval quality
python scripts/evaluate.py --config config/config.yaml

# 4. Run the tests
pytest -v
```

The sample knowledge base in `data/docs/` covers RAG concepts, so try:
- *"How do embeddings capture meaning?"*
- *"What is the difference between dense and sparse retrieval?"*
- *"How do you evaluate a RAG system?"*

## Use your own documents

Drop `.txt`, `.md`, or `.pdf` files into `data/docs/` (or point
`ingest.docs_dir` at your folder in `config/config.yaml`) and re-run
`scripts/ingest.py`.

## LLM backends (`generate.backend`)

| Backend | Needs | Notes |
|---------|-------|-------|
| `extractive` (default) | nothing | returns the top chunks as a grounded answer — great for demos |
| `ollama` | Ollama running locally | set `generate.model`, e.g. `llama3.1` |
| `openai` | `OPENAI_API_KEY` | set `generate.model`, e.g. `gpt-4o-mini` |

The prompt template lives in `config/config.yaml` (`generate.prompt_template`).

## Configuration (`config/config.yaml`)

| Key | Description |
|-----|-------------|
| `ingest.docs_dir` | folder of source documents |
| `chunking.chunk_size` / `chunk_overlap` | characters per chunk / overlap |
| `embeddings.model` | sentence-transformers model id |
| `index.path` | where the built index is stored |
| `retrieval.top_k` | chunks retrieved per query |
| `generate.backend` / `generate.model` | answer generation backend |

## Project structure

```
rag-pipeline/
├── config/config.yaml        # all knobs in one place
├── data/
│   ├── docs/                 # sample knowledge base (RAG concepts)
│   └── index/                # built vector index (generated)
├── src/
│   ├── documents.py          # loaders + overlapping chunker
│   ├── embeddings.py         # sentence-transformers wrapper
│   ├── vector_store.py       # NumPy cosine-similarity store (save/load)
│   ├── retriever.py          # top-k search
│   ├── generator.py          # extractive / ollama / openai backends
│   ├── pipeline.py           # RAGPipeline: ingest() + answer()
│   └── evaluate.py           # recall@k, MRR over a QA set
├── scripts/
│   ├── ingest.py             # CLI: build the index
│   ├── ask.py                # CLI: ask questions
│   └── evaluate.py           # CLI: run retrieval eval
├── tests/test_rag.py
└── run_demo.sh
```

## Sample run (SBERT backend)

```
$ python scripts/ask.py --question "What is chunking and why does it matter?"

Q: What is chunking and why does it matter?

[chunk 0 from 02-chunking.md | score 0.57]
# Chunking Strategies Chunking splits long documents into smaller passages
before embedding...
...

Sources: 01-rag-overview.md, 02-chunking.md, 04-vector-search.md
```

Retrieval eval on the 5-question QA set (`scripts/evaluate.py`):

| Backend | recall@4 | MRR |
|---------|----------|-----|
| SBERT (`all-MiniLM-L6-v2`) | 1.00 | 1.00 |
| TF-IDF | 1.00 | 0.80 |

## Techniques demonstrated

- Document chunking with overlap, metadata tracking (source, chunk id)
- Dense embeddings + cosine similarity search from first principles
- Persisted vector index (no external DB required)
- Prompt templating, pluggable LLM backends
- Retrieval evaluation: recall@k, mean reciprocal rank (MRR)
- Seeded, tested, one-command reproducible

## License

MIT — see [LICENSE](LICENSE).
