"""Retrieval: embed the query, search the index, return ranked chunks."""

import logging

from .documents import Chunk
from .embeddings import Embedder
from .vector_store import VectorStore

log = logging.getLogger("rag")


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, top_k: int = 4):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[Chunk, float]]:
        qvec = self.embedder.embed_query(query)
        results = self.store.search(qvec, top_k or self.top_k)
        log.info("Retrieved %d chunk(s) for query: %r", len(results), query[:60])
        return results
