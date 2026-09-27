"""A small persisted vector store: cosine similarity over a NumPy matrix."""

import logging
import os

import numpy as np

from .documents import Chunk

log = logging.getLogger("rag")


class VectorStore:
    def __init__(self):
        self.vectors: np.ndarray | None = None
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        assert vectors.ndim == 2 and len(vectors) == len(chunks)
        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        log.info("Index now holds %d vector(s)", len(self.chunks))

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        """Cosine similarity search (vectors are L2-normalized)."""
        if self.vectors is None or not len(self.chunks):
            return []
        scores = self.vectors @ query_vec
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.chunks[i], float(scores[i])) for i in idx]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            vectors=self.vectors,
            texts=np.array([c.text for c in self.chunks]),
            sources=np.array([c.source for c in self.chunks]),
            chunk_ids=np.array([c.chunk_id for c in self.chunks]),
        )
        log.info("Index saved to %s (%d vectors)", path, len(self.chunks))

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        data = np.load(path, allow_pickle=True)
        store = cls()
        store.vectors = data["vectors"]
        store.chunks = [Chunk(text=t, source=s, chunk_id=int(i))
                        for t, s, i in zip(data["texts"], data["sources"], data["chunk_ids"])]
        log.info("Index loaded from %s (%d vectors)", path, len(store.chunks))
        return store

    def __len__(self) -> int:
        return len(self.chunks)
