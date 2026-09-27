"""RAGPipeline: the orchestrator tying ingest -> retrieve -> generate together."""

import logging
import os

import yaml

from .documents import chunk_documents, load_documents
from .embeddings import create_embedder
from .generator import Generator
from .retriever import Retriever
from .vector_store import VectorStore

log = logging.getLogger("rag")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class RAGPipeline:
    def __init__(self, config: dict, embedder=None):
        self.cfg = config
        self.embedder = embedder or create_embedder(config["embeddings"])
        self.store = VectorStore()
        self.retriever = Retriever(self.store, self.embedder,
                                   top_k=config["retrieval"]["top_k"])
        self.generator = Generator(config["generate"])

    # ---- offline phase ----
    def ingest(self) -> int:
        docs = load_documents(self.cfg["ingest"]["docs_dir"],
                              self.cfg["ingest"]["supported_exts"])
        chunks = chunk_documents(docs, self.cfg["chunking"]["chunk_size"],
                                 self.cfg["chunking"]["chunk_overlap"])
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        self.store.save(self.cfg["index"]["path"])
        return len(chunks)

    def load_index(self) -> None:
        self.store = VectorStore.load(self.cfg["index"]["path"])
        if hasattr(self.embedder, "fit_corpus"):
            # corpus-fit backends (e.g. TF-IDF) refit deterministically on the
            # indexed texts so query vectors live in the same space
            self.embedder.fit_corpus([c.text for c in self.store.chunks])
        self.retriever = Retriever(self.store, self.embedder,
                                   top_k=self.cfg["retrieval"]["top_k"])

    # ---- online phase ----
    def answer(self, question: str) -> dict:
        results = self.retriever.retrieve(question)
        answer = self.generator.generate(question, results)
        return {
            "question": question,
            "answer": answer,
            "sources": sorted({c.source for c, _ in results}),
            "scores": [round(s, 3) for _, s in results],
        }
