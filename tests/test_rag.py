"""Tests for the RAG pipeline (uses a tiny deterministic dummy embedder)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.documents import Chunk, chunk_documents, chunk_text  # noqa: E402
from src.evaluate import reciprocal_rank, recall_at_k  # noqa: E402
from src.generator import Generator  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402


class DummyEmbedder:
    """Deterministic bag-of-hashed-words embedder for fast offline tests."""

    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in text.lower().split():
            v[hash(w) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / max(n, 1e-12)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self._vec(query)


def _chunks():
    return [
        Chunk(text="retrieval augmented generation grounds language models", source="a.md", chunk_id=0),
        Chunk(text="chunking splits documents into overlapping passages", source="b.md", chunk_id=0),
        Chunk(text="embeddings map text to dense vector spaces", source="c.md", chunk_id=0),
    ]


# ---- chunking ----

def test_chunk_text_overlap():
    chunks = chunk_text("abcdefghijklmnopqrstuvwxyz", chunk_size=10, chunk_overlap=4)
    assert chunks[0] == "abcdefghij"
    assert chunks[1] == "ghijklmnop"  # 4-char overlap
    assert len(chunks) == 4


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=5, chunk_overlap=5)


def test_chunk_documents_tracks_metadata():
    chunks = chunk_documents([("doc.md", "x" * 1500)], chunk_size=600, chunk_overlap=100)
    assert len(chunks) == 3
    assert all(c.source == "doc.md" for c in chunks)
    assert [c.chunk_id for c in chunks] == [0, 1, 2]


# ---- vector store ----

def test_vector_store_search_ranks_relevant_first(tmp_path):
    chunks = _chunks()
    vecs = DummyEmbedder().embed([c.text for c in chunks])
    store = VectorStore()
    store.add(chunks, vecs)
    hits = store.search(DummyEmbedder().embed_query("what is chunking"), top_k=2)
    assert hits[0][0].source == "b.md"
    assert hits[0][1] >= hits[1][1]  # scores descending


def test_vector_store_save_load_roundtrip(tmp_path):
    chunks = _chunks()
    vecs = DummyEmbedder().embed([c.text for c in chunks])
    store = VectorStore()
    store.add(chunks, vecs)
    path = str(tmp_path / "index.npz")
    store.save(path)
    loaded = VectorStore.load(path)
    assert len(loaded) == 3
    hits = loaded.search(DummyEmbedder().embed_query("dense vector embeddings"), top_k=1)
    assert hits[0][0].source == "c.md"


def test_vector_store_empty_search():
    assert VectorStore().search(np.zeros(64, dtype=np.float32), top_k=3) == []


# ---- evaluation helpers ----

def test_recall_and_mrr():
    ranked = ["b.md", "a.md", "c.md"]
    assert recall_at_k(ranked, "a.md", k=2) == 1.0
    assert recall_at_k(ranked, "c.md", k=2) == 0.0
    assert reciprocal_rank(ranked, "a.md") == pytest.approx(0.5)
    assert reciprocal_rank(ranked, "zzz.md") == 0.0


# ---- generator (extractive) ----

def test_extractive_generator_cites_sources():
    gen = Generator({"backend": "extractive"})
    chunks = _chunks()
    scored = list(zip(chunks, [0.9, 0.7, 0.5]))
    out = gen.generate("what is chunking", scored)
    assert "b.md" in out  # source cited in the chunk header
    assert "score 0.90" in out


def test_extractive_generator_no_chunks():
    gen = Generator({"backend": "extractive"})
    assert "don't know" in gen.generate("anything", [])


def test_generator_rejects_unknown_backend():
    gen = Generator({"backend": "nope"})
    with pytest.raises(ValueError):
        gen.generate("q", [(_chunks()[0], 1.0)])


# ---- tfidf embedder backend ----

def test_tfidf_embedder_fit_query_roundtrip(tmp_path):
    from src.embeddings import TfidfEmbedder, create_embedder
    chunks = _chunks()
    emb = create_embedder({"backend": "tfidf", "max_features": 128})
    assert isinstance(emb, TfidfEmbedder)
    vecs = emb.embed([c.text for c in chunks])
    assert vecs.shape[0] == 3 and vecs.shape[1] <= 128

    # simulate a fresh process: new embedder, refit on the indexed texts
    emb2 = TfidfEmbedder(max_features=128)
    emb2.fit_corpus([c.text for c in chunks])
    q = emb2.embed_query("what is chunking")
    store = VectorStore()
    store.add(chunks, vecs)
    hits = store.search(q, top_k=1)
    assert hits[0][0].source == "b.md"

    # querying before any fit must fail loudly
    with pytest.raises(RuntimeError):
        TfidfEmbedder().embed_query("hello")


def test_create_embedder_rejects_unknown_backend():
    from src.embeddings import create_embedder
    with pytest.raises(ValueError):
        create_embedder({"backend": "nope"})
