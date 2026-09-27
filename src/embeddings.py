"""Embedding backends.

- sbert:  dense embeddings via sentence-transformers (best quality,
          downloads a ~90MB model on first use).
- tfidf:  lightweight sklearn TF-IDF baseline — no downloads, runs anywhere.
"""

import logging

import numpy as np

log = logging.getLogger("rag")


class SbertEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 batch_size: int = 32, normalize: bool = True):
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.normalize = normalize
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode texts -> (n, dim) float32 array, L2-normalized if configured."""
        vecs = self.model.encode(texts, batch_size=self.batch_size,
                                 show_progress_bar=False,
                                 convert_to_numpy=True).astype(np.float32)
        if self.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-12)
        return vecs

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]


class TfidfEmbedder:
    """TF-IDF baseline: fit on the corpus at ingest time, transform queries."""

    def __init__(self, max_features: int = 4096):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(max_features=max_features,
                                           ngram_range=(1, 2),
                                           sublinear_tf=True)
        self._fitted = False
        self.dim = max_features

    def embed(self, texts: list[str]) -> np.ndarray:
        mat = self._vectorizer.fit_transform(texts)
        self._fitted = True
        return mat.toarray().astype(np.float32)

    def fit_corpus(self, texts: list[str]) -> None:
        """(Re)fit the vectorizer on corpus texts — used when loading a saved index."""
        self._vectorizer.fit(texts)
        self._fitted = True

    def embed_query(self, query: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must embed documents before queries")
        return self._vectorizer.transform([query]).toarray().astype(np.float32)[0]


def create_embedder(cfg: dict):
    backend = cfg.get("backend", "sbert")
    if backend == "sbert":
        return SbertEmbedder(model_name=cfg.get("model", "all-MiniLM-L6-v2"),
                             batch_size=cfg.get("batch_size", 32),
                             normalize=cfg.get("normalize", True))
    if backend == "tfidf":
        return TfidfEmbedder(max_features=cfg.get("max_features", 4096))
    raise ValueError(f"Unknown embeddings backend: {backend!r}")


# Backwards-compatible alias
Embedder = SbertEmbedder
