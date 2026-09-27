# Vector Search: Dense vs Sparse Retrieval

Dense retrieval uses embedding vectors and approximate nearest-neighbor
search: the query is embedded, and the index returns the chunks with the
highest cosine similarity. It captures paraphrase and synonymy well but
can miss exact terms like product codes or names.

Sparse retrieval, exemplified by BM25, scores chunks by weighted keyword
overlap. It excels at exact matches and rare terms but understands no
semantics — "automobile" will not match "car".

Production systems often combine both (hybrid search): run dense and
sparse retrieval in parallel, then fuse the rankings with a method like
reciprocal rank fusion, optionally followed by a cross-encoder reranker
that scores each query–chunk pair precisely. The index itself can be a
simple NumPy array for small collections or a dedicated vector database
such as FAISS, Qdrant, or Pinecone at larger scale.
