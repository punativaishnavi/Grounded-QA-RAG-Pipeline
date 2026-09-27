# Embeddings

Embeddings are dense vector representations of text. An embedding model
maps a sentence or paragraph to a point in a high-dimensional space —
typically a few hundred dimensions — where texts with similar meaning
end up close together.

This works because embedding models are trained on massive text corpora
to predict context: words and sentences that appear in similar contexts
get similar vectors. Cosine similarity between two vectors then measures
semantic closeness, regardless of exact word overlap. That is why a
query about "canine" can retrieve a passage about "dogs" even though
they share no words.

Popular open embedding models include the MiniLM family and E5, while
commercial APIs offer larger alternatives. For retrieval, embeddings are
usually L2-normalized so that cosine similarity reduces to a dot
product, which is fast to compute over millions of vectors.
