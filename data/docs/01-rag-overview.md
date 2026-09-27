# RAG Overview

Retrieval-Augmented Generation (RAG) is a technique that grounds a large
language model in external knowledge. Instead of relying only on what the
model memorized during training, a RAG system retrieves relevant passages
from a document collection at query time and feeds them to the model along
with the question.

A typical RAG pipeline has two phases. Offline, documents are loaded, split
into chunks, embedded into vectors, and stored in an index. Online, the user
question is embedded the same way, the most similar chunks are retrieved,
and a prompt containing the question plus the retrieved context is sent to
the language model.

RAG reduces hallucinations because the model can quote or paraphrase the
retrieved passages. It also keeps knowledge fresh: updating the document
collection is cheaper than retraining a model. Common uses include
enterprise search, customer support assistants, and research copilots.
