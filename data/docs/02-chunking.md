# Chunking Strategies

Chunking splits long documents into smaller passages before embedding.
Each chunk should express one focused idea: if chunks are too large, the
embedding dilutes the signal with unrelated content; if they are too
small, they lose the surrounding context needed to interpret them.

A common approach is fixed-size chunking with overlap — for example,
600-character chunks with 120 characters of overlap — so that ideas
spanning a boundary still appear whole in at least one chunk. More
advanced strategies split on semantic boundaries: paragraphs, sections,
or sentence windows.

Metadata matters too. Storing the source filename, page number, and chunk
index alongside each chunk lets the system cite sources and lets users
verify answers against the original documents. Good chunking is often
the highest-leverage improvement in a RAG pipeline.
