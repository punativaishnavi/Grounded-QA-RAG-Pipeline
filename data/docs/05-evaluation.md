# Evaluating RAG Systems

A RAG system should be evaluated at both stages: retrieval and
generation. Retrieval quality is measured with ranking metrics computed
over labeled question–document pairs. Recall@k asks: is a relevant
document among the top-k retrieved chunks? Mean Reciprocal Rank (MRR)
rewards putting the first relevant chunk as high as possible.

Generation quality is harder to measure automatically. Common approaches
include faithfulness checks (does every claim in the answer appear in
the retrieved context?), answer relevance scoring by a second model,
and human preference ratings on sampled queries.

A practical evaluation loop: maintain a small set of representative
questions with known-good source documents, re-run retrieval metrics on
every pipeline change (chunk size, embedding model, top-k), and track
whether generation answers stay grounded. Improvements to chunking and
retrieval usually move the needle more than swapping the LLM.
