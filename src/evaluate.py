"""Retrieval evaluation: recall@k and MRR over labeled QA pairs."""

import logging

log = logging.getLogger("rag")


def recall_at_k(ranked_sources: list[str], expected: str, k: int) -> float:
    return 1.0 if expected in ranked_sources[:k] else 0.0


def reciprocal_rank(ranked_sources: list[str], expected: str) -> float:
    try:
        rank = ranked_sources.index(expected) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def evaluate(pipeline, qa_pairs: list[dict], k: int = 4) -> dict:
    recalls, rrs, details = [], [], []
    for qa in qa_pairs:
        results = pipeline.retriever.retrieve(qa["question"], top_k=k)
        ranked = [c.source for c, _ in results]
        r = recall_at_k(ranked, qa["expected_source"], k)
        rr = reciprocal_rank(ranked, qa["expected_source"])
        recalls.append(r)
        rrs.append(rr)
        details.append({"question": qa["question"][:60],
                        "expected": qa["expected_source"],
                        "got": ranked[0] if ranked else None,
                        f"recall@{k}": r, "rr": round(rr, 3)})
    summary = {f"recall@{k}": sum(recalls) / len(recalls),
               "mrr": sum(rrs) / len(rrs),
               "n": len(qa_pairs)}
    log.info("Eval: %s", summary)
    return {"summary": summary, "details": details}
