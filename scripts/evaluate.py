"""CLI: evaluate retrieval quality over the labeled QA set."""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import evaluate  # noqa: E402
from src.pipeline import RAGPipeline, load_config  # noqa: E402

logging.basicConfig(level=logging.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate retrieval recall@k and MRR.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    pipe = RAGPipeline(cfg)
    pipe.load_index()
    report = evaluate(pipe, cfg["evaluate"]["qa_pairs"], k=args.k)

    s = report["summary"]
    print(f"\nRetrieval eval over {s['n']} questions (k={args.k}):")
    print(f"  recall@{args.k}: {s[f'recall@{args.k}']:.2f}")
    print(f"  MRR:            {s['mrr']:.2f}\n")
    for d in report["details"]:
        mark = "✓" if d[f"recall@{args.k}"] else "✗"
        print(f"  {mark} {d['question']}… -> {d['got']} (expected {d['expected']})")


if __name__ == "__main__":
    main()
