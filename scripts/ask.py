"""CLI: ask the RAG pipeline a question."""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import RAGPipeline, load_config  # noqa: E402

logging.basicConfig(level=logging.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask a question over the indexed documents.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--question", required=True)
    args = ap.parse_args()

    pipe = RAGPipeline(load_config(args.config))
    pipe.load_index()
    out = pipe.answer(args.question)
    print(f"\nQ: {out['question']}\n")
    print(out["answer"])
    print(f"\nSources: {', '.join(out['sources'])}")


if __name__ == "__main__":
    main()
