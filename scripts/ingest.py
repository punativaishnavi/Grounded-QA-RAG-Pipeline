"""CLI: build the vector index from the document folder."""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import RAGPipeline, load_config  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest documents and build the RAG index.")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    pipe = RAGPipeline(load_config(args.config))
    n = pipe.ingest()
    print(f"\nIndexed {n} chunks -> {pipe.cfg['index']['path']}")


if __name__ == "__main__":
    main()
