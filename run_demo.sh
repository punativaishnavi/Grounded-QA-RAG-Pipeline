#!/usr/bin/env bash
# End-to-end RAG demo: ingest -> ask -> evaluate.
set -euo pipefail
cd "$(dirname "$0")"

python scripts/ingest.py --config config/config.yaml
echo ""
python scripts/ask.py --config config/config.yaml \
  --question "What is chunking and why does it matter?"
echo ""
python scripts/evaluate.py --config config/config.yaml --k 4
