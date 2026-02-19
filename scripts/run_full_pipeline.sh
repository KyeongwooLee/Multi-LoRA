#!/usr/bin/env bash
set -euo pipefail

QUERY="${1:-Explain Newton's second law with an easy example.}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

python -m src.pipeline.run_full_pipeline --query "${QUERY}"
