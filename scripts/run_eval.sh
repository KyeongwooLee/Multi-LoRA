#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 \"<query>\" [sample_size]"
  exit 1
fi

QUERY="$1"
SAMPLE_SIZE="${2:-10}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

python -m src.eval.eval_correctness --sample-size "${SAMPLE_SIZE}"
python -m src.eval.eval_personalization --sample-size "${SAMPLE_SIZE}"
python -m src.eval.eval_system --query "${QUERY}" --num-runs 5
python - <<'PY'
from src.config import ProjectConfig
from src.eval.build_report import build_report

config = ProjectConfig()
report = build_report(config)
print(report)
PY
