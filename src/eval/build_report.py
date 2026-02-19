from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import ProjectConfig, ensure_project_dirs


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(config: ProjectConfig) -> dict:
    ensure_project_dirs(config)

    correctness = _read_json(config.reports_dir / "correctness_eval.json")
    personalization = _read_json(config.reports_dir / "personalization_eval.json")
    system = _read_json(config.reports_dir / "system_eval.json")
    pipeline_summary = _read_json(config.logs_dir / "pipeline_summary.json")

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "base_model": config.base_model_name,
            "router_embedding_model": config.router_embedding_model,
        },
        "scores": {
            "correctness": correctness.get("avg_correctness", 0.0),
            "personalization": personalization.get("style_match_rate", 0.0),
            "system_latency_p95_ms": system.get("latency_ms_p95", 0.0),
        },
        "details": {
            "correctness": correctness,
            "personalization": personalization,
            "system": system,
            "pipeline": pipeline_summary,
        },
    }

    output_path = config.reports_dir / "final_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["path"] = str(output_path)
    return report
