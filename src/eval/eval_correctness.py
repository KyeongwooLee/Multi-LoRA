from __future__ import annotations

import argparse
import json
import re
from statistics import mean

from src.config import ProjectConfig, ensure_project_dirs
from src.data.ingest import load_jsonl_records
from src.inference.generate import run_inference


TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def _overlap_score(prediction: str, reference: str) -> float:
    pred_tokens = set(_tokenize(prediction))
    ref_tokens = set(_tokenize(reference))
    if not ref_tokens:
        return 0.0
    return len(pred_tokens & ref_tokens) / len(ref_tokens)


def _load_eval_records(config: ProjectConfig) -> list[dict]:
    for path in (config.processed_eval_path, config.processed_test_path, config.processed_train_path):
        records = load_jsonl_records(path)
        if records:
            return records
    return []


def run_correctness_eval(config: ProjectConfig, sample_size: int | None = None) -> dict:
    ensure_project_dirs(config)

    records = _load_eval_records(config)
    if not records:
        result = {
            "sample_count": 0,
            "avg_correctness": 0.0,
            "items": [],
        }
        output_path = config.reports_dir / "correctness_eval.json"
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["path"] = str(output_path)
        return result

    limit = sample_size or config.eval_sample_size
    records = records[:limit]

    items = []
    scores = []
    for record in records:
        query = record.get("instruction", "")
        reference = record.get("response", "")
        infer_result = run_inference(query=query, config=config)
        prediction = infer_result["response"]
        score = _overlap_score(prediction=prediction, reference=reference)
        scores.append(score)
        items.append(
            {
                "query": query,
                "reference": reference,
                "prediction": prediction,
                "style_label": record.get("style_label"),
                "selected_style": infer_result.get("selected_style"),
                "overlap_score": score,
            }
        )

    result = {
        "sample_count": len(items),
        "avg_correctness": mean(scores) if scores else 0.0,
        "items": items,
    }

    output_path = config.reports_dir / "correctness_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["path"] = str(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run correctness evaluation")
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()

    config = ProjectConfig()
    result = run_correctness_eval(config=config, sample_size=args.sample_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
