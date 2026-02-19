from __future__ import annotations

import argparse
import json
from statistics import mean

from src.config import ProjectConfig, ensure_project_dirs
from src.data.ingest import load_jsonl_records
from src.data.style_selector import style_phrase_score
from src.inference.generate import run_inference


def _load_eval_records(config: ProjectConfig) -> list[dict]:
    for path in (config.processed_eval_path, config.processed_test_path, config.processed_train_path):
        records = load_jsonl_records(path)
        if records:
            return records
    return []


def run_personalization_eval(config: ProjectConfig, sample_size: int | None = None) -> dict:
    ensure_project_dirs(config)

    records = _load_eval_records(config)
    if not records:
        result = {
            "sample_count": 0,
            "style_match_rate": 0.0,
            "style_phrase_alignment": 0.0,
            "items": [],
        }
        output_path = config.reports_dir / "personalization_eval.json"
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["path"] = str(output_path)
        return result

    limit = sample_size or config.eval_sample_size
    records = records[:limit]

    items = []
    style_matches = []
    phrase_scores = []

    for record in records:
        query = record.get("instruction", "")
        gold_style = record.get("style_label")
        infer_result = run_inference(query=query, config=config)
        predicted_style = infer_result.get("selected_style")
        response = infer_result.get("response", "")

        matched = int(predicted_style == gold_style)
        phrase_alignment = style_phrase_score(predicted_style, response)

        style_matches.append(matched)
        phrase_scores.append(phrase_alignment)
        items.append(
            {
                "query": query,
                "gold_style": gold_style,
                "predicted_style": predicted_style,
                "matched": bool(matched),
                "phrase_alignment": phrase_alignment,
            }
        )

    result = {
        "sample_count": len(items),
        "style_match_rate": mean(style_matches) if style_matches else 0.0,
        "style_phrase_alignment": mean(phrase_scores) if phrase_scores else 0.0,
        "items": items,
    }

    output_path = config.reports_dir / "personalization_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["path"] = str(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run personalization evaluation")
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()

    config = ProjectConfig()
    result = run_personalization_eval(config=config, sample_size=args.sample_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
