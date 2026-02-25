from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = SCRIPT_DIR / "sample_eval_dataset.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_JUDGE_MODEL = "gemini-2.5-flash-lite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate question/reference/model-answer triples with Gemini as judge."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="JSON file that contains eval items.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where criteria, item scores, and summary report are saved.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help="Evaluation model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Gemini judge calls.",
    )
    return parser.parse_args()


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_loads_safe(raw_text: str) -> Any:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(raw_text[start : end + 1])


def _to_100_scale(score_1_to_5: float) -> float:
    return round((score_1_to_5 / 5.0) * 100.0, 2)


def load_eval_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. "
            f"Use the default sample file at {DEFAULT_DATASET_PATH} or pass --dataset."
        )

    data = _json_loads_safe(path.read_text(encoding="utf-8"))
    normalized: list[dict[str, Any]] = []

    if isinstance(data, dict):
        for key, item in data.items():
            if not isinstance(item, dict):
                raise ValueError("Each object value in dict dataset must be a JSON object.")
            normalized.append(
                {
                    "id": str(item.get("id", key)),
                    "question": str(item.get("question", "")),
                    "reference_answer": str(item.get("reference_answer", "")),
                    "model_answer": str(item.get("model_answer", "")),
                    "style_label": item.get("style_label"),
                }
            )
    elif isinstance(data, list):
        for idx, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                raise ValueError("Each item in list dataset must be a JSON object.")
            normalized.append(
                {
                    "id": str(item.get("id", f"item_{idx:03d}")),
                    "question": str(item.get("question", "")),
                    "reference_answer": str(item.get("reference_answer", "")),
                    "model_answer": str(item.get("model_answer", "")),
                    "style_label": item.get("style_label"),
                }
            )
    else:
        raise ValueError("Dataset must be either a JSON array or JSON object.")

    if not normalized:
        raise ValueError("Dataset is empty.")

    missing_fields = []
    for item in normalized:
        for field in ("question", "reference_answer", "model_answer"):
            if not item[field]:
                missing_fields.append((item["id"], field))
    if missing_fields:
        sample = ", ".join([f"{item_id}:{field}" for item_id, field in missing_fields[:5]])
        raise ValueError(f"Dataset has empty required fields. Examples: {sample}")

    return normalized


def default_criteria() -> dict[str, Any]:
    return {
        "criteria": [
            {
                "name": "reference_correctness",
                "description": "정답(레퍼런스)과 핵심 사실/논리가 얼마나 일치하는지 평가",
                "weight": 0.5,
                "score_min": 1,
                "score_max": 5,
            },
            {
                "name": "explanation_clarity",
                "description": "설명이 단계적이고 학습자가 이해하기 쉬운지 평가",
                "weight": 0.2,
                "score_min": 1,
                "score_max": 5,
            },
            {
                "name": "hallucination_control",
                "description": "레퍼런스에 없는 사실 추가, 왜곡, 과장 여부를 평가",
                "weight": 0.2,
                "score_min": 1,
                "score_max": 5,
            },
            {
                "name": "personalization_alignment",
                "description": "요청된 튜터링 스타일/개인화 톤과의 정렬도를 평가",
                "weight": 0.1,
                "score_min": 1,
                "score_max": 5,
            },
        ],
        "pass_threshold": 70.0,
        "overall_formula": "overall_score = (sum(weight * score_1_to_5) / 5) * 100",
        "generation_mode": "fixed_default_criteria",
    }


class GeminiJudge:
    def __init__(self, model_name: str, temperature: float) -> None:
        api_key = os.getenv("GENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("GENAI_API_KEY is required.")

        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._temperature = temperature

    def _generate_json(self, prompt: str) -> dict[str, Any]:
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=self._temperature,
                thinking_config=self._types.ThinkingConfig(thinking_budget=0),
            ),
        )
        if not response.text:
            raise ValueError("Gemini returned an empty response.")
        parsed = _json_loads_safe(response.text)
        if not isinstance(parsed, dict):
            raise ValueError("Gemini response was not a JSON object.")
        return parsed

    def evaluate_item(self, item: dict[str, Any], criteria: dict[str, Any]) -> dict[str, Any]:
        prompt = f"""
You are an LLM judge.
Evaluate one model answer using ONLY the criteria provided below.
Do not invent additional criteria. Return JSON only.

Input item:
{json.dumps(item, ensure_ascii=False)}

Fixed evaluation criteria:
{json.dumps(criteria, ensure_ascii=False)}

Output schema:
{{
  "item_id": "{item['id']}",
  "criterion_scores": [
    {{
      "name": "criterion_name_from_input",
      "score": 1,
      "reason": "short reason"
    }}
  ],
  "overall_score": 0.0,
  "reference_match": "high|medium|low",
  "major_issues": ["issue1", "issue2"],
  "improvement_tip": "one practical fix"
}}

Scoring rules:
- For each criterion, score in [1, 5].
- You must use the exact criterion names from input.
- Compute weighted overall score in [0, 100]:
  overall_score = (sum(weight * score_1_to_5) / 5) * 100
- Keep reasons concise and grounded in question/reference/model_answer.
"""
        return self._generate_json(prompt)


def normalize_item_result(item: dict[str, Any], result: dict[str, Any], criteria: dict[str, Any]) -> dict[str, Any]:
    score_map = {
        str(s.get("name")): s for s in (result.get("criterion_scores") or []) if isinstance(s, dict)
    }
    normalized_scores = []

    for c in criteria["criteria"]:
        name = c["name"]
        entry = score_map.get(name, {})
        try:
            score = int(round(float(entry.get("score", 1))))
        except (TypeError, ValueError):
            score = 1
        score = max(c["score_min"], min(c["score_max"], score))
        reason = str(entry.get("reason", "")).strip() or "No reason provided."
        normalized_scores.append({"name": name, "score": score, "reason": reason})

    weighted_score = sum(
        c["weight"] * next(s["score"] for s in normalized_scores if s["name"] == c["name"])
        for c in criteria["criteria"]
    )
    recomputed_overall = round((weighted_score / 5.0) * 100.0, 2)

    return {
        "item_id": item["id"],
        "question": item["question"],
        "reference_answer": item["reference_answer"],
        "model_answer": item["model_answer"],
        "style_label": item.get("style_label"),
        "criterion_scores": normalized_scores,
        "overall_score": recomputed_overall,
        "reference_match": str(result.get("reference_match", "unknown")),
        "major_issues": result.get("major_issues", []),
        "improvement_tip": str(result.get("improvement_tip", "")),
        "passed": recomputed_overall >= float(criteria["pass_threshold"]),
    }


def build_summary(items: list[dict[str, Any]], criteria: dict[str, Any]) -> dict[str, Any]:
    overall_scores = [float(item["overall_score"]) for item in items]
    pass_rate = (sum(1 for item in items if item["passed"]) / len(items)) if items else 0.0

    criterion_averages = []
    for c in criteria["criteria"]:
        name = c["name"]
        per_item_scores = [
            next(s["score"] for s in item["criterion_scores"] if s["name"] == name)
            for item in items
        ]
        avg_1_to_5 = mean(per_item_scores) if per_item_scores else 0.0
        criterion_averages.append(
            {
                "name": name,
                "weight": c["weight"],
                "avg_score_1_to_5": round(avg_1_to_5, 3),
                "avg_score_100": _to_100_scale(avg_1_to_5),
            }
        )

    return {
        "num_items": len(items),
        "avg_overall_score": round(mean(overall_scores), 3) if overall_scores else 0.0,
        "min_overall_score": round(min(overall_scores), 3) if overall_scores else 0.0,
        "max_overall_score": round(max(overall_scores), 3) if overall_scores else 0.0,
        "pass_rate": round(pass_rate, 4),
        "pass_threshold": float(criteria["pass_threshold"]),
        "criterion_averages": criterion_averages,
    }


def main() -> None:
    args = parse_args()
    items = load_eval_items(args.dataset)
    criteria = default_criteria()
    judge = GeminiJudge(model_name=args.judge_model, temperature=args.temperature)

    results = []
    for idx, item in enumerate(items, start=1):
        raw_result = judge.evaluate_item(item=item, criteria=criteria)
        normalized_result = normalize_item_result(item=item, result=raw_result, criteria=criteria)
        results.append(normalized_result)
        print(f"[{idx}/{len(items)}] {item['id']} -> score={normalized_result['overall_score']}")

    summary = build_summary(results, criteria)

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "judge_model": args.judge_model,
            "dataset_path": str(args.dataset.resolve()),
        },
        "criteria_generation": {
            "criteria": criteria,
        },
        "summary": summary,
        "items": results,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    criteria_path = output_dir / "criteria.json"
    items_path = output_dir / "item_results.json"
    report_path = output_dir / "final_report.json"
    save_json(criteria_path, criteria)
    save_json(items_path, {"items": results})
    save_json(report_path, report)

    print(f"\nSaved criteria: {criteria_path}")
    print(f"Saved item results: {items_path}")
    print(f"Saved final report: {report_path}")
    print(f"Average overall score: {summary['avg_overall_score']}")


if __name__ == "__main__":
    main()
