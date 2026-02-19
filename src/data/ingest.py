from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.config import ProjectConfig, STYLE_LABELS, ensure_project_dirs, seed_everything
from src.data.catalog import DATASET_SPECS, build_source_path
from src.data.preprocess import UnifiedSample, deduplicate, is_quality_sample, sanitize_sample
from src.data.split import split_by_style
from src.data.style_selector import choose_style


TEXT_FILE_SUFFIXES = (".jsonl", ".json", ".csv", ".txt")


def load_jsonl_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _pick_first(item: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_pairs_from_item(item: dict) -> list[tuple[str, str]]:
    if not isinstance(item, dict):
        return []

    prompt = _pick_first(
        item,
        (
            "instruction",
            "prompt",
            "question",
            "input",
            "query",
            "student_text",
            "student",
            "user",
        ),
    )
    response = _pick_first(
        item,
        (
            "response",
            "answer",
            "output",
            "teacher_text",
            "teacher",
            "assistant",
            "target",
            "feedback",
        ),
    )
    if prompt and response:
        return [(prompt, response)]

    for list_key in ("messages", "conversation", "dialogue", "turns"):
        turns = item.get(list_key)
        if not isinstance(turns, list):
            continue
        pairs: list[tuple[str, str]] = []
        for idx in range(len(turns) - 1):
            current = turns[idx]
            nxt = turns[idx + 1]
            if not isinstance(current, dict) or not isinstance(nxt, dict):
                continue
            role_a = str(current.get("role", current.get("speaker", ""))).lower()
            role_b = str(nxt.get("role", nxt.get("speaker", ""))).lower()
            text_a = _pick_first(current, ("content", "text", "utterance", "message"))
            text_b = _pick_first(nxt, ("content", "text", "utterance", "message"))
            if not text_a or not text_b:
                continue
            if role_a in {"user", "student", "human"} and role_b in {
                "assistant",
                "teacher",
                "bot",
            }:
                pairs.append((text_a, text_b))
        if pairs:
            return pairs

    text = _pick_first(item, ("text", "content"))
    if "\t" in text:
        left, right = text.split("\t", 1)
        if left.strip() and right.strip():
            return [(left.strip(), right.strip())]

    return []


def _iter_json_items(path: Path):
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        yield from payload
        return
    if isinstance(payload, dict):
        for key in ("data", "records", "examples", "items", "conversations"):
            value = payload.get(key)
            if isinstance(value, list):
                yield from value
                return
        yield payload


def _read_candidates_from_file(path: Path) -> list[tuple[str, str]]:
    suffix = path.suffix.lower()
    pairs: list[tuple[str, str]] = []

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                pairs.extend(_extract_pairs_from_item(item))
        return pairs

    if suffix == ".json":
        for item in _iter_json_items(path):
            pairs.extend(_extract_pairs_from_item(item))
        return pairs

    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                pairs.extend(_extract_pairs_from_item(row))
        return pairs

    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    prompt, response = line.split("\t", 1)
                elif "|||" in line:
                    prompt, response = line.split("|||", 1)
                else:
                    continue
                if prompt.strip() and response.strip():
                    pairs.append((prompt.strip(), response.strip()))
        return pairs

    return pairs


def _iter_dataset_files(dataset_dir: Path):
    if not dataset_dir.exists():
        return
    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_FILE_SUFFIXES:
            yield path


def _assign_style(dataset_name: str, default_style: str | None, instruction: str, response: str) -> str:
    if dataset_name == "tscc":
        merged = f"{instruction} {response}"
        return choose_style(merged, ["direct", "motivational"], default="direct")
    return default_style or "direct"


def _synthetic_instruction(style: str, idx: int) -> str:
    prompt_map = {
        "direct": "Solve this math problem and explain each step clearly:",
        "socratic": "I am stuck on this concept. Ask me guiding questions:",
        "scaffolding": "Give me a small hint first, then guide me gradually:",
        "feedback": "Please review my answer and tell me what to improve:",
        "motivational": "I feel discouraged about studying today. Help me continue:",
    }
    return f"{prompt_map[style]} sample-{idx}"


def _synthetic_response(style: str, idx: int) -> str:
    response_map = {
        "direct": "First identify known values, then apply the formula, and finally verify the result.",
        "socratic": "What do you already know about the relationship? Why might that matter here?",
        "scaffolding": "Great start. Try computing the first small step, then we can build the next step together.",
        "feedback": "Your structure is strong, but clarify reasoning in the second paragraph and add one concrete example.",
        "motivational": "You are making real progress. Keep going one small step at a time; you can do this.",
    }
    return f"{response_map[style]} (synthetic-{idx})"


def _pad_with_synthetic(
    style: str,
    existing_samples: list[UnifiedSample],
    min_samples: int,
    source_name: str,
) -> tuple[list[UnifiedSample], int]:
    padded = list(existing_samples)
    required = max(0, min_samples - len(padded))
    for idx in range(required):
        padded.append(
            UnifiedSample(
                instruction=_synthetic_instruction(style, idx),
                response=_synthetic_response(style, idx),
                style_label=style,
                source=source_name,
                metadata={"synthetic": True},
            )
        )
    return padded, required


def prepare_datasets(config: ProjectConfig) -> dict:
    ensure_project_dirs(config)
    seed_everything(config.seed)

    style_to_samples: dict[str, list[UnifiedSample]] = defaultdict(list)
    dataset_stats: dict[str, dict] = {}

    for spec in DATASET_SPECS:
        source_dir = build_source_path(config.raw_data_dir, spec)
        file_count = 0
        pair_count = 0
        accepted_count = 0

        for file_path in _iter_dataset_files(source_dir) or []:
            file_count += 1
            for instruction, response in _read_candidates_from_file(file_path):
                pair_count += 1
                style = _assign_style(
                    dataset_name=spec.dataset_name,
                    default_style=spec.style_label,
                    instruction=instruction,
                    response=response,
                )
                sample = sanitize_sample(
                    UnifiedSample(
                        instruction=instruction,
                        response=response,
                        style_label=style,
                        source=f"{spec.dataset_name}:{file_path.name}",
                        metadata={"dataset": spec.dataset_name},
                    )
                )
                if not is_quality_sample(sample):
                    continue
                style_to_samples[style].append(sample)
                accepted_count += 1

        dataset_stats[spec.dataset_name] = {
            "source_dir": str(source_dir),
            "files_found": file_count,
            "pairs_parsed": pair_count,
            "samples_accepted": accepted_count,
        }

    # Motivational and Direct can both originate from TSCC. Make sure both exist.
    for style in STYLE_LABELS:
        style_to_samples.setdefault(style, [])

    # Deduplicate and clamp upper bound first.
    for style in STYLE_LABELS:
        style_to_samples[style] = deduplicate(style_to_samples[style])
        if len(style_to_samples[style]) > config.max_samples_per_style:
            style_to_samples[style] = style_to_samples[style][: config.max_samples_per_style]

    synthetic_counts: dict[str, int] = {}
    for style in STYLE_LABELS:
        style_to_samples[style], added = _pad_with_synthetic(
            style=style,
            existing_samples=style_to_samples[style],
            min_samples=config.min_samples_per_style,
            source_name="synthetic:fallback",
        )
        synthetic_counts[style] = added

    all_records: list[dict] = []
    for style in STYLE_LABELS:
        for sample in style_to_samples[style]:
            all_records.append(sample.to_dict())

    train_records, eval_records, test_records = split_by_style(all_records)

    # Guarantee each style has at least one train sample.
    train_style_counts = Counter(item["style_label"] for item in train_records)
    for style in STYLE_LABELS:
        if train_style_counts.get(style, 0) > 0:
            continue
        fallback = next((x for x in all_records if x["style_label"] == style), None)
        if fallback is not None:
            train_records.append(fallback)

    write_jsonl_records(config.processed_train_path, train_records)
    write_jsonl_records(config.processed_eval_path, eval_records)
    write_jsonl_records(config.processed_test_path, test_records)

    for style in STYLE_LABELS:
        per_style_records = [item for item in train_records if item["style_label"] == style]
        write_jsonl_records(config.processed_data_dir / f"train_{style}.jsonl", per_style_records)

    style_counts = {
        style: {
            "total": len(style_to_samples[style]),
            "train": sum(1 for item in train_records if item["style_label"] == style),
            "eval": sum(1 for item in eval_records if item["style_label"] == style),
            "test": sum(1 for item in test_records if item["style_label"] == style),
            "synthetic_added": synthetic_counts[style],
        }
        for style in STYLE_LABELS
    }

    summary = {
        "dataset_stats": dataset_stats,
        "style_counts": style_counts,
        "processed_paths": {
            "train": str(config.processed_train_path),
            "eval": str(config.processed_eval_path),
            "test": str(config.processed_test_path),
        },
    }

    provenance_path = config.logs_dir / "data_provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary["provenance_path"] = str(provenance_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multi-style tutoring datasets")
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = ProjectConfig()
    if args.min_samples is not None:
        config.min_samples_per_style = args.min_samples
    if args.max_samples is not None:
        config.max_samples_per_style = args.max_samples

    summary = prepare_datasets(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
