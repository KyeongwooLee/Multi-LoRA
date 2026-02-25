from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.config import ProjectConfig, STYLE_LABELS, ensure_project_dirs, seed_everything
from src.data.catalog import DATASET_SPECS, DatasetSpec, build_source_path
from src.data.dataset_preprocess import apply_dataset_preprocessing
from src.data.preprocess import UnifiedSample, deduplicate, is_quality_sample, sanitize_sample
from src.data.split import split_by_style


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


def _pick_first(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_pairs_from_item(item: dict[str, Any]) -> list[tuple[str, str]]:
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
            "problem",
            "student_question",
            "question_text",
            "query_text",
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
            "solution",
            "analysis",
            "explanation",
            "rationale",
            "teacher_response",
            "tutor_response",
            "assistant_response",
            "hint",
            "guidance",
            "next_step",
        ),
    )
    if prompt and response:
        return [(prompt, response)]

    for list_key in ("messages", "conversation", "dialogue", "turns", "anchored_dialogue"):
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
            text_a = _pick_first(current, ("content", "text", "utterance", "message", "query"))
            text_b = _pick_first(nxt, ("content", "text", "utterance", "message", "response"))
            if not text_a or not text_b:
                continue
            if role_a in {"user", "student", "human", "learner"} and role_b in {
                "assistant",
                "teacher",
                "bot",
                "tutor",
                "system",
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


def _iter_dataset_files(dataset_path: Path) -> list[Path]:
    if not dataset_path.exists():
        return []
    if dataset_path.is_file():
        if dataset_path.suffix.lower() in TEXT_FILE_SUFFIXES:
            return [dataset_path]
        return []

    return [
        path
        for path in dataset_path.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_FILE_SUFFIXES
    ]


def _local_samples_from_pairs(
    *,
    files: list[Path],
    style_label: str,
    dataset_name: str,
) -> tuple[list[UnifiedSample], dict[str, Any]]:
    samples: list[UnifiedSample] = []
    pair_count = 0
    for file_path in files:
        pairs = _read_candidates_from_file(file_path)
        pair_count += len(pairs)
        for instruction, response in pairs:
            samples.append(
                UnifiedSample(
                    instruction=instruction,
                    response=response,
                    style_label=style_label,
                    source=f"{dataset_name}:{file_path.name}",
                    metadata={"dataset": dataset_name, "source_file": file_path.name},
                )
            )
    stats = {
        "source": "local",
        "files_found": len(files),
        "pairs_parsed": pair_count,
    }
    return samples, stats


def _extract_socrateach_multi_samples(path: Path, style_label: str) -> tuple[list[UnifiedSample], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object.")

    samples: list[UnifiedSample] = []
    dialogues_seen = 0
    turns_seen = 0

    for problem_id, item in payload.items():
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        analysis = str(item.get("analysis", "")).strip()
        answer = str(item.get("answer", "")).strip()
        dialogues = item.get("dialogues")
        if not isinstance(dialogues, dict):
            continue

        for dialogue_id, turns in dialogues.items():
            if not isinstance(turns, list):
                continue
            dialogues_seen += 1
            for turn_index, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    continue
                user_text = str(turn.get("user", "")).strip()
                system_text = str(turn.get("system", "")).strip()
                user_type = str(turn.get("user_type", "")).strip()
                if not user_text or not system_text:
                    continue

                turns_seen += 1
                instruction_parts = []
                if question:
                    instruction_parts.append(f"Problem: {question}")
                instruction_parts.append(f"Student: {user_text}")
                instruction = "\n".join(instruction_parts)

                metadata: dict[str, Any] = {
                    "dataset": "socrateach_multi",
                    "problem_id": str(problem_id),
                    "dialogue_id": str(dialogue_id),
                    "turn_index": turn_index,
                }
                if user_type:
                    metadata["user_type"] = user_type
                if analysis:
                    metadata["reference_analysis"] = analysis
                if answer:
                    metadata["reference_answer"] = answer

                samples.append(
                    UnifiedSample(
                        instruction=instruction,
                        response=system_text,
                        style_label=style_label,
                        source=f"socrateach_multi:{path.name}",
                        metadata=metadata,
                    )
                )

    stats = {
        "source": "local",
        "files_found": 1,
        "pairs_parsed": len(samples),
        "dialogues_seen": dialogues_seen,
        "turns_seen": turns_seen,
    }
    return samples, stats


def _extract_socrateach_single_samples(path: Path, style_label: str) -> tuple[list[UnifiedSample], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object.")

    samples: list[UnifiedSample] = []
    for sample_id, item in payload.items():
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt", "")).strip()
        response = str(item.get("response", "")).strip()
        if not prompt or not response:
            continue

        tag = str(sample_id).split("#", 1)[0] if "#" in str(sample_id) else "unknown"
        metadata: dict[str, Any] = {
            "dataset": "socrateach_single",
            "sample_id": str(sample_id),
            "sample_type": tag,
        }
        history = item.get("history")
        if isinstance(history, list):
            metadata["history_turns"] = len(history)

        samples.append(
            UnifiedSample(
                instruction=prompt,
                response=response,
                style_label=style_label,
                source=f"socrateach_single:{path.name}",
                metadata=metadata,
            )
        )

    stats = {
        "source": "local",
        "files_found": 1,
        "pairs_parsed": len(samples),
    }
    return samples, stats


def _extract_gsm8k_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    question = _pick_first(row, ("question", "instruction", "prompt", "query"))
    analysis = _pick_first(row, ("analysis", "solution", "rationale", "explanation", "reasoning"))
    answer = _pick_first(row, ("answer", "final_answer", "label", "target"))
    if not question:
        return None

    if analysis and answer:
        response = analysis if answer in analysis else f"{analysis}\nFinal answer: {answer}"
        return question, response
    if analysis:
        return question, analysis
    if answer:
        return question, f"The final answer is {answer}."
    return None


def _load_hf_dataset_rows(dataset_name: str, config_name: str) -> tuple[list[dict[str, Any]], str | None, str | None]:
    import os

    try:
        from datasets import load_dataset
    except Exception as error:  # pragma: no cover - depends on environment
        return [], None, f"datasets import failed: {error}"

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    try:
        dataset = load_dataset(dataset_name, config_name, token=hf_token)
    except Exception as error:  # pragma: no cover - depends on environment
        return [], None, f"load_dataset failed: {error}"

    try:
        if hasattr(dataset, "keys"):
            split_name = "train" if "train" in dataset else next(iter(dataset.keys()))
            rows = [dict(row) for row in dataset[split_name]]
            return rows, split_name, None
        rows = [dict(row) for row in dataset]
        return rows, "train", None
    except Exception as error:  # pragma: no cover - depends on environment
        return [], None, f"dataset iteration failed: {error}"


def _load_gsm8k_samples(config: ProjectConfig, spec: DatasetSpec) -> tuple[list[UnifiedSample], dict[str, Any], list[str]]:
    issues: list[str] = []

    local_candidates = [
        build_source_path(config.raw_data_dir, spec),
        config.raw_data_dir / "GSM8K.jsonl",
        config.raw_data_dir / "GSM8K.json",
        config.raw_data_dir / "gsm8k.jsonl",
        config.raw_data_dir / "gsm8k.json",
    ]
    seen: set[Path] = set()
    local_files: list[Path] = []
    for candidate in local_candidates:
        for file_path in _iter_dataset_files(candidate):
            if file_path in seen:
                continue
            seen.add(file_path)
            local_files.append(file_path)

    if local_files:
        samples: list[UnifiedSample] = []
        pair_count = 0
        for file_path in local_files:
            for instruction, response in _read_candidates_from_file(file_path):
                pair_count += 1
                samples.append(
                    UnifiedSample(
                        instruction=instruction,
                        response=response,
                        style_label=spec.style_label or "direct",
                        source=f"gsm8k:{file_path.name}",
                        metadata={"dataset": "gsm8k", "source_file": file_path.name},
                    )
                )
        return (
            samples,
            {
                "source": "local",
                "files_found": len(local_files),
                "pairs_parsed": pair_count,
            },
            issues,
        )

    rows, split_name, error = _load_hf_dataset_rows("openai/gsm8k", "main")
    if error is not None:
        issues.append(
            "GSM8K is missing locally and HuggingFace load failed. "
            "Provide data/raw/GSM8K.json(l) or enable datasets download. "
            f"detail={error}"
        )
        return [], {"source": "huggingface", "files_found": 0, "pairs_parsed": 0}, issues

    samples: list[UnifiedSample] = []
    for idx, row in enumerate(rows):
        pair = _extract_gsm8k_pair(row)
        if pair is None:
            continue
        instruction, response = pair
        samples.append(
            UnifiedSample(
                instruction=instruction,
                response=response,
                style_label=spec.style_label or "direct",
                source=f"gsm8k:hf:{split_name}",
                metadata={"dataset": "gsm8k", "hf_split": split_name, "row_index": idx},
            )
        )

    return (
        samples,
        {
            "source": "huggingface",
            "files_found": 0,
            "pairs_parsed": len(samples),
            "hf_split": split_name,
        },
        issues,
    )


def _load_eedi_samples(config: ProjectConfig, spec: DatasetSpec) -> tuple[list[UnifiedSample], dict[str, Any], list[str]]:
    issues: list[str] = []

    local_candidates = [
        build_source_path(config.raw_data_dir, spec),
        config.raw_data_dir / "Eedi.jsonl",
        config.raw_data_dir / "Eedi.json",
        config.raw_data_dir / "eedi.jsonl",
        config.raw_data_dir / "eedi.json",
    ]
    seen: set[Path] = set()
    local_files: list[Path] = []
    for candidate in local_candidates:
        for file_path in _iter_dataset_files(candidate):
            if file_path in seen:
                continue
            seen.add(file_path)
            local_files.append(file_path)

    if local_files:
        samples, stats = _local_samples_from_pairs(
            files=local_files,
            style_label=spec.style_label or "scaffolding",
            dataset_name="eedi",
        )
        return samples, stats, issues

    rows, split_name, error = _load_hf_dataset_rows(
        "Eedi/Question-Anchored-Tutoring-Dialogues-2k",
        "anchored-dialogues",
    )
    if error is not None:
        issues.append(
            "Eedi is missing locally and HuggingFace load failed. "
            "Provide data/raw/Eedi.json(l) or enable datasets download. "
            f"detail={error}"
        )
        return [], {"source": "huggingface", "files_found": 0, "pairs_parsed": 0}, issues

    samples: list[UnifiedSample] = []

    # Eedi HF dataset is one row per utterance.
    # Build student->tutor response pairs within each intervention timeline.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        intervention_id = str(row.get("InterventionId", ""))
        question_id = str(row.get("QuestionId_DQ", ""))
        if not intervention_id:
            continue
        grouped[(intervention_id, question_id)].append(row)

    for (intervention_id, question_id), utterances in grouped.items():
        sorted_rows = sorted(
            utterances,
            key=lambda x: int(x.get("MessageSequence", 0) or 0),
        )
        for idx in range(len(sorted_rows) - 1):
            current = sorted_rows[idx]
            nxt = sorted_rows[idx + 1]
            current_is_tutor = int(current.get("IsTutor", 0) or 0)
            next_is_tutor = int(nxt.get("IsTutor", 0) or 0)
            if current_is_tutor != 0 or next_is_tutor != 1:
                continue

            student_text = str(current.get("MessageString", "")).strip()
            tutor_text = str(nxt.get("MessageString", "")).strip()
            if not student_text or not tutor_text:
                continue

            instruction = (
                f"QuestionId: {question_id}\n"
                f"Student: {student_text}"
            )
            samples.append(
                UnifiedSample(
                    instruction=instruction,
                    response=tutor_text,
                    style_label=spec.style_label or "scaffolding",
                    source=f"eedi:hf:{split_name}",
                    metadata={
                        "dataset": "eedi",
                        "hf_split": split_name,
                        "intervention_id": intervention_id,
                        "question_id_dq": question_id,
                        "student_msg_seq": current.get("MessageSequence"),
                        "tutor_msg_seq": nxt.get("MessageSequence"),
                        "talk_move_prediction": nxt.get("TalkMovePrediction"),
                    },
                )
            )

    return (
        samples,
        {
            "source": "huggingface",
            "files_found": 0,
            "pairs_parsed": len(samples),
            "hf_split": split_name,
        },
        issues,
    )


def _load_socrateach_multi_samples(
    config: ProjectConfig,
    spec: DatasetSpec,
) -> tuple[list[UnifiedSample], dict[str, Any], list[str]]:
    issues: list[str] = []
    source_path = build_source_path(config.raw_data_dir, spec)
    if not source_path.exists():
        issues.append(
            f"SocraTeach_multi file is missing: {source_path}. "
            "Provide data/raw/SocraTeach_multi.json."
        )
        return [], {"source": "local", "files_found": 0, "pairs_parsed": 0}, issues

    samples, stats = _extract_socrateach_multi_samples(source_path, spec.style_label or "socratic")
    return samples, stats, issues


def _load_socrateach_single_samples(
    config: ProjectConfig,
    spec: DatasetSpec,
) -> tuple[list[UnifiedSample], dict[str, Any], list[str]]:
    issues: list[str] = []
    source_path = build_source_path(config.raw_data_dir, spec)
    if not source_path.exists():
        issues.append(
            f"SocraTeach_single file is missing: {source_path}. "
            "Provide data/raw/SocraTeach_single.json."
        )
        return [], {"source": "local", "files_found": 0, "pairs_parsed": 0}, issues

    samples, stats = _extract_socrateach_single_samples(source_path, spec.style_label or "feedback")
    return samples, stats, issues


def _load_samples_for_spec(
    config: ProjectConfig,
    spec: DatasetSpec,
) -> tuple[list[UnifiedSample], dict[str, Any], list[str]]:
    if spec.dataset_name == "gsm8k":
        return _load_gsm8k_samples(config, spec)
    if spec.dataset_name == "socrateach_multi":
        return _load_socrateach_multi_samples(config, spec)
    if spec.dataset_name == "eedi":
        return _load_eedi_samples(config, spec)
    if spec.dataset_name == "socrateach_single":
        return _load_socrateach_single_samples(config, spec)
    raise ValueError(f"Unsupported dataset spec: {spec.dataset_name}")


def prepare_datasets(config: ProjectConfig) -> dict:
    ensure_project_dirs(config)
    seed_everything(config.seed)

    style_to_samples: dict[str, list[UnifiedSample]] = defaultdict(list)
    dataset_stats: dict[str, dict[str, Any]] = {}
    dataset_issues: list[str] = []

    for spec in DATASET_SPECS:
        raw_samples, source_stats, issues = _load_samples_for_spec(config, spec)
        dataset_issues.extend(issues)

        accepted_samples: list[UnifiedSample] = []
        rejected_dataset_rules = 0
        rejected_quality = 0
        for sample in raw_samples:
            transformed = apply_dataset_preprocessing(sample)
            if transformed is None:
                rejected_dataset_rules += 1
                continue

            sanitized = sanitize_sample(transformed)
            if not is_quality_sample(sanitized):
                rejected_quality += 1
                continue
            accepted_samples.append(sanitized)
            style_to_samples[sanitized.style_label].append(sanitized)

        dataset_stats[spec.dataset_name] = {
            "source_path": str(build_source_path(config.raw_data_dir, spec)),
            "style_label": spec.style_label,
            "raw_samples": len(raw_samples),
            "accepted_samples": len(accepted_samples),
            "rejected_dataset_rules": rejected_dataset_rules,
            "rejected_quality": rejected_quality,
            **source_stats,
        }

    for style in STYLE_LABELS:
        style_to_samples.setdefault(style, [])

    for style in STYLE_LABELS:
        style_to_samples[style] = deduplicate(style_to_samples[style])
        if len(style_to_samples[style]) > config.max_samples_per_style:
            style_to_samples[style] = style_to_samples[style][: config.max_samples_per_style]

    insufficient_styles = {
        style: len(style_to_samples[style])
        for style in STYLE_LABELS
        if len(style_to_samples[style]) < config.min_samples_per_style
    }

    summary = {
        "dataset_stats": dataset_stats,
        "style_counts": {
            style: {
                "total": len(style_to_samples[style]),
            }
            for style in STYLE_LABELS
        },
        "issues": dataset_issues,
        "constraints": {
            "min_samples_per_style": config.min_samples_per_style,
            "max_samples_per_style": config.max_samples_per_style,
        },
    }

    if insufficient_styles:
        summary["insufficient_styles"] = insufficient_styles
        provenance_path = config.logs_dir / "data_provenance.json"
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        details = ", ".join(
            [f"{style}={count} (<{config.min_samples_per_style})" for style, count in insufficient_styles.items()]
        )
        issue_text = " | ".join(dataset_issues) if dataset_issues else "none"
        raise RuntimeError(
            "Insufficient training data. Synthetic/dummy fallback is disabled. "
            f"style_counts: {details}. dataset_issues: {issue_text}."
        )

    all_records: list[dict[str, Any]] = []
    for style in STYLE_LABELS:
        for sample in style_to_samples[style]:
            all_records.append(sample.to_dict())

    train_records, eval_records, test_records = split_by_style(all_records)

    # Preserve at least one train example per style from real collected data.
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

    summary["style_counts"] = {
        style: {
            "total": len(style_to_samples[style]),
            "train": sum(1 for item in train_records if item["style_label"] == style),
            "eval": sum(1 for item in eval_records if item["style_label"] == style),
            "test": sum(1 for item in test_records if item["style_label"] == style),
        }
        for style in STYLE_LABELS
    }
    summary["processed_paths"] = {
        "train": str(config.processed_train_path),
        "eval": str(config.processed_eval_path),
        "test": str(config.processed_test_path),
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
