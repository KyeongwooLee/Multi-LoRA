from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import ProjectConfig


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


@pytest.fixture()
def test_config(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"

    tscc_items = [
        {
            "instruction": "Please solve this algebra question and explain each calculation step in order.",
            "response": "Great, first isolate the variable, second simplify both sides, and third verify the final value.",
        },
        {
            "instruction": "I am worried I will fail this exam. Help me keep studying tonight.",
            "response": "You can do this. Keep going with one small step now, and your confidence will grow.",
        },
    ]
    socratic_items = [
        {
            "prompt": "I do not understand why acceleration changes when force changes.",
            "answer": "What relationship do you already know between force and acceleration? Why might mass matter?",
        }
    ]
    eedi_items = [
        {
            "question": "I can start this geometry question but I get stuck in the middle.",
            "output": "Nice start. Try finding one angle first, then use that as a hint for the next step.",
        }
    ]
    feedback_items = [
        {
            "instruction": "Here is my essay answer about climate change causes.",
            "response": "Your structure is clear, but improve evidence quality and add one counterargument paragraph.",
        }
    ]

    _write_jsonl(raw_dir / "tscc" / "sample.jsonl", tscc_items)
    _write_jsonl(raw_dir / "socraticlm" / "sample.jsonl", socratic_items)
    _write_jsonl(raw_dir / "eedi" / "sample.jsonl", eedi_items)
    _write_jsonl(raw_dir / "feedback_prize" / "sample.jsonl", feedback_items)

    artifacts_dir = tmp_path / "artifacts"
    processed_dir = data_dir / "processed"

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("RAW_DATA_DIR", str(raw_dir))
    monkeypatch.setenv("PROCESSED_DATA_DIR", str(processed_dir))
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("ADAPTERS_DIR", str(artifacts_dir / "adapters"))
    monkeypatch.setenv("PERSONA_ADAPTERS_DIR", str(artifacts_dir / "adapters" / "persona"))
    monkeypatch.setenv("ROUTER_DIR", str(artifacts_dir / "router"))
    monkeypatch.setenv("LOGS_DIR", str(artifacts_dir / "logs"))
    monkeypatch.setenv("REPORTS_DIR", str(artifacts_dir / "reports"))

    monkeypatch.setenv("ENABLE_REAL_TRAINING", "0")
    monkeypatch.setenv("ENABLE_REAL_GENERATION", "0")
    monkeypatch.setenv("ENABLE_REAL_ROUTER_EMBEDDING", "0")
    monkeypatch.setenv("LOCAL_FILES_ONLY", "1")

    monkeypatch.setenv("MIN_SAMPLES_PER_STYLE", "2")
    monkeypatch.setenv("MAX_SAMPLES_PER_STYLE", "5")
    monkeypatch.setenv("EVAL_SAMPLE_SIZE", "2")
    monkeypatch.setenv("TRAIN_MAX_EXAMPLES", "3")

    return ProjectConfig()
