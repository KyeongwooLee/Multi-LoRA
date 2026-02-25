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

    gsm8k_items = [
        {
            "question": "Natalia sold 48 clips in April and half as many in May. How many clips in total?",
            "answer": "Natalia sold 24 clips in May and 72 clips in total.",
        },
        {
            "question": "Weng earns 12 dollars per hour and worked 50 minutes. How much did she earn?",
            "answer": "Weng earns 0.2 dollars per minute, so she earned 10 dollars.",
        },
    ]
    socrateach_multi = {
        "GSM8K_train_0": {
            "question": "Find total clips sold in April and May.",
            "analysis": "Half of 48 is 24; 48 + 24 = 72.",
            "answer": "72",
            "steps": ["Find May clips", "Add totals"],
            "dialogues": {
                "GSM8K_train_0_0": [
                    {
                        "system": "What does half as many mean for 48?",
                        "user": "It means divide 48 by 2.",
                        "user_type": "(1)",
                    },
                    {
                        "system": "Great. Now add April and May to get the total.",
                        "user": "48 plus 24 equals 72.",
                        "user_type": "(2)",
                    },
                ]
            },
        }
    }
    eedi_items = [
        {
            "question": "I can start this geometry question but I get stuck in the middle.",
            "output": "Nice start. Try finding one angle first, then use that as a hint for the next step.",
        }
    ]
    socrateach_single = {
        "incorrect#GSM8K_train_0_0_4@0": {
            "prompt": "I think the total is 79.9.",
            "response": "Check the sum of 48 and 24 again and focus on arithmetic accuracy.",
            "history": ["..."],
        },
        "correct#GSM8K_train_0_0_4@0": {
            "prompt": "The total is 72 clips.",
            "response": "Correct result. Next, explain why May is half of April in one sentence.",
            "history": ["..."],
        },
    }

    _write_jsonl(raw_dir / "GSM8K.jsonl", gsm8k_items)
    _write_jsonl(raw_dir / "Eedi.jsonl", eedi_items)
    (raw_dir / "SocraTeach_multi.json").write_text(
        json.dumps(socrateach_multi, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (raw_dir / "SocraTeach_single.json").write_text(
        json.dumps(socrateach_single, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

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

    monkeypatch.setenv("MIN_SAMPLES_PER_STYLE", "1")
    monkeypatch.setenv("MAX_SAMPLES_PER_STYLE", "5")
    monkeypatch.setenv("EVAL_SAMPLE_SIZE", "2")
    monkeypatch.setenv("TRAIN_MAX_EXAMPLES", "3")

    return ProjectConfig()
