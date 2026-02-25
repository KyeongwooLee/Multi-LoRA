from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    style_label: str | None
    relative_dir: str
    description: str


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        dataset_name="gsm8k",
        style_label="direct",
        relative_dir="GSM8K.jsonl",
        description="GSM8K dataset (Direct style, local file or HuggingFace)",
    ),
    DatasetSpec(
        dataset_name="socrateach_multi",
        style_label="socratic",
        relative_dir="SocraTeach_multi.json",
        description="SocraTeach_multi dataset (Socratic style)",
    ),
    DatasetSpec(
        dataset_name="eedi",
        style_label="scaffolding",
        relative_dir="Eedi.jsonl",
        description="Eedi Question-Anchored Tutoring Dialogues (local file or HuggingFace)",
    ),
    DatasetSpec(
        dataset_name="socrateach_single",
        style_label="feedback",
        relative_dir="SocraTeach_single.json",
        description="SocraTeach_single dataset (Feedback style)",
    ),
)


def build_source_path(raw_data_root, spec: DatasetSpec):
    return raw_data_root / spec.relative_dir
