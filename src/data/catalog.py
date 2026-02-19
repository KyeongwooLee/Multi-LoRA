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
        dataset_name="tscc",
        style_label=None,
        relative_dir="tscc",
        description="Teacher-Student Chatroom Corpus (shared by Direct/Motivational)",
    ),
    DatasetSpec(
        dataset_name="socraticlm",
        style_label="socratic",
        relative_dir="socraticlm",
        description="SocraticLM dataset",
    ),
    DatasetSpec(
        dataset_name="eedi",
        style_label="scaffolding",
        relative_dir="eedi",
        description="Eedi Question-Anchored Tutoring Dialogues",
    ),
    DatasetSpec(
        dataset_name="feedback_prize",
        style_label="feedback",
        relative_dir="feedback_prize",
        description="Feedback Prize dataset",
    ),
)


def build_source_path(raw_data_root, spec: DatasetSpec):
    return raw_data_root / spec.relative_dir
