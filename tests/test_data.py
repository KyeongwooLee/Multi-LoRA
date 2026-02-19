from __future__ import annotations

from src.config import STYLE_LABELS
from src.data.ingest import prepare_datasets


def test_prepare_datasets_outputs_files_and_counts(test_config):
    summary = prepare_datasets(test_config)

    assert test_config.processed_train_path.exists()
    assert test_config.processed_eval_path.exists()
    assert test_config.processed_test_path.exists()

    style_counts = summary["style_counts"]
    for style in STYLE_LABELS:
        assert style in style_counts
        assert style_counts[style]["total"] >= test_config.min_samples_per_style

    assert summary["provenance_path"]
