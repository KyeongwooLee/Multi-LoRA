from __future__ import annotations

from src.pipeline.run_full_pipeline import run_full_pipeline


def test_full_pipeline_generates_report(test_config):
    summary = run_full_pipeline(
        config=test_config,
        query="Please explain photosynthesis in a student-friendly way.",
    )

    assert summary["report_path"]
    assert (test_config.reports_dir / "final_report.json").exists()
    assert (test_config.logs_dir / "pipeline_summary.json").exists()
