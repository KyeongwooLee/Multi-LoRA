from __future__ import annotations

from src.data.ingest import prepare_datasets
from src.eval.eval_correctness import run_correctness_eval
from src.eval.eval_personalization import run_personalization_eval
from src.eval.eval_system import run_system_eval
from src.routing.train_router import train_router
from src.training.train_persona_lora import train_all_styles


def test_eval_outputs_exist(test_config):
    prepare_datasets(test_config)
    train_all_styles(config=test_config, max_examples=2)
    train_router(test_config)

    correctness = run_correctness_eval(test_config, sample_size=2)
    personalization = run_personalization_eval(test_config, sample_size=2)
    system_eval = run_system_eval(test_config, query="Explain inertia simply.", num_runs=2)

    assert correctness["sample_count"] >= 1
    assert personalization["sample_count"] >= 1
    assert system_eval["num_runs"] == 2

    assert (test_config.reports_dir / "correctness_eval.json").exists()
    assert (test_config.reports_dir / "personalization_eval.json").exists()
    assert (test_config.reports_dir / "system_eval.json").exists()
