from __future__ import annotations

from src.data.ingest import prepare_datasets
from src.training.train_persona_lora import train_all_styles, train_style_adapter


def test_train_style_adapter_creates_artifacts(test_config):
    prepare_datasets(test_config)
    result = train_style_adapter("direct", config=test_config, max_examples=2)

    assert result.style == "direct"
    assert result.sample_count > 0
    assert result.training_mode in {"mock", "hf"}

    output_dir = test_config.persona_adapters_dir / "direct"
    assert (output_dir / "adapter_config.json").exists()
    assert (output_dir / "adapter_meta.json").exists()
    assert (output_dir / "train_metrics.json").exists()


def test_train_all_styles_runs(test_config):
    prepare_datasets(test_config)
    results = train_all_styles(config=test_config, max_examples=2)

    assert set(results.keys()) == {"direct", "socratic", "scaffolding", "feedback", "motivational"}
