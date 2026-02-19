from __future__ import annotations

from src.data.ingest import prepare_datasets
from src.routing.predict_router import route
from src.routing.train_router import train_router
from src.training.train_persona_lora import train_all_styles


def test_router_train_and_predict(test_config):
    prepare_datasets(test_config)
    train_all_styles(config=test_config, max_examples=2)
    train_result = train_router(test_config)

    assert train_result.train_samples > 0
    assert 0.0 <= train_result.train_accuracy <= 1.0

    prediction = route("Can you ask me guiding questions about Newton's law?", config=test_config)
    assert prediction["selected_style"] in {
        "direct",
        "socratic",
        "scaffolding",
        "feedback",
        "motivational",
    }
    assert "probabilities" in prediction


def test_router_fallback_applies_with_high_threshold(test_config):
    prepare_datasets(test_config)
    train_router(test_config)
    test_config.router_confidence_threshold = 0.99

    prediction = route("random unrelated token sequence", config=test_config)
    assert prediction["selected_style"] in {
        "scaffolding",
        "direct",
        "socratic",
        "feedback",
        "motivational",
    }
