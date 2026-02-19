from __future__ import annotations

import json
import math
from pathlib import Path

from src.config import ProjectConfig
from src.routing.features import QueryEmbedder, cosine_similarity


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    exps = {label: math.exp(score - max_score) for label, score in scores.items()}
    total = sum(exps.values())
    if total <= 0:
        uniform = 1.0 / max(1, len(scores))
        return {label: uniform for label in scores}
    return {label: value / total for label, value in exps.items()}


class RouterPredictor:
    def __init__(self, config: ProjectConfig, router_path: Path | None = None):
        self.config = config
        self.router_path = router_path or (config.router_dir / "router_model.json")
        self.embedder = QueryEmbedder(config)
        self.model = self._load_model(self.router_path)

    @staticmethod
    def _load_model(path: Path) -> dict:
        if not path.exists():
            return {
                "styles": ["scaffolding"],
                "centroids": {},
                "threshold": 1.0,
                "fallback_style": "scaffolding",
            }
        return json.loads(path.read_text(encoding="utf-8"))

    def route(self, query: str) -> dict:
        text = (query or "").strip()
        if not text:
            return {
                "selected_style": "scaffolding",
                "probabilities": {"scaffolding": 1.0},
                "confidence": 1.0,
                "fallback_applied": True,
            }

        centroids: dict[str, list[float]] = self.model.get("centroids", {})
        if not centroids:
            return {
                "selected_style": "scaffolding",
                "probabilities": {"scaffolding": 1.0},
                "confidence": 1.0,
                "fallback_applied": True,
            }

        embedding = self.embedder.encode([text])[0]
        sims = {style: cosine_similarity(embedding, centroid) for style, centroid in centroids.items()}
        probabilities = _softmax(sims)

        selected_style = max(probabilities, key=probabilities.get)
        confidence = float(probabilities[selected_style])
        threshold = float(self.model.get("threshold", self.config.router_confidence_threshold))
        fallback_style = self.model.get("fallback_style", "scaffolding")
        fallback_applied = False

        if confidence < threshold:
            selected_style = fallback_style
            fallback_applied = True

        return {
            "selected_style": selected_style,
            "probabilities": probabilities,
            "confidence": confidence,
            "fallback_applied": fallback_applied,
        }


def route(query: str, config: ProjectConfig | None = None) -> dict:
    resolved_config = config or ProjectConfig()
    predictor = RouterPredictor(config=resolved_config)
    return predictor.route(query)
