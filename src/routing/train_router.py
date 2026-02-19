from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.config import ProjectConfig, STYLE_LABELS, ensure_project_dirs, seed_everything
from src.data.ingest import load_jsonl_records
from src.routing.features import QueryEmbedder, cosine_similarity


@dataclass
class RouterTrainResult:
    router_path: str
    metrics_path: str
    backend: str
    train_samples: int
    train_accuracy: float


def _query_text(record: dict) -> str:
    instruction = record.get("instruction", "")
    metadata = record.get("metadata") or {}
    hints = " ".join([str(metadata.get("student_preference", "")), str(metadata.get("difficulty", ""))])
    return f"{instruction} {hints}".strip()


def _build_centroids(embeddings: list[list[float]], labels: list[str]) -> dict[str, list[float]]:
    sums: dict[str, list[float]] = {}
    counts: dict[str, int] = defaultdict(int)

    for embedding, label in zip(embeddings, labels):
        if label not in sums:
            sums[label] = [0.0 for _ in embedding]
        for idx, value in enumerate(embedding):
            sums[label][idx] += value
        counts[label] += 1

    centroids: dict[str, list[float]] = {}
    for label, vec_sum in sums.items():
        count = max(1, counts[label])
        centroids[label] = [value / count for value in vec_sum]
    return centroids


def _predict_with_centroids(embedding: list[float], centroids: dict[str, list[float]]) -> str:
    best_label = "scaffolding"
    best_score = float("-inf")
    for label, centroid in centroids.items():
        score = cosine_similarity(embedding, centroid)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def train_router(config: ProjectConfig) -> RouterTrainResult:
    ensure_project_dirs(config)
    seed_everything(config.seed)

    records = load_jsonl_records(config.processed_train_path)
    if not records:
        raise RuntimeError("No processed train data found. Run data ingest first.")

    records = [item for item in records if item.get("style_label") in STYLE_LABELS]
    if not records:
        raise RuntimeError("No valid style-labeled records found for router training.")

    texts = [_query_text(record) for record in records]
    labels = [record["style_label"] for record in records]

    embedder = QueryEmbedder(config)
    embeddings = embedder.encode(texts)
    centroids = _build_centroids(embeddings, labels)

    correct = 0
    for emb, gold in zip(embeddings, labels):
        pred = _predict_with_centroids(emb, centroids)
        if pred == gold:
            correct += 1

    train_accuracy = correct / max(1, len(labels))

    router_payload = {
        "styles": list(STYLE_LABELS),
        "centroids": centroids,
        "threshold": config.router_confidence_threshold,
        "fallback_style": "scaffolding",
        "embedder": {
            "backend": embedder.info.backend,
            "model_name": embedder.info.model_name,
        },
        "label_counts": dict(Counter(labels)),
    }

    router_path = config.router_dir / "router_model.json"
    router_path.parent.mkdir(parents=True, exist_ok=True)
    router_path.write_text(json.dumps(router_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_payload = {
        "train_samples": len(labels),
        "train_accuracy": train_accuracy,
        "backend": embedder.info.backend,
    }
    metrics_path = config.router_dir / "router_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return RouterTrainResult(
        router_path=str(router_path),
        metrics_path=str(metrics_path),
        backend=embedder.info.backend,
        train_samples=len(labels),
        train_accuracy=train_accuracy,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the style router")
    _ = parser.parse_args()

    config = ProjectConfig()
    result = train_router(config)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
