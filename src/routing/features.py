from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from src.config import ProjectConfig


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0:
        return vec
    return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    dot = sum(a[idx] * b[idx] for idx in range(size))
    norm_a = math.sqrt(sum(x * x for x in a[:size]))
    norm_b = math.sqrt(sum(x * x for x in b[:size]))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class EmbedderInfo:
    backend: str
    model_name: str


class QueryEmbedder:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.info = EmbedderInfo(
            backend="hash",
            model_name="hashing-fallback",
        )

        if config.enable_real_router_embedding:
            self._try_load_bge()

    def _try_load_bge(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.router_embedding_model,
                local_files_only=self.config.local_files_only,
            )
            self.model = AutoModel.from_pretrained(
                self.config.router_embedding_model,
                local_files_only=self.config.local_files_only,
            )
            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.info = EmbedderInfo(
                backend="bge-m3",
                model_name=self.config.router_embedding_model,
            )
        except Exception:
            self.model = None
            self.tokenizer = None
            self.device = None
            self.info = EmbedderInfo(
                backend="hash",
                model_name="hashing-fallback",
            )

    def _encode_hash(self, text: str) -> list[float]:
        vec = [0.0 for _ in range(self.config.router_hash_dim)]
        tokens = TOKEN_PATTERN.findall(_normalize(text))
        if not tokens:
            return vec

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.config.router_hash_dim
            sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
            vec[idx] += sign

        return _l2_normalize(vec)

    def _encode_bge(self, texts: list[str]) -> list[list[float]]:
        import torch

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("BGE model is not available")

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            masked_hidden = hidden * mask
            summed = masked_hidden.sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu().tolist()

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.info.backend == "bge-m3":
            try:
                return self._encode_bge(texts)
            except Exception:
                self.info = EmbedderInfo(backend="hash", model_name="hashing-fallback")
        return [self._encode_hash(text) for text in texts]
