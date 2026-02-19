from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RAW_DATA_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_PROCESSED_DATA_DIR = DEFAULT_DATA_DIR / "processed"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_ADAPTERS_DIR = DEFAULT_ARTIFACTS_DIR / "adapters"
DEFAULT_PERSONA_ADAPTERS_DIR = DEFAULT_ADAPTERS_DIR / "persona"
DEFAULT_ROUTER_DIR = DEFAULT_ARTIFACTS_DIR / "router"
DEFAULT_LOGS_DIR = DEFAULT_ARTIFACTS_DIR / "logs"
DEFAULT_REPORTS_DIR = DEFAULT_ARTIFACTS_DIR / "reports"

STYLE_LABELS: tuple[str, ...] = (
    "direct",
    "socratic",
    "scaffolding",
    "feedback",
    "motivational",
)

STYLE_DISPLAY_NAMES: dict[str, str] = {
    "direct": "Direct",
    "socratic": "Socratic",
    "scaffolding": "Scaffolding",
    "feedback": "Feedback",
    "motivational": "Motivational",
}

STYLE_ALIASES: dict[str, str] = {
    "direct_instruction": "direct",
    "direct": "direct",
    "socratic": "socratic",
    "scaffolding": "scaffolding",
    "feedback": "feedback",
    "motivational": "motivational",
}


def normalize_style_label(value: str) -> str:
    normalized = (value or "").strip().lower()
    return STYLE_ALIASES.get(normalized, normalized)


@dataclass
class ProjectConfig:
    base_model_name: str = field(
        default_factory=lambda: os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct")
    )
    router_embedding_model: str = field(
        default_factory=lambda: os.getenv("ROUTER_EMBEDDING_MODEL", "BAAI/bge-m3")
    )

    local_files_only: bool = field(
        default_factory=lambda: os.getenv("LOCAL_FILES_ONLY", "1") == "1"
    )
    use_4bit: bool = field(default_factory=lambda: os.getenv("USE_4BIT", "1") == "1")

    # Real training/inference can be expensive. Keep a safe default and allow opt-in.
    enable_real_training: bool = field(
        default_factory=lambda: os.getenv("ENABLE_REAL_TRAINING", "0") == "1"
    )
    enable_real_generation: bool = field(
        default_factory=lambda: os.getenv("ENABLE_REAL_GENERATION", "0") == "1"
    )
    enable_real_router_embedding: bool = field(
        default_factory=lambda: os.getenv("ENABLE_REAL_ROUTER_EMBEDDING", "0") == "1"
    )

    train_batch_size: int = field(default_factory=lambda: int(os.getenv("TRAIN_BATCH_SIZE", "1")))
    gradient_accumulation_steps: int = field(
        default_factory=lambda: int(os.getenv("GRAD_ACC_STEPS", "8"))
    )
    learning_rate: float = field(default_factory=lambda: float(os.getenv("LEARNING_RATE", "2e-4")))
    num_train_epochs: float = field(
        default_factory=lambda: float(os.getenv("NUM_TRAIN_EPOCHS", "1"))
    )
    warmup_ratio: float = field(default_factory=lambda: float(os.getenv("WARMUP_RATIO", "0.03")))
    max_seq_length: int = field(default_factory=lambda: int(os.getenv("MAX_SEQ_LENGTH", "512")))
    max_new_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_NEW_TOKENS", "256")))
    train_max_examples: int = field(
        default_factory=lambda: int(os.getenv("TRAIN_MAX_EXAMPLES", "1200"))
    )

    lora_rank: int = field(default_factory=lambda: int(os.getenv("LORA_RANK", "16")))
    lora_rank_fallback: int = field(
        default_factory=lambda: int(os.getenv("LORA_RANK_FALLBACK", "8"))
    )
    lora_alpha: int = field(default_factory=lambda: int(os.getenv("LORA_ALPHA", "32")))
    lora_dropout: float = field(default_factory=lambda: float(os.getenv("LORA_DROPOUT", "0.05")))

    router_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("ROUTER_CONF_THRESHOLD", "0.30"))
    )
    router_hash_dim: int = field(default_factory=lambda: int(os.getenv("ROUTER_HASH_DIM", "384")))

    # Production target from spec is 10k~30k. Smaller defaults keep local tests tractable.
    min_samples_per_style: int = field(
        default_factory=lambda: int(os.getenv("MIN_SAMPLES_PER_STYLE", "100"))
    )
    max_samples_per_style: int = field(
        default_factory=lambda: int(os.getenv("MAX_SAMPLES_PER_STYLE", "30000"))
    )
    eval_sample_size: int = field(default_factory=lambda: int(os.getenv("EVAL_SAMPLE_SIZE", "10")))

    seed: int = field(default_factory=lambda: int(os.getenv("SEED", "42")))

    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", str(DEFAULT_DATA_DIR))))
    raw_data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("RAW_DATA_DIR", str(DEFAULT_RAW_DATA_DIR)))
    )
    processed_data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PROCESSED_DATA_DIR", str(DEFAULT_PROCESSED_DATA_DIR))
        )
    )
    artifacts_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ARTIFACTS_DIR", str(DEFAULT_ARTIFACTS_DIR)))
    )
    adapters_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ADAPTERS_DIR", str(DEFAULT_ADAPTERS_DIR)))
    )
    persona_adapters_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PERSONA_ADAPTERS_DIR", str(DEFAULT_PERSONA_ADAPTERS_DIR))
        )
    )
    router_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ROUTER_DIR", str(DEFAULT_ROUTER_DIR)))
    )
    logs_dir: Path = field(default_factory=lambda: Path(os.getenv("LOGS_DIR", str(DEFAULT_LOGS_DIR))))
    reports_dir: Path = field(
        default_factory=lambda: Path(os.getenv("REPORTS_DIR", str(DEFAULT_REPORTS_DIR)))
    )

    processed_train_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("PROCESSED_TRAIN_PATH", str(DEFAULT_PROCESSED_DATA_DIR / "train.jsonl"))
        )
    )
    processed_eval_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("PROCESSED_EVAL_PATH", str(DEFAULT_PROCESSED_DATA_DIR / "eval.jsonl"))
        )
    )
    processed_test_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("PROCESSED_TEST_PATH", str(DEFAULT_PROCESSED_DATA_DIR / "test.jsonl"))
        )
    )

    def __post_init__(self) -> None:
        if "RAW_DATA_DIR" not in os.environ:
            self.raw_data_dir = self.data_dir / "raw"
        if "PROCESSED_DATA_DIR" not in os.environ:
            self.processed_data_dir = self.data_dir / "processed"
        if "ADAPTERS_DIR" not in os.environ:
            self.adapters_dir = self.artifacts_dir / "adapters"
        if "PERSONA_ADAPTERS_DIR" not in os.environ:
            self.persona_adapters_dir = self.adapters_dir / "persona"
        if "ROUTER_DIR" not in os.environ:
            self.router_dir = self.artifacts_dir / "router"
        if "LOGS_DIR" not in os.environ:
            self.logs_dir = self.artifacts_dir / "logs"
        if "REPORTS_DIR" not in os.environ:
            self.reports_dir = self.artifacts_dir / "reports"

        if "PROCESSED_TRAIN_PATH" not in os.environ:
            self.processed_train_path = self.processed_data_dir / "train.jsonl"
        if "PROCESSED_EVAL_PATH" not in os.environ:
            self.processed_eval_path = self.processed_data_dir / "eval.jsonl"
        if "PROCESSED_TEST_PATH" not in os.environ:
            self.processed_test_path = self.processed_data_dir / "test.jsonl"

        if self.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def ensure_project_dirs(config: ProjectConfig) -> None:
    for path in (
        config.data_dir,
        config.raw_data_dir,
        config.processed_data_dir,
        config.artifacts_dir,
        config.adapters_dir,
        config.persona_adapters_dir,
        config.router_dir,
        config.logs_dir,
        config.reports_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def dataclass_to_dict(config: ProjectConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.__dict__.items()
    }
