from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import ProjectConfig, STYLE_LABELS, ensure_project_dirs, seed_everything
from src.data.ingest import load_jsonl_records


@dataclass
class PersonaTrainingResult:
    style: str
    output_dir: str
    sample_count: int
    rank: int
    training_mode: str
    duration_sec: float
    metrics_path: str


def _style_train_path(config: ProjectConfig, style: str) -> Path:
    return config.processed_data_dir / f"train_{style}.jsonl"


def _load_style_records(config: ProjectConfig, style: str, max_examples: int | None = None) -> list[dict]:
    style_path = _style_train_path(config, style)
    records = load_jsonl_records(style_path)
    if not records:
        records = [
            item
            for item in load_jsonl_records(config.processed_train_path)
            if item.get("style_label") == style
        ]
    if max_examples is not None:
        records = records[:max_examples]
    return records


def _format_training_text(record: dict) -> str:
    instruction = record.get("instruction", "")
    response = record.get("response", "")
    return f"### Instruction\n{instruction}\n\n### Response\n{response}"


def _is_oom_error(error: BaseException) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "cuda oom" in text


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_mock_adapter(
    output_dir: Path,
    style: str,
    sample_count: int,
    rank: int,
    config: ProjectConfig,
    reason: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_config = {
        "base_model_name_or_path": config.base_model_name,
        "r": rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "task_type": "CAUSAL_LM",
        "mock_adapter": True,
        "style": style,
    }
    _write_json(output_dir / "adapter_config.json", adapter_config)

    # Placeholder artifact for environments where real LoRA training is disabled.
    (output_dir / "adapter_model.safetensors").write_text(
        "mock adapter artifact\n",
        encoding="utf-8",
    )

    metrics = {
        "train_loss": 0.0,
        "samples": sample_count,
        "mode": "mock",
        "reason": reason,
    }
    _write_json(output_dir / "train_metrics.json", metrics)

    meta = {
        "style": style,
        "sample_count": sample_count,
        "rank": rank,
        "base_model": config.base_model_name,
        "training_mode": "mock",
        "reason": reason,
    }
    _write_json(output_dir / "adapter_meta.json", meta)
    _write_json(output_dir / "config_snapshot.json", {k: str(v) for k, v in config.__dict__.items()})
    return metrics


def _train_with_hf(
    output_dir: Path,
    style: str,
    records: list[dict],
    config: ProjectConfig,
    rank: int,
) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    class _SFTDataset(Dataset):
        def __init__(self, texts: list[str], tokenizer, max_length: int):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoded = self.tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),
            }

    model_kwargs = {"local_files_only": config.local_files_only}
    quant_config = None
    if config.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        local_files_only=config.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.base_model_name, **model_kwargs)

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )
    model = get_peft_model(model, lora_config)

    texts = [_format_training_text(record) for record in records]
    train_dataset = _SFTDataset(texts=texts, tokenizer=tokenizer, max_length=config.max_seq_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_outputs"),
        per_device_train_batch_size=config.train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    train_result = trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "train_loss": float(getattr(train_result, "training_loss", 0.0)),
        "samples": len(records),
        "mode": "hf",
    }
    _write_json(output_dir / "train_metrics.json", metrics)
    _write_json(
        output_dir / "adapter_meta.json",
        {
            "style": style,
            "sample_count": len(records),
            "rank": rank,
            "base_model": config.base_model_name,
            "training_mode": "hf",
            "use_4bit": config.use_4bit,
        },
    )
    _write_json(output_dir / "config_snapshot.json", {k: str(v) for k, v in config.__dict__.items()})
    return metrics


def train_style_adapter(
    style: str,
    config: ProjectConfig,
    max_examples: int | None = None,
) -> PersonaTrainingResult:
    if style not in STYLE_LABELS:
        raise ValueError(f"Unsupported style: {style}")

    ensure_project_dirs(config)
    seed_everything(config.seed)

    records = _load_style_records(
        config=config,
        style=style,
        max_examples=max_examples or config.train_max_examples,
    )
    if not records:
        raise RuntimeError(
            f"No training records found for style={style}. Run data ingest first."
        )

    output_dir = config.persona_adapters_dir / style
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    training_mode = "mock"
    rank_order = [config.lora_rank]
    if config.lora_rank_fallback != config.lora_rank:
        rank_order.append(config.lora_rank_fallback)

    metrics: dict = {}
    selected_rank = rank_order[-1]

    if not config.enable_real_training:
        selected_rank = rank_order[0]
        metrics = _write_mock_adapter(
            output_dir=output_dir,
            style=style,
            sample_count=len(records),
            rank=selected_rank,
            config=config,
            reason="real training disabled (ENABLE_REAL_TRAINING=0)",
        )
    else:
        last_error: Exception | None = None
        for rank in rank_order:
            try:
                selected_rank = rank
                metrics = _train_with_hf(
                    output_dir=output_dir,
                    style=style,
                    records=records,
                    config=config,
                    rank=rank,
                )
                training_mode = "hf"
                last_error = None
                break
            except Exception as error:  # pragma: no cover - runtime dependent
                last_error = error
                if _is_oom_error(error) and rank != rank_order[-1]:
                    continue
                metrics = _write_mock_adapter(
                    output_dir=output_dir,
                    style=style,
                    sample_count=len(records),
                    rank=rank,
                    config=config,
                    reason=f"fallback from HF training: {error}",
                )
                training_mode = "mock"
                break

        if last_error is not None and training_mode != "hf":
            training_mode = "mock"

    duration = time.perf_counter() - started_at
    metrics_path = output_dir / "train_metrics.json"
    if not metrics_path.exists():
        _write_json(metrics_path, metrics)

    return PersonaTrainingResult(
        style=style,
        output_dir=str(output_dir),
        sample_count=len(records),
        rank=selected_rank,
        training_mode=training_mode,
        duration_sec=duration,
        metrics_path=str(metrics_path),
    )


def train_all_styles(config: ProjectConfig, max_examples: int | None = None) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for style in STYLE_LABELS:
        result = train_style_adapter(style=style, config=config, max_examples=max_examples)
        results[style] = result.__dict__
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train persona LoRA adapters")
    parser.add_argument("--style", type=str, default="all", help="Target style label or 'all'")
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    config = ProjectConfig()

    if args.style == "all":
        payload = train_all_styles(config=config, max_examples=args.max_examples)
    else:
        payload = train_style_adapter(
            style=args.style,
            config=config,
            max_examples=args.max_examples,
        ).__dict__

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
