# Multi-LoRA Personal Tutor

This directory implements a complete Multi-LoRA pipeline for a personalized tutoring assistant.

## Features

- 5 persona adapters: `Direct`, `Socratic`, `Scaffolding`, `Feedback`, `Motivational`
- Dataset ingest/normalization with style filtering, quality filtering, dedup, and provenance logs
- Top-1 router (`BAAI/bge-m3` embedding backend with hash fallback)
- Routed inference (`query -> selected style adapter -> response`)
- Automated evaluation: correctness / personalization / system
- One-command end-to-end pipeline

## Quick Start

```bash
cd Multi-LoRA
python3 -m src.pipeline.run_full_pipeline --query "Explain photosynthesis simply."
```

Or via script:

```bash
bash scripts/run_full_pipeline.sh "Explain photosynthesis simply."
```

## Important Environment Variables

- `ENABLE_REAL_TRAINING=1`: enable actual LoRA fine-tuning (default `0`)
- `ENABLE_REAL_GENERATION=1`: enable base model + adapter generation (default `0`)
- `ENABLE_REAL_ROUTER_EMBEDDING=1`: enable actual `BAAI/bge-m3` embedding backend (default `0`)
- `MIN_SAMPLES_PER_STYLE`, `MAX_SAMPLES_PER_STYLE`: style dataset size bounds
- `BASE_MODEL_NAME`: default `Qwen/Qwen2.5-14B-Instruct`
- `LOCAL_FILES_ONLY`: offline-safe model loading toggle

## Entry Points

- `python -m src.data.ingest`
- `python -m src.training.train_persona_lora --style <style|all>`
- `python -m src.routing.train_router`
- `python -m src.inference.generate "<query>"`
- `python -m src.pipeline.run_full_pipeline --query "<query>"`
