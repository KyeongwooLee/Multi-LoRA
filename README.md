# Multi-LoRA Personal Tutor

This directory contains a full Multi-LoRA pipeline for a personalized tutoring assistant:

- 4 persona adapters: `direct`, `socratic`, `scaffolding`, `feedback`
- dataset ingest and preprocessing
- style router training
- routed inference and evaluation

## Project Structure

- `src/data`: dataset ingest and preprocessing
- `src/training`: style-wise LoRA training
- `src/routing`: router feature extraction, training, and prediction
- `src/inference`: routed generation
- `src/eval`: correctness/personalization/system evaluation
- `artifacts`: trained adapters, router model, logs, reports

## Setup

```bash
cd Multi-LoRA
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Raw Data Inputs

Expected under `data/raw`:

- `SocraTeach_multi.json` (socratic)
- `SocraTeach_single.json` (feedback)
- optional local `GSM8K.jsonl` or `GSM8K.json` (direct)
- optional local `Eedi.jsonl` or `Eedi.json` (scaffolding)

If local `GSM8K`/`Eedi` files are missing, ingest tries Hugging Face datasets.

## Dataset-Specific Preprocessing

Data-specific preprocessing is implemented in `src/data/dataset_preprocess.py` and applied during ingest.

- `gsm8k` -> `direct`
  - cleans calculation markers (`<<...>>`, `#### ...`)
  - normalizes direct-answer style instruction/response
- `socrateach_multi` -> `socratic`
  - reinforces socratic response shape (question-guided)
  - adds explicit socratic instruction context
- `eedi` -> `scaffolding`
  - removes emoji and low-information student utterances (`ok`, `yes`, `thx`, ...)
  - drops social-only tutor responses
  - enriches instruction with `QuestionId`, turn, and tutor-goal metadata
- `socrateach_single` -> `feedback`
  - formats instruction as student answer + feedback type
  - normalizes formative feedback response shape

Ingest summary now includes `rejected_dataset_rules` per dataset in:

- `artifacts/logs/data_provenance.json`

## Training Flow (Real LoRA + Real Router Embedding)

```bash
cd Multi-LoRA
source .venv/bin/activate
export PYTHONPATH=$(pwd)

export ENABLE_REAL_TRAINING=1
export ENABLE_REAL_ROUTER_EMBEDDING=1
export ENABLE_REAL_GENERATION=0

export LOCAL_FILES_ONLY=0
export USE_4BIT=1
```

1) Prepare processed data:

```bash
python3 -m src.data.ingest --min-samples 100 --max-samples 30000
```

2) Train all persona adapters:

```bash
python3 -m src.training.train_persona_lora --style all --max-examples 1200
```

3) Train router:

```bash
python3 -m src.routing.train_router
```

## Training Prompt Format

LoRA training now uses the same conversation framing as inference:

- `### System`
- `### User`
- `### Assistant`

This is handled in `src/training/train_persona_lora.py`.

## Artifacts Needed for Deployment

After training, these files are required:

- `artifacts/adapters/persona/<style>/adapter_config.json`
- `artifacts/adapters/persona/<style>/adapter_model.safetensors`
- `artifacts/adapters/persona/<style>/adapter_meta.json` (for verification)
- `artifacts/router/router_model.json`

For sanity checks:

- adapter training mode should be `hf` in each `adapter_meta.json`
- router backend should be `bge-m3` in `artifacts/router/router_metrics.json`

## Main Entry Points

- `python3 -m src.data.ingest`
- `python3 -m src.training.train_persona_lora --style <style|all>`
- `python3 -m src.routing.train_router`
- `python3 -m src.inference.generate "<query>"`
- `python3 -m src.pipeline.run_full_pipeline --query "<query>"`
