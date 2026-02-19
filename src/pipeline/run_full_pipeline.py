from __future__ import annotations

import argparse
import json

from src.config import ProjectConfig, dataclass_to_dict, ensure_project_dirs, seed_everything
from src.data.ingest import prepare_datasets
from src.eval.build_report import build_report
from src.eval.eval_correctness import run_correctness_eval
from src.eval.eval_personalization import run_personalization_eval
from src.eval.eval_system import run_system_eval
from src.inference.generate import run_inference
from src.routing.train_router import train_router
from src.training.train_persona_lora import train_all_styles


DEFAULT_QUERY = "Explain Newton's second law with an easy example."


def run_full_pipeline(config: ProjectConfig, query: str = DEFAULT_QUERY) -> dict:
    ensure_project_dirs(config)
    seed_everything(config.seed)

    data_summary = prepare_datasets(config)
    training_summary = train_all_styles(config=config, max_examples=config.train_max_examples)
    router_summary = train_router(config).__dict__
    inference_summary = run_inference(query=query, config=config)

    correctness_summary = run_correctness_eval(
        config=config,
        sample_size=config.eval_sample_size,
    )
    personalization_summary = run_personalization_eval(
        config=config,
        sample_size=config.eval_sample_size,
    )
    system_summary = run_system_eval(
        config=config,
        query=query,
        num_runs=max(3, min(config.eval_sample_size, 10)),
    )
    summary = {
        "config": dataclass_to_dict(config),
        "query": query,
        "data_summary": data_summary,
        "training_summary": training_summary,
        "router_summary": router_summary,
        "inference_summary": inference_summary,
        "correctness_summary": correctness_summary,
        "personalization_summary": personalization_summary,
        "system_summary": system_summary,
        "report_path": None,
    }

    summary_path = config.logs_dir / "pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_summary = build_report(config)
    summary["report_path"] = report_summary.get("path")
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Multi-LoRA full pipeline")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY)
    parser.add_argument("--eval-sample-size", type=int, default=None)
    parser.add_argument("--train-max-examples", type=int, default=None)
    args = parser.parse_args()

    config = ProjectConfig()
    if args.eval_sample_size is not None:
        config.eval_sample_size = args.eval_sample_size
    if args.train_max_examples is not None:
        config.train_max_examples = args.train_max_examples

    summary = run_full_pipeline(config=config, query=args.query)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
