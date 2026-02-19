from __future__ import annotations

import argparse
import json
import time

from src.config import ProjectConfig, ensure_project_dirs
from src.inference.generate import run_inference


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(round((len(sorted_values) - 1) * q))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


def run_system_eval(config: ProjectConfig, query: str, num_runs: int = 5) -> dict:
    ensure_project_dirs(config)

    latencies: list[float] = []
    gpu_mem_peaks: list[float] = []

    started = time.perf_counter()
    for _ in range(max(1, num_runs)):
        infer_result = run_inference(query=query, config=config)
        latencies.append(float(infer_result.get("latency_ms", 0.0)))
        gpu_mem_peaks.append(float(infer_result.get("gpu_mem_peak_mb", 0.0)))
    total_sec = max(1e-6, time.perf_counter() - started)

    result = {
        "num_runs": len(latencies),
        "latency_ms_p50": _percentile(latencies, 0.5),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "throughput_qps": len(latencies) / total_sec,
        "gpu_mem_peak_mb": max(gpu_mem_peaks) if gpu_mem_peaks else 0.0,
    }

    output_path = config.reports_dir / "system_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["path"] = str(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run system performance evaluation")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--num-runs", type=int, default=5)
    args = parser.parse_args()

    config = ProjectConfig()
    result = run_system_eval(config=config, query=args.query, num_runs=args.num_runs)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
