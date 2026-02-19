from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.config import ProjectConfig, STYLE_DISPLAY_NAMES, ensure_project_dirs
from src.routing.predict_router import route


STYLE_SYSTEM_PROMPTS = {
    "direct": "Give a clear answer with key concepts and step-by-step explanation.",
    "socratic": "Guide the student with questions so they discover the answer.",
    "scaffolding": "Provide partial hints and gradual guidance between direct and socratic style.",
    "feedback": "Evaluate the student's answer and provide concrete improvements.",
    "motivational": "Encourage the student and maintain confidence while teaching.",
}


def _mock_generate(query: str, style: str) -> str:
    if style == "direct":
        return (
            f"핵심 개념부터 정리할게요: {query}. "
            "1) 문제를 작은 단계로 나눕니다. 2) 각 단계를 계산/추론합니다. 3) 최종 결과를 검증합니다."
        )
    if style == "socratic":
        return (
            f"좋아요. 먼저 '{query}'에서 이미 알고 있는 조건은 무엇인가요? "
            "그 조건이 답을 찾는 데 어떤 힌트를 줄까요?"
        )
    if style == "scaffolding":
        return (
            f"좋은 출발이에요. '{query}'를 바로 풀기 전에 첫 단계만 같이 해볼게요. "
            "작은 단서 하나를 적용한 뒤 다음 단계를 이어가면 됩니다."
        )
    if style == "feedback":
        return (
            f"'{query}'에 대한 답변을 기준으로 보면, 구조는 좋지만 근거를 더 명확히 쓰면 좋아요. "
            "특히 두 번째 단락에 예시를 추가해보세요."
        )
    return (
        f"'{query}'를 충분히 해낼 수 있어요. 지금처럼 한 단계씩 진행하면 됩니다. "
        "이미 중요한 포인트를 잘 잡고 있으니 계속 가봅시다."
    )


def _resolve_device(model):
    try:
        import torch

        if hasattr(model, "device"):
            return model.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return None


def _real_generate(query: str, style: str, config: ProjectConfig) -> tuple[str, str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_kwargs = {"local_files_only": config.local_files_only}
    if config.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        local_files_only=config.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.base_model_name, **model_kwargs)

    adapter_dir = config.persona_adapters_dir / style
    adapter_meta_path = adapter_dir / "adapter_meta.json"
    adapter_mode = "none"
    if adapter_meta_path.exists() and (adapter_dir / "adapter_config.json").exists():
        adapter_meta = json.loads(adapter_meta_path.read_text(encoding="utf-8"))
        if adapter_meta.get("training_mode") == "hf":
            model = PeftModel.from_pretrained(model, str(adapter_dir))
            adapter_mode = "hf_adapter"
        else:
            adapter_mode = "mock_adapter"

    system_prompt = STYLE_SYSTEM_PROMPTS.get(style, STYLE_SYSTEM_PROMPTS["scaffolding"])
    full_prompt = (
        f"### System\n{system_prompt}\n\n"
        f"### User\n{query}\n\n"
        "### Assistant\n"
    )

    device = _resolve_device(model)
    inputs = tokenizer(full_prompt, return_tensors="pt")
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "### Assistant" in generated_text:
        generated_text = generated_text.split("### Assistant", 1)[1].strip()
    return generated_text.strip(), adapter_mode


def run_inference(query: str, config: ProjectConfig) -> dict:
    ensure_project_dirs(config)

    started = time.perf_counter()
    router_result = route(query=query, config=config)
    selected_style = router_result["selected_style"]

    generation_mode = "mock"
    adapter_mode = "none"

    if config.enable_real_generation:
        try:
            response, adapter_mode = _real_generate(query=query, style=selected_style, config=config)
            generation_mode = "hf"
        except Exception as error:  # pragma: no cover - runtime dependent
            response = _mock_generate(query=query, style=selected_style)
            generation_mode = f"mock_fallback:{error}"
    else:
        response = _mock_generate(query=query, style=selected_style)

    latency_ms = (time.perf_counter() - started) * 1000.0

    gpu_mem_peak_mb = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            gpu_mem_peak_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        gpu_mem_peak_mb = 0.0

    result = {
        "query": query,
        "selected_style": selected_style,
        "selected_style_display": STYLE_DISPLAY_NAMES.get(selected_style, selected_style),
        "router": router_result,
        "response": response,
        "latency_ms": latency_ms,
        "gpu_mem_peak_mb": gpu_mem_peak_mb,
        "generation_mode": generation_mode,
        "adapter_mode": adapter_mode,
    }

    output_path = config.logs_dir / "latest_inference.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["log_path"] = str(output_path)
    return result


def generate(query: str, selected_style: str, config: ProjectConfig | None = None) -> str:
    # API compatibility helper requested in plan.
    _ = config
    if selected_style not in STYLE_SYSTEM_PROMPTS:
        selected_style = "scaffolding"
    return _mock_generate(query=query, style=selected_style)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run routed inference")
    parser.add_argument("query", type=str, help="User query string")
    args = parser.parse_args()

    config = ProjectConfig()
    result = run_inference(query=args.query, config=config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
