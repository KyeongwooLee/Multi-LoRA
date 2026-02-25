from __future__ import annotations

import re
from typing import Any

from src.data.preprocess import UnifiedSample, normalize_whitespace


EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\u2600-\u27BF"
    "]",
    flags=re.UNICODE,
)

GSM8K_INLINE_CALC_RE = re.compile(r"<<([^<>]+)>>")
GSM8K_INLINE_DUP_VALUE_RE = re.compile(
    r"<<(?:[^<>]*=)?\s*([-+]?\d+(?:\.\d+)?)>>\s*\1"
)
GSM8K_FINAL_RE = re.compile(r"####\s*([^\n\r]+)")
GSM8K_FINAL_PREFIX_RE = re.compile(r"^\s*the final answer is\s*", flags=re.IGNORECASE)
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

LOW_INFO_STUDENT_UTTERANCES = {
    "ok",
    "okay",
    "yes",
    "no",
    "yeah",
    "yep",
    "thanks",
    "thankyou",
    "thx",
    "k",
}

SOCIAL_ONLY_PATTERNS = (
    "thank you",
    "thanks",
    "you are welcome",
    "you're welcome",
    "well done",
    "good job",
    "great work",
    "bye",
    "see you",
)

SOCRATIC_CUES = (
    "what",
    "why",
    "how",
    "can you",
    "could you",
    "do you",
    "which",
    "suppose",
)


def _strip_emojis(text: str) -> str:
    return EMOJI_RE.sub("", text or "")


def _clean_text(text: str) -> str:
    return normalize_whitespace(_strip_emojis(text))


def _normalized_short_utterance(text: str) -> str:
    lowered = (text or "").strip().lower()
    lowered = NON_ALNUM_RE.sub("", lowered)
    return lowered


def _extract_question_id(instruction: str, metadata: dict[str, Any]) -> str:
    value = metadata.get("question_id_dq")
    if value:
        return str(value).strip()
    match = re.search(r"QuestionId:\s*([^\s]+)", instruction or "")
    if match:
        return match.group(1).strip()
    return ""


def _extract_student_text(instruction: str) -> str:
    if "Student:" in instruction:
        _, student_text = instruction.split("Student:", 1)
        return student_text.strip()
    return instruction.strip()


def _is_social_only_response(response: str) -> bool:
    lowered = (response or "").lower()
    if "?" in lowered:
        return False
    words = [token for token in re.findall(r"[a-z0-9']+", lowered) if token]
    if len(words) > 10:
        return False
    return any(pattern in lowered for pattern in SOCIAL_ONLY_PATTERNS)


def _looks_socratic(text: str) -> bool:
    lowered = (text or "").lower()
    if "?" in lowered:
        return True
    return any(cue in lowered for cue in SOCRATIC_CUES)


def _transform_gsm8k(sample: UnifiedSample) -> UnifiedSample | None:
    instruction = _clean_text(sample.instruction)
    response = _clean_text(sample.response)
    if not instruction or not response:
        return None

    response = GSM8K_INLINE_DUP_VALUE_RE.sub(r"\1", response)

    def _replace_inline_calc(match: re.Match[str]) -> str:
        expr = match.group(1).strip()
        if "=" in expr:
            _, rhs = expr.rsplit("=", 1)
            rhs = rhs.strip()
            if rhs:
                return rhs
        return expr

    response = GSM8K_INLINE_CALC_RE.sub(_replace_inline_calc, response)
    final_answer = ""
    final_match = GSM8K_FINAL_RE.search(response)
    if final_match:
        final_answer = _clean_text(final_match.group(1))
        response = GSM8K_FINAL_RE.sub("", response)

    response = GSM8K_FINAL_PREFIX_RE.sub("", response)
    response = _clean_text(response).strip(" .")
    if final_answer:
        if response:
            response = f"{response}. Final answer: {final_answer}."
        else:
            response = f"Final answer: {final_answer}."

    if not response:
        return None

    metadata = dict(sample.metadata or {})
    metadata["preprocess_profile"] = "gsm8k_direct_v2"
    transformed_instruction = f"Solve directly with clear steps.\nProblem: {instruction}"
    return UnifiedSample(
        instruction=transformed_instruction,
        response=response,
        style_label=sample.style_label,
        source=sample.source,
        metadata=metadata,
    )


def _transform_socrateach_multi(sample: UnifiedSample) -> UnifiedSample | None:
    instruction = _clean_text(sample.instruction)
    response = _clean_text(sample.response)
    if not instruction or not response:
        return None

    if not _looks_socratic(response):
        response = f"{response} What do you think the next step should be?"

    metadata = dict(sample.metadata or {})
    metadata["preprocess_profile"] = "socrateach_multi_socratic_v2"

    transformed_instruction = (
        f"{instruction}\n"
        "Respond in Socratic style: guide with questions instead of giving the full answer."
    )
    return UnifiedSample(
        instruction=transformed_instruction,
        response=response,
        style_label=sample.style_label,
        source=sample.source,
        metadata=metadata,
    )


def _transform_eedi(sample: UnifiedSample) -> UnifiedSample | None:
    raw_instruction = sample.instruction or ""
    metadata = dict(sample.metadata or {})

    student_text = _clean_text(_extract_student_text(raw_instruction))
    response = _clean_text(sample.response)
    if not student_text or not response:
        return None

    short_key = _normalized_short_utterance(student_text)
    if short_key in LOW_INFO_STUDENT_UTTERANCES:
        return None
    if _is_social_only_response(response):
        return None

    question_id = _extract_question_id(raw_instruction, metadata)
    turn_index = metadata.get("student_msg_seq")
    talk_move = _clean_text(str(metadata.get("talk_move_prediction", "")).strip("<> "))

    instruction_lines: list[str] = []
    if question_id:
        instruction_lines.append(f"QuestionId: {question_id}")
    if turn_index not in (None, ""):
        instruction_lines.append(f"Turn: {turn_index}")
    instruction_lines.append(f"Student: {student_text}")
    if talk_move and talk_move.lower() not in {"none", "<none>"}:
        instruction_lines.append(f"Tutor goal: {talk_move}")

    transformed_instruction = "\n".join(instruction_lines)
    if "?" not in response:
        response = f"{response} What is one small step you can try next?"

    metadata["preprocess_profile"] = "eedi_scaffolding_v2"
    return UnifiedSample(
        instruction=transformed_instruction,
        response=response,
        style_label=sample.style_label,
        source=sample.source,
        metadata=metadata,
    )


def _transform_socrateach_single(sample: UnifiedSample) -> UnifiedSample | None:
    metadata = dict(sample.metadata or {})
    sample_type = str(metadata.get("sample_type", "")).strip().lower()

    student_answer = _clean_text(sample.instruction)
    response = _clean_text(sample.response)
    if not student_answer or not response:
        return None

    instruction_lines = [f"Student answer: {student_answer}"]
    if sample_type:
        instruction_lines.append(f"Feedback type: {sample_type}")
    instruction_lines.append("Provide concise formative feedback with one concrete next step.")
    transformed_instruction = "\n".join(instruction_lines)

    lowered = response.lower()
    if sample_type == "correct":
        if not any(token in lowered for token in ("good", "great", "well done", "correct", "nice")):
            response = f"Good work. {response}"
    elif sample_type == "incorrect":
        if not any(token in lowered for token in ("check", "try", "revisit", "improve", "next step")):
            response = f"{response} Try revisiting your calculation step by step."

    metadata["preprocess_profile"] = "socrateach_single_feedback_v2"
    return UnifiedSample(
        instruction=transformed_instruction,
        response=response,
        style_label=sample.style_label,
        source=sample.source,
        metadata=metadata,
    )


def apply_dataset_preprocessing(sample: UnifiedSample) -> UnifiedSample | None:
    dataset_name = str((sample.metadata or {}).get("dataset", "")).strip().lower()
    if dataset_name == "gsm8k":
        return _transform_gsm8k(sample)
    if dataset_name == "socrateach_multi":
        return _transform_socrateach_multi(sample)
    if dataset_name == "eedi":
        return _transform_eedi(sample)
    if dataset_name == "socrateach_single":
        return _transform_socrateach_single(sample)
    return sample
