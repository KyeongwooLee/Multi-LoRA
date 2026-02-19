from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field

from src.config import STYLE_LABELS, normalize_style_label


@dataclass
class UnifiedSample:
    instruction: str
    response: str
    style_label: str
    source: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_whitespace(text: str) -> str:
    text = (text or "").replace("\r", " ").replace("\n", " ").strip()
    return re.sub(r"\s+", " ", text)


def sanitize_sample(sample: UnifiedSample) -> UnifiedSample:
    style = normalize_style_label(sample.style_label)
    return UnifiedSample(
        instruction=normalize_whitespace(sample.instruction),
        response=normalize_whitespace(sample.response),
        style_label=style,
        source=sample.source,
        metadata=sample.metadata,
    )


def is_quality_sample(sample: UnifiedSample, min_chars: int = 20) -> bool:
    if sample.style_label not in STYLE_LABELS:
        return False
    if len(sample.instruction) < min_chars:
        return False
    if len(sample.response) < min_chars:
        return False
    if sample.instruction == sample.response:
        return False
    return True


def sample_signature(sample: UnifiedSample) -> str:
    key = f"{sample.instruction.lower()}|||{sample.response.lower()}|||{sample.style_label}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def deduplicate(samples: list[UnifiedSample]) -> list[UnifiedSample]:
    seen: set[str] = set()
    deduped: list[UnifiedSample] = []
    for sample in samples:
        signature = sample_signature(sample)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(sample)
    return deduped


def dumps_jsonl(items: list[dict]) -> str:
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in items)
