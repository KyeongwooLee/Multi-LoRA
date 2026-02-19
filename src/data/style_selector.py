from __future__ import annotations

import re
from collections import Counter


STYLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "direct": (
        "step",
        "first",
        "second",
        "summary",
        "concept",
        "therefore",
        "formula",
        "explain",
    ),
    "socratic": (
        "what do you think",
        "why",
        "how could",
        "can you",
        "question",
        "hint",
        "suppose",
    ),
    "scaffolding": (
        "let's try",
        "small hint",
        "next step",
        "partially",
        "almost there",
        "guide",
    ),
    "feedback": (
        "strength",
        "improve",
        "revision",
        "feedback",
        "rubric",
        "clarity",
        "suggestion",
    ),
    "motivational": (
        "great job",
        "you can do it",
        "confidence",
        "keep going",
        "proud",
        "encourage",
        "motivate",
    ),
}


def _normalize(text: str) -> str:
    text = (text or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def score_style(style: str, text: str) -> int:
    normalized = _normalize(text)
    score = 0
    for keyword in STYLE_KEYWORDS.get(style, ()):
        if keyword in normalized:
            score += 1
    return score


def choose_style(text: str, candidates: list[str], default: str) -> str:
    if not candidates:
        return default
    counts = Counter({style: score_style(style, text) for style in candidates})
    winner, winner_score = counts.most_common(1)[0]
    if winner_score <= 0:
        return default
    return winner


def style_phrase_score(style: str, text: str) -> float:
    tokens = re.findall(r"[a-zA-Z0-9']+", _normalize(text))
    if not tokens:
        return 0.0
    keyword_hits = 0
    for keyword in STYLE_KEYWORDS.get(style, ()):
        if keyword in _normalize(text):
            keyword_hits += 1
    return keyword_hits / max(1.0, len(STYLE_KEYWORDS.get(style, ())))
