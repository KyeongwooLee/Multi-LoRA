from __future__ import annotations

import hashlib
from collections import defaultdict


def _stable_fraction(key: str) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    as_int = int(digest[:12], 16)
    return (as_int % 10_000_000) / 10_000_000.0


def split_by_ratio(samples: list[dict], train_ratio: float = 0.9, eval_ratio: float = 0.05):
    train: list[dict] = []
    eval_samples: list[dict] = []
    test: list[dict] = []

    for sample in samples:
        key = f"{sample['instruction']}|||{sample['response']}|||{sample['style_label']}"
        frac = _stable_fraction(key)
        if frac < train_ratio:
            train.append(sample)
        elif frac < train_ratio + eval_ratio:
            eval_samples.append(sample)
        else:
            test.append(sample)
    return train, eval_samples, test


def split_by_style(samples: list[dict], train_ratio: float = 0.9, eval_ratio: float = 0.05):
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[sample["style_label"]].append(sample)

    all_train: list[dict] = []
    all_eval: list[dict] = []
    all_test: list[dict] = []
    for style_samples in grouped.values():
        train, eval_samples, test = split_by_ratio(
            style_samples,
            train_ratio=train_ratio,
            eval_ratio=eval_ratio,
        )
        all_train.extend(train)
        all_eval.extend(eval_samples)
        all_test.extend(test)

    return all_train, all_eval, all_test
