#!/usr/bin/env python3
"""Shared timing and robust-statistics helpers for one-electron benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen = True)
class TimedRunner:
    """One benchmark target with untimed per-call preparation."""

    prepare: Callable[[], None]
    run: Callable[[], None]


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = ":".join(str(part) for part in (base_seed, *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot take percentile of an empty sample")
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_median_ci(
    samples: list[float],
    *,
    confidence: float = 0.95,
    resamples: int = 4000,
    seed: int = 0,
) -> tuple[float, float]:
    if len(samples) == 1:
        return samples[0], samples[0]
    rng = random.Random(seed)
    n = len(samples)
    medians = [
        statistics.median(rng.choices(samples, k=n))
        for _ in range(resamples)
    ]
    tail = 0.5 * (1.0 - confidence)
    return percentile(medians, tail), percentile(medians, 1.0 - tail)


def sample_stats(samples: list[float], *, seed: int = 0) -> dict[str, float]:
    if not samples:
        raise ValueError("Empty timing sample")
    ci_low, ci_high = bootstrap_median_ci(samples, seed = seed)
    return {
        "min_s": min(samples),
        "median_s": statistics.median(samples),
        "mean_s": statistics.fmean(samples),
        "std_s": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "q25_s": percentile(samples, 0.25),
        "q75_s": percentile(samples, 0.75),
        "median_ci_low_s": ci_low,
        "median_ci_high_s": ci_high,
    }


def bootstrap_ratio_ci(
    numerator: list[float],
    denominator: list[float],
    *,
    confidence: float = 0.95,
    resamples: int = 4000,
    seed: int = 0,
) -> tuple[float, float]:
    if not numerator or not denominator:
        raise ValueError("Ratio samples cannot be empty")
    rng = random.Random(seed)
    n_num = len(numerator)
    n_den = len(denominator)
    ratios = []
    for _ in range(resamples):
        num = statistics.median(rng.choices(numerator, k = n_num))
        den = statistics.median(rng.choices(denominator, k = n_den))
        ratios.append(num / den)
    tail = 0.5 * (1.0 - confidence)
    return percentile(ratios, tail), percentile(ratios, 1.0 - tail)


def time_interleaved(
    runners: dict[str, TimedRunner],
    *,
    repeats: int,
    warmups: int,
    seed: int,
) -> tuple[dict[str, list[float]], list[str]]:
    """Time all targets once per round with a shuffled order in every round."""

    if repeats < 1 or warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative")
    rng = random.Random(seed)
    labels = list(runners)
    order_log: list[str] = []

    for _ in range(warmups):
        rng.shuffle(labels)
        for label in labels:
            runner = runners[label]
            runner.prepare()
            runner.run()

    samples = {label: [] for label in runners}
    for _ in range(repeats):
        rng.shuffle(labels)
        order_log.append(">".join(labels))
        for label in labels:
            runner = runners[label]
            runner.prepare()
            start = time.perf_counter()
            runner.run()
            samples[label].append(time.perf_counter() - start)
    return samples, order_log


def encode_samples(samples: list[float]) -> str:
    return json.dumps(samples, separators=(",", ":"))


def decode_samples(text: str) -> list[float]:
    return [float(value) for value in json.loads(text)]
