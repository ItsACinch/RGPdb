"""Ranking-quality metrics, all defined for a single query."""
from __future__ import annotations

K_VALUES: tuple[int, ...] = (1, 5, 10, 20)


def hits_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    """1.0 if any gold id appears in the top-k, else 0.0."""
    if not gold:
        return 0.0
    return 1.0 if any(x in gold for x in ranked[:k]) else 0.0


def recall_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    """Fraction of gold ids present in the top-k."""
    if not gold:
        return 0.0
    hit = sum(1 for x in ranked[:k] if x in gold)
    return hit / len(gold)


def mrr(ranked: list[int], gold: set[int]) -> float:
    """Reciprocal rank of the first gold id (0.0 if none present)."""
    if not gold:
        return 0.0
    for i, x in enumerate(ranked, start=1):
        if x in gold:
            return 1.0 / i
    return 0.0
