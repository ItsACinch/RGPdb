"""Common interface every contender implements."""
from __future__ import annotations
from typing import Protocol


class Ranker(Protocol):
    name: str

    def rank(
        self, seeds: list[int], query_relation: str | None, k: int
    ) -> list[int]:
        """Return up to k node ids, best first."""
        ...
