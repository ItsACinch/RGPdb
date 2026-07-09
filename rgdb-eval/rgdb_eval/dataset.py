"""Dataset primitives shared by all contenders and loaders."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TypedGraph:
    num_nodes: int
    entity_names: list[str]
    relations: list[str]
    edges: list[tuple[int, int, int]]  # (src, dst, relation_id)
    name_to_id: dict[str, int] = field(default_factory=dict)
    relation_to_id: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name_to_id:
            self.name_to_id = {n: i for i, n in enumerate(self.entity_names)}
        if not self.relation_to_id:
            self.relation_to_id = {r: i for i, r in enumerate(self.relations)}

    def out_neighbors(self, node: int) -> list[tuple[int, int]]:
        """Return [(dst, relation_id), ...] for edges leaving `node`."""
        return [(d, r) for (s, d, r) in self.edges if s == node]


@dataclass
class Question:
    text: str
    topic_id: int
    answer_ids: list[int]
    relation: str | None
    hop: int
