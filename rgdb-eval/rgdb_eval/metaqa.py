"""MetaQA knowledge-base and question loaders."""
from __future__ import annotations
import re

from .dataset import TypedGraph, Question

_TOPIC_RE = re.compile(r"\[(.+?)\]")


def parse_kb_line(line: str) -> tuple[str, str, str]:
    head, rel, tail = line.rstrip("\n").split("|")
    return head, rel, tail


def parse_qa_line(line: str) -> tuple[str, list[str]]:
    q, ans = line.rstrip("\n").split("\t")
    m = _TOPIC_RE.search(q)
    topic = m.group(1) if m else ""
    answers = ans.split("|") if ans else []
    return topic, answers


def load_kb_from_lines(lines: list[str], add_inverse: bool = True) -> TypedGraph:
    names: dict[str, int] = {}
    rels: dict[str, int] = {}
    triples: list[tuple[str, str, str]] = []

    def nid(name: str) -> int:
        if name not in names:
            names[name] = len(names)
        return names[name]

    def rid(rel: str) -> int:
        if rel not in rels:
            rels[rel] = len(rels)
        return rels[rel]

    edges: list[tuple[int, int, int]] = []
    for line in lines:
        if not line.strip():
            continue
        h, r, t = parse_kb_line(line)
        triples.append((h, r, t))
        hi, ti = nid(h), nid(t)
        edges.append((hi, ti, rid(r)))
        if add_inverse:
            # Add the inverse edge so the graph is navigable both ways (KGQA
            # standard: MetaQA questions traverse relations in either direction).
            # Kept as a DISTINCT relation type ("<r>_inv") so refraction can still
            # tell a forward hop from a reverse one.
            edges.append((ti, hi, rid(r + "_inv")))

    entity_names = [""] * len(names)
    for name, i in names.items():
        entity_names[i] = name
    relations = [""] * len(rels)
    for rel, i in rels.items():
        relations[i] = rel
    return TypedGraph(num_nodes=len(names), entity_names=entity_names,
                      relations=relations, edges=edges)


def load_kb(kb_path: str, add_inverse: bool = True) -> TypedGraph:
    with open(kb_path, encoding="utf-8") as f:
        return load_kb_from_lines(f.readlines(), add_inverse=add_inverse)


def load_questions(qa_path: str, hop: int, graph: TypedGraph,
                   limit: int | None = None) -> list[Question]:
    out: list[Question] = []
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            topic, answers = parse_qa_line(line)
            if topic not in graph.name_to_id:
                continue
            answer_ids = [graph.name_to_id[a] for a in answers
                          if a in graph.name_to_id]
            if not answer_ids:
                continue
            out.append(Question(text=line.split("\t")[0],
                                topic_id=graph.name_to_id[topic],
                                answer_ids=answer_ids, relation=None, hop=hop))
            if limit is not None and len(out) >= limit:
                break
    return out
