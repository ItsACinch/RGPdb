"""MetaQA knowledge-base and question loaders."""
from __future__ import annotations
import os
import re

from .dataset import TypedGraph, Question

_TOPIC_RE = re.compile(r"\[(.+?)\]")

# MetaQA entity-type pair -> base relation name. The inverse-augmented loader
# also exposes "<rel>_inv" for the reverse direction, so a qtype segment can
# resolve in either direction.
_BASE_TYPE_RELATIONS = {
    ("movie", "director"): "directed_by",
    ("movie", "writer"): "written_by",
    ("movie", "actor"): "starred_actors",
    ("movie", "year"): "release_year",
    ("movie", "language"): "in_language",
    ("movie", "tags"): "has_tags",
    ("movie", "genre"): "has_genre",
}


def _type_pair_to_relation(a: str, b: str) -> str | None:
    # normalize singular/plural token variants (e.g. "tag" vs "tags")
    norm = {"tag": "tags"}
    a, b = norm.get(a, a), norm.get(b, b)
    if (a, b) in _BASE_TYPE_RELATIONS:
        return _BASE_TYPE_RELATIONS[(a, b)]
    if (b, a) in _BASE_TYPE_RELATIONS:
        return _BASE_TYPE_RELATIONS[(b, a)] + "_inv"
    return None


def query_relation_from_qtype(qtype: str) -> str | None:
    """First-hop relation implied by a MetaQA qtype.

    qtypes look like ``actor_to_movie`` (1-hop) or
    ``movie_to_actor_to_movie_to_director`` (multi-hop). We take the first
    ``<from>_to_<to>`` pair and map it to a relation (using the inverse when the
    pair is reversed). Returns None if it can't be resolved.
    """
    parts = qtype.strip().split("_to_")
    if len(parts) < 2:
        return None
    return _type_pair_to_relation(parts[0], parts[1])


def qtype_to_relation_sequence(qtype: str) -> list[str]:
    """Full gold relation sequence implied by a qtype (one relation per hop).

    ``movie_to_actor_to_movie_to_director`` ->
    ``["starred_actors", "starred_actors_inv", "directed_by"]``.
    Returns [] if any segment can't be resolved.
    """
    parts = qtype.strip().split("_to_")
    seq: list[str] = []
    for i in range(len(parts) - 1):
        r = _type_pair_to_relation(parts[i], parts[i + 1])
        if r is None:
            return []
        seq.append(r)
    return seq


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
                   limit: int | None = None,
                   qtype_path: str | None = None) -> list[Question]:
    # Optional per-question types (line-aligned with qa_path). When present,
    # they give each question a query relation (the first-hop relation), so
    # relation-aware contenders get a real signal instead of None.
    qtypes: list[str] = []
    if qtype_path and os.path.exists(qtype_path):
        with open(qtype_path, encoding="utf-8") as f:
            qtypes = [ln.strip() for ln in f]

    out: list[Question] = []
    with open(qa_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            topic, answers = parse_qa_line(line)
            if topic not in graph.name_to_id:
                continue
            answer_ids = [graph.name_to_id[a] for a in answers
                          if a in graph.name_to_id]
            if not answer_ids:
                continue
            relation = None
            if i < len(qtypes):
                rel = query_relation_from_qtype(qtypes[i])
                # only keep it if that relation actually exists in the graph
                if rel is not None and rel in graph.relation_to_id:
                    relation = rel
            out.append(Question(text=line.split("\t")[0],
                                topic_id=graph.name_to_id[topic],
                                answer_ids=answer_ids, relation=relation, hop=hop))
            if limit is not None and len(out) >= limit:
                break
    return out
