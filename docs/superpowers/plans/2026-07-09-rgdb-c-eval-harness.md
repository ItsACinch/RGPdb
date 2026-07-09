# RGDB Sub-project C — Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A repeatable Python benchmark that scores any ranking function on multi-hop retrieval (MetaQA + a synthetic probe suite), so the core rewrite (sub-project B) lands with a measured before/after.

**Architecture:** A new `rgdb-eval/` Python package. A common `Ranker` protocol (`rank(seeds, query_relation, k) -> ranked node ids`) is implemented by several contenders (vector-only, vector+2-hop, untyped PPR, current RGDB via bindings). A dataset layer produces a typed graph + question sets from MetaQA and from a synthetic generator. A metrics module computes Hits@k / recall@k / MRR by hop count. A runner CLI emits a markdown table under `results/`.

**Tech Stack:** Python ≥3.9, numpy, scipy (sparse PPR), pandas (tables), sentence-transformers (entity/question embeddings), maturin-built `rgdb-core` bindings.

## Global Constraints

- Built **before** sub-project B, so the "current RGDB" contender exercises the *existing* bindings (`build_graph(num_nodes, adjacency=[(dst, attenuation, angle_bin)], ...)`, `propagate_light(graph, source, initial_bin, params)`). Do not assume the new API.
- The `Ranker` protocol signature is fixed for all contenders: `rank(self, seeds: list[int], query_relation: str | None, k: int) -> list[int]`. Sub-project B adds new-RGDB contenders behind this same protocol.
- MetaQA seeding is **exact**: the topic entity named in each question is the single seed. No embedding lookup is used to find the source node (isolates graph ranking from seed retrieval).
- Metrics: Hits@k and recall@k for k ∈ {1, 5, 10, 20}, plus MRR, always broken down by hop count (1/2/3).
- Results are written as markdown to `rgdb-eval/results/`.
- MetaQA relation vocabulary (9): `directed_by`, `written_by`, `starred_actors`, `release_year`, `in_language`, `has_tags`, `has_genre`, `has_imdb_rating`, `has_imdb_votes`.
- Commit after each task.

---

### Task 1: Verify maturin build of `rgdb-python` and add a CI smoke check

**Files:**
- Create: `.github/workflows/python.yml`
- Create: `rgdb-eval/scripts/smoke_bindings.py`

**Interfaces:**
- Consumes: existing `rgdb-python` crate (module `rgdb_embeddings._rgdb_core`, per its `pyproject.toml` `module-name`).
- Produces: a confirmed-importable native module and a CI job that builds it.

**Background:** `rgdb-python`'s `pyproject.toml` sets `module-name = "rgdb_embeddings._rgdb_core"`. The eval harness depends on this building. This task de-risks the one external dependency (maturin-on-Windows) before writing harness code.

- [ ] **Step 1: Write the smoke script**

Create `rgdb-eval/scripts/smoke_bindings.py`:

```python
"""Confirm the native rgdb bindings import and a trivial propagation runs."""
from rgdb_embeddings import _rgdb_core as core


def main() -> None:
    # 0 -> 1 -> 2 chain; attenuation 0.1, angle_bin 0
    adjacency = [[(1, 0.1, 0)], [(2, 0.1, 0)], []]
    g = core.build_graph(3, adjacency)
    assert g.num_nodes == 3, g.num_nodes
    assert g.num_edges == 2, g.num_edges
    intensities = core.propagate_light(g, 0)
    assert len(intensities) == 3, len(intensities)
    assert intensities[0] > 0.0, "source should have positive intensity"
    print("bindings smoke ok:", list(intensities))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Build and install the bindings into the current environment**

Run:
```bash
pip install maturin
maturin develop -m rgdb-python/Cargo.toml
```
Expected: `🛠 Installed rgdb-core` (or `Built wheel` + install). If this fails on Windows, resolve the toolchain here — this is the gate the whole harness depends on.

- [ ] **Step 3: Run the smoke script**

Run: `python rgdb-eval/scripts/smoke_bindings.py`
Expected: `bindings smoke ok: [...]` with a positive first element.

- [ ] **Step 4: Add the Python CI workflow**

Create `.github/workflows/python.yml`:

```yaml
name: Python bindings

on:
  push:
    branches: [main]
  pull_request:

jobs:
  bindings:
    name: maturin build + import
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install maturin
      - run: maturin develop -m rgdb-python/Cargo.toml
      - run: python rgdb-eval/scripts/smoke_bindings.py
```

- [ ] **Step 5: Validate YAML and commit**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/python.yml')); print('yaml ok')"`
Expected: `yaml ok`

```bash
git add rgdb-eval/scripts/smoke_bindings.py .github/workflows/python.yml
git commit -m "eval: verify rgdb bindings build + add python CI smoke check"
```

---

### Task 2: Scaffold the `rgdb-eval` package

**Files:**
- Create: `rgdb-eval/pyproject.toml`
- Create: `rgdb-eval/rgdb_eval/__init__.py`
- Create: `rgdb-eval/rgdb_eval/dataset.py`
- Create: `rgdb-eval/tests/test_smoke.py`

**Interfaces:**
- Produces: `TypedGraph` and `Question` dataclasses consumed by every later task.

```python
# TypedGraph: a KG the harness ranks over.
#   num_nodes: int
#   entity_names: list[str]                # index == node id
#   name_to_id: dict[str, int]
#   relations: list[str]                   # index == relation id
#   relation_to_id: dict[str, int]
#   edges: list[tuple[int, int, int]]      # (src, dst, relation_id), directed
# Question:
#   text: str
#   topic_id: int                          # exact seed
#   answer_ids: list[int]                  # gold
#   relation: str | None                   # gold relation for 1-hop; None otherwise
#   hop: int                               # 1 | 2 | 3
```

- [ ] **Step 1: Write the package metadata**

Create `rgdb-eval/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "rgdb-eval"
version = "0.1.0"
description = "Retrieval-quality evaluation harness for RGDB"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "pandas>=2.0",
    "sentence-transformers>=2.2",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["rgdb_eval*"]
```

- [ ] **Step 2: Write the dataclasses**

Create `rgdb-eval/rgdb_eval/dataset.py`:

```python
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
```

- [ ] **Step 3: Write the package init**

Create `rgdb-eval/rgdb_eval/__init__.py`:

```python
from .dataset import TypedGraph, Question

__all__ = ["TypedGraph", "Question"]
```

- [ ] **Step 4: Write the smoke test**

Create `rgdb-eval/tests/test_smoke.py`:

```python
from rgdb_eval import TypedGraph, Question


def test_typed_graph_indexes():
    g = TypedGraph(
        num_nodes=3,
        entity_names=["a", "b", "c"],
        relations=["r0", "r1"],
        edges=[(0, 1, 0), (0, 2, 1)],
    )
    assert g.name_to_id["b"] == 1
    assert g.relation_to_id["r1"] == 1
    assert sorted(g.out_neighbors(0)) == [(1, 0), (2, 1)]


def test_question_fields():
    q = Question(text="what?", topic_id=0, answer_ids=[2], relation="r1", hop=1)
    assert q.hop == 1 and q.answer_ids == [2]
```

- [ ] **Step 5: Install and run tests**

Run:
```bash
pip install -e rgdb-eval
pytest rgdb-eval/tests/test_smoke.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add rgdb-eval/pyproject.toml rgdb-eval/rgdb_eval/__init__.py rgdb-eval/rgdb_eval/dataset.py rgdb-eval/tests/test_smoke.py
git commit -m "eval: scaffold rgdb-eval package with TypedGraph/Question primitives"
```

---

### Task 3: Metrics module

**Files:**
- Create: `rgdb-eval/rgdb_eval/metrics.py`
- Create: `rgdb-eval/tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `hits_at_k(ranked: list[int], gold: set[int], k: int) -> float`
  - `recall_at_k(ranked: list[int], gold: set[int], k: int) -> float`
  - `mrr(ranked: list[int], gold: set[int]) -> float`
  - `K_VALUES: tuple[int, ...] = (1, 5, 10, 20)`

- [ ] **Step 1: Write the failing test**

Create `rgdb-eval/tests/test_metrics.py`:

```python
from rgdb_eval.metrics import hits_at_k, recall_at_k, mrr, K_VALUES


def test_k_values():
    assert K_VALUES == (1, 5, 10, 20)


def test_hits_at_k():
    ranked = [9, 3, 7, 1]
    gold = {7}
    assert hits_at_k(ranked, gold, 1) == 0.0   # 7 not in top-1
    assert hits_at_k(ranked, gold, 5) == 1.0   # 7 within top-5


def test_recall_at_k():
    ranked = [9, 3, 7, 1]
    gold = {7, 1, 42}
    # top-4 contains 7 and 1 of the 3 gold -> 2/3
    assert abs(recall_at_k(ranked, gold, 4) - (2 / 3)) < 1e-9


def test_mrr():
    ranked = [9, 3, 7, 1]
    gold = {7}
    assert abs(mrr(ranked, gold) - (1 / 3)) < 1e-9  # first gold at rank 3
    assert mrr(ranked, set()) == 0.0
    assert mrr([], {1}) == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rgdb-eval/tests/test_metrics.py -v`
Expected: FAIL (`ModuleNotFoundError: rgdb_eval.metrics`).

- [ ] **Step 3: Implement the metrics**

Create `rgdb-eval/rgdb_eval/metrics.py`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest rgdb-eval/tests/test_metrics.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add rgdb-eval/rgdb_eval/metrics.py rgdb-eval/tests/test_metrics.py
git commit -m "eval: add Hits@k / recall@k / MRR metrics"
```

---

### Task 4: Ranker protocol + untyped PPR baseline

**Files:**
- Create: `rgdb-eval/rgdb_eval/rankers/__init__.py`
- Create: `rgdb-eval/rgdb_eval/rankers/base.py`
- Create: `rgdb-eval/rgdb_eval/rankers/ppr.py`
- Create: `rgdb-eval/tests/test_ppr.py`

**Interfaces:**
- Produces:
  - `Ranker` protocol: `rank(self, seeds: list[int], query_relation: str | None, k: int) -> list[int]`
  - `PPRRanker(graph: TypedGraph, damping: float = 0.85, iters: int = 30)` — untyped personalized PageRank via scipy sparse.

- [ ] **Step 1: Write the failing test**

Create `rgdb-eval/tests/test_ppr.py`:

```python
from rgdb_eval import TypedGraph
from rgdb_eval.rankers.ppr import PPRRanker


def line_graph() -> TypedGraph:
    # 0 -> 1 -> 2 -> 3, single relation
    return TypedGraph(
        num_nodes=4,
        entity_names=["n0", "n1", "n2", "n3"],
        relations=["r"],
        edges=[(0, 1, 0), (1, 2, 0), (2, 3, 0)],
    )


def test_ppr_ranks_closer_nodes_higher():
    r = PPRRanker(line_graph())
    ranked = r.rank(seeds=[0], query_relation=None, k=4)
    # From seed 0, node 1 must outrank node 3 (closer on the chain).
    assert ranked.index(1) < ranked.index(3)


def test_ppr_returns_at_most_k():
    r = PPRRanker(line_graph())
    assert len(r.rank(seeds=[0], query_relation=None, k=2)) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rgdb-eval/tests/test_ppr.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write the protocol**

Create `rgdb-eval/rgdb_eval/rankers/base.py`:

```python
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
```

Create `rgdb-eval/rgdb_eval/rankers/__init__.py`:

```python
from .base import Ranker

__all__ = ["Ranker"]
```

- [ ] **Step 4: Implement the PPR ranker**

Create `rgdb-eval/rgdb_eval/rankers/ppr.py`:

```python
"""Untyped personalized PageRank baseline (scipy sparse)."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp

from ..dataset import TypedGraph


class PPRRanker:
    name = "untyped-ppr"

    def __init__(self, graph: TypedGraph, damping: float = 0.85, iters: int = 30):
        self.graph = graph
        self.damping = damping
        self.iters = iters
        n = graph.num_nodes
        if graph.edges:
            rows = np.fromiter((s for (s, _, _) in graph.edges), dtype=np.int64)
            cols = np.fromiter((d for (_, d, _) in graph.edges), dtype=np.int64)
            data = np.ones(len(graph.edges), dtype=np.float64)
            adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        else:
            adj = sp.csr_matrix((n, n), dtype=np.float64)
        # Row-normalize to a transition matrix (dangling rows stay zero).
        out = np.asarray(adj.sum(axis=1)).ravel()
        inv = np.divide(1.0, out, out=np.zeros_like(out), where=out > 0)
        self.trans = sp.diags(inv) @ adj  # row-stochastic where out>0

    def rank(self, seeds, query_relation, k):
        n = self.graph.num_nodes
        if not seeds:
            return []
        restart = np.zeros(n, dtype=np.float64)
        restart[seeds] = 1.0 / len(seeds)
        scores = restart.copy()
        tt = self.trans.T.tocsr()
        for _ in range(self.iters):
            scores = self.damping * (tt @ scores) + (1 - self.damping) * restart
        order = np.argsort(-scores)
        return [int(i) for i in order[:k]]
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest rgdb-eval/tests/test_ppr.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add rgdb-eval/rgdb_eval/rankers rgdb-eval/tests/test_ppr.py
git commit -m "eval: add Ranker protocol + untyped PPR baseline"
```

---

### Task 5: Vector-only and vector+2-hop baselines

**Files:**
- Create: `rgdb-eval/rgdb_eval/embeddings.py`
- Create: `rgdb-eval/rgdb_eval/rankers/vector.py`
- Create: `rgdb-eval/tests/test_vector.py`

**Interfaces:**
- Consumes: `Ranker` protocol, `TypedGraph`.
- Produces:
  - `embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray` (L2-normalized rows).
  - `VectorRanker(graph, node_vecs: np.ndarray, query_vec_fn)` — ranks all nodes by cosine to the query vector.
  - `VectorTwoHopRanker(graph, node_vecs, query_vec_fn)` — restricts candidates to the ≤2-hop out-neighborhood of the seeds, then ranks by cosine.
  - Both take `query_vec_fn: Callable[[str | None], np.ndarray]` returning a unit query vector; the runner supplies question-text embeddings.

- [ ] **Step 1: Write the failing test**

Create `rgdb-eval/tests/test_vector.py`:

```python
import numpy as np
from rgdb_eval import TypedGraph
from rgdb_eval.rankers.vector import VectorRanker, VectorTwoHopRanker


def graph4():
    return TypedGraph(
        num_nodes=4,
        entity_names=["n0", "n1", "n2", "n3"],
        relations=["r"],
        edges=[(0, 1, 0), (1, 2, 0)],  # 3 is unreachable from 0
    )


def fixed_vecs():
    # node 2 is identical to the query; node 3 also close but unreachable
    return np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.9, 0.1]], dtype=np.float64
    )


def test_vector_ranks_by_cosine():
    q = np.array([1.0, 0.0])
    r = VectorRanker(graph4(), fixed_vecs(), lambda _rel: q)
    ranked = r.rank(seeds=[0], query_relation=None, k=4)
    assert ranked[0] in (0, 2)  # cosine-identical to query


def test_two_hop_excludes_unreachable():
    q = np.array([0.9, 0.1])
    r = VectorTwoHopRanker(graph4(), fixed_vecs(), lambda _rel: q)
    ranked = r.rank(seeds=[0], query_relation=None, k=4)
    # node 3 is closest to q but not within 2 hops of seed 0 -> excluded
    assert 3 not in ranked
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rgdb-eval/tests/test_vector.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement embeddings helper**

Create `rgdb-eval/rgdb_eval/embeddings.py`:

```python
"""Sentence-transformer embeddings, L2-normalized."""
from __future__ import annotations
import numpy as np

_MODEL_CACHE: dict[str, object] = {}


def _get_model(name: str):
    if name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = _get_model(model_name)
    vecs = np.asarray(model.encode(texts, show_progress_bar=False), dtype=np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms
```

- [ ] **Step 4: Implement the vector rankers**

Create `rgdb-eval/rgdb_eval/rankers/vector.py`:

```python
"""Vector-similarity contenders."""
from __future__ import annotations
from typing import Callable
import numpy as np

from ..dataset import TypedGraph


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class VectorRanker:
    name = "vector-only"

    def __init__(self, graph: TypedGraph, node_vecs: np.ndarray,
                 query_vec_fn: Callable[[str | None], np.ndarray]):
        self.graph = graph
        self.node_vecs = node_vecs
        self.query_vec_fn = query_vec_fn

    def _scores(self, query_relation):
        q = _unit(np.asarray(self.query_vec_fn(query_relation), dtype=np.float64))
        return self.node_vecs @ q

    def rank(self, seeds, query_relation, k):
        scores = self._scores(query_relation)
        order = np.argsort(-scores)
        return [int(i) for i in order[:k]]


class VectorTwoHopRanker(VectorRanker):
    name = "vector+2hop"

    def _candidates(self, seeds: list[int]) -> list[int]:
        frontier = set(seeds)
        seen = set(seeds)
        for _ in range(2):
            nxt = set()
            for u in frontier:
                for (d, _r) in self.graph.out_neighbors(u):
                    if d not in seen:
                        seen.add(d)
                        nxt.add(d)
            frontier = nxt
        return list(seen)

    def rank(self, seeds, query_relation, k):
        scores = self._scores(query_relation)
        cands = self._candidates(seeds)
        cands.sort(key=lambda i: -scores[i])
        return cands[:k]
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest rgdb-eval/tests/test_vector.py -v`
Expected: 2 passed. (First run downloads the model only if a test calls `embed_texts`; these tests inject fixed vectors, so no download.)

- [ ] **Step 6: Commit**

```bash
git add rgdb-eval/rgdb_eval/embeddings.py rgdb-eval/rgdb_eval/rankers/vector.py rgdb-eval/tests/test_vector.py
git commit -m "eval: add vector-only and vector+2hop baselines"
```

---

### Task 6: Current-RGDB contender via bindings

**Files:**
- Create: `rgdb-eval/rgdb_eval/rankers/rgdb_current.py`
- Create: `rgdb-eval/tests/test_rgdb_current.py`

**Interfaces:**
- Consumes: `rgdb_embeddings._rgdb_core` (existing bindings), `TypedGraph`.
- Produces: `CurrentRgdbRanker(graph: TypedGraph, relation_bins: dict[str, int] | None = None)` — builds the RGDB graph once (relations → angle bins by relation id mod 16), propagates from the seed with `initial_bin = bin(query_relation)`, ranks by intensity.

**Background:** The current bindings model edges as `(dst, attenuation, angle_bin)` and propagate with `propagate_light(graph, source, initial_bin, params)`. This contender is the pre-rewrite baseline; sub-project B replaces it with the typed-PPR contender behind the same protocol.

- [ ] **Step 1: Write the failing test**

Create `rgdb-eval/tests/test_rgdb_current.py`:

```python
import pytest
from rgdb_eval import TypedGraph

core = pytest.importorskip("rgdb_embeddings._rgdb_core")
from rgdb_eval.rankers.rgdb_current import CurrentRgdbRanker


def line_graph():
    return TypedGraph(
        num_nodes=4,
        entity_names=["n0", "n1", "n2", "n3"],
        relations=["r"],
        edges=[(0, 1, 0), (1, 2, 0), (2, 3, 0)],
    )


def test_current_rgdb_ranks_reachable_nodes():
    r = CurrentRgdbRanker(line_graph())
    ranked = r.rank(seeds=[0], query_relation="r", k=4)
    # Node 1 (1 hop) should outrank node 3 (3 hops) by intensity.
    assert ranked.index(1) < ranked.index(3)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rgdb-eval/tests/test_rgdb_current.py -v`
Expected: FAIL (`ModuleNotFoundError: rgdb_eval.rankers.rgdb_current`). (If the bindings are not installed, the test is skipped — install via `maturin develop -m rgdb-python/Cargo.toml`.)

- [ ] **Step 3: Implement the contender**

Create `rgdb-eval/rgdb_eval/rankers/rgdb_current.py`:

```python
"""Pre-rewrite RGDB contender using the current angle-bin bindings."""
from __future__ import annotations
import numpy as np
from rgdb_embeddings import _rgdb_core as core

from ..dataset import TypedGraph

N_BINS = 16


class CurrentRgdbRanker:
    name = "rgdb-current"

    def __init__(self, graph: TypedGraph, relation_bins: dict[str, int] | None = None):
        self.graph = graph
        if relation_bins is None:
            relation_bins = {r: (i % N_BINS) for i, r in enumerate(graph.relations)}
        self.relation_bins = relation_bins
        # Build adjacency in binding format: adj[u] = [(dst, attenuation, angle_bin)]
        adj: list[list[tuple[int, float, int]]] = [[] for _ in range(graph.num_nodes)]
        for (s, d, rel_id) in graph.edges:
            bin_ = rel_id % N_BINS
            adj[s].append((d, 0.1, bin_))
        self._g = core.build_graph(graph.num_nodes, adj)

    def rank(self, seeds, query_relation, k):
        if not seeds:
            return []
        bin_ = self.relation_bins.get(query_relation, 0) if query_relation else 0
        acc = np.zeros(self.graph.num_nodes, dtype=np.float64)
        for s in seeds:
            acc += np.asarray(core.propagate_light(self._g, s, bin_), dtype=np.float64)
        order = np.argsort(-acc)
        return [int(i) for i in order[:k]]
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest rgdb-eval/tests/test_rgdb_current.py -v`
Expected: 1 passed (requires bindings installed).

- [ ] **Step 5: Commit**

```bash
git add rgdb-eval/rgdb_eval/rankers/rgdb_current.py rgdb-eval/tests/test_rgdb_current.py
git commit -m "eval: add current-RGDB contender via existing bindings"
```

---

### Task 7: Synthetic probe suite

**Files:**
- Create: `rgdb-eval/rgdb_eval/synthetic.py`
- Create: `rgdb-eval/tests/test_synthetic.py`

**Interfaces:**
- Consumes: `TypedGraph`, `Question`.
- Produces: `make_probe(kind: str, seed: int, n_relations: int = 6, depth: int = 3, noise_nodes: int = 50) -> tuple[TypedGraph, list[Question]]` where `kind ∈ {"coherent", "incoherent"}`.
  - `"coherent"`: the gold path uses a single repeated relation (relation-coherent) → refraction should win.
  - `"incoherent"`: the gold path alternates relations (relation-incoherent) → refraction should not help and should not badly hurt.

**Background:** Deterministic given `seed` (no wall-clock randomness). Each probe plants one topic→answer path of length `depth`, plus distractor edges among noise nodes.

- [ ] **Step 1: Write the failing test**

Create `rgdb-eval/tests/test_synthetic.py`:

```python
from rgdb_eval.synthetic import make_probe


def test_coherent_probe_is_deterministic():
    g1, q1 = make_probe("coherent", seed=7)
    g2, q2 = make_probe("coherent", seed=7)
    assert g1.edges == g2.edges
    assert q1[0].answer_ids == q2[0].answer_ids


def test_coherent_path_uses_one_relation():
    g, qs = make_probe("coherent", seed=1, depth=3)
    q = qs[0]
    assert q.hop == 3
    # walk the planted path from topic; every edge should share one relation id
    rels = set()
    node = q.topic_id
    for _ in range(3):
        outs = g.out_neighbors(node)
        # the planted successor is the lowest-id neighbor
        nxt, rel = min(outs)
        rels.add(rel)
        node = nxt
    assert len(rels) == 1
    assert node in q.answer_ids


def test_incoherent_path_uses_varied_relations():
    g, qs = make_probe("incoherent", seed=2, depth=3)
    q = qs[0]
    rels, node = [], q.topic_id
    for _ in range(3):
        nxt, rel = min(g.out_neighbors(node))
        rels.append(rel)
        node = nxt
    assert len(set(rels)) > 1
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rgdb-eval/tests/test_synthetic.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the generator**

Create `rgdb-eval/rgdb_eval/synthetic.py`:

```python
"""Deterministic synthetic probe graphs for controlled refraction ablations."""
from __future__ import annotations
import random

from .dataset import TypedGraph, Question


def make_probe(kind: str, seed: int, n_relations: int = 6,
               depth: int = 3, noise_nodes: int = 50):
    if kind not in ("coherent", "incoherent"):
        raise ValueError(f"unknown probe kind: {kind}")
    rng = random.Random(seed)

    # Nodes 0..depth are the planted path; the rest are noise.
    n_nodes = depth + 1 + noise_nodes
    relations = [f"r{i}" for i in range(n_relations)]
    edges: list[tuple[int, int, int]] = []

    # Planted path 0 -> 1 -> ... -> depth.
    # Successor of a node on the path is always the next id, so `min(out_neighbors)`
    # in the tests deterministically follows the planted edge.
    if kind == "coherent":
        path_rel = rng.randrange(n_relations)
        for i in range(depth):
            edges.append((i, i + 1, path_rel))
    else:  # incoherent: cycle through distinct relations
        for i in range(depth):
            edges.append((i, i + 1, i % n_relations))

    # Noise edges among the higher-id nodes, none re-entering the planted path.
    noise_start = depth + 1
    for u in range(noise_start, n_nodes):
        for _ in range(2):
            v = rng.randrange(noise_start, n_nodes)
            if v != u:
                edges.append((u, v, rng.randrange(n_relations)))

    names = [f"n{i}" for i in range(n_nodes)]
    g = TypedGraph(num_nodes=n_nodes, entity_names=names,
                   relations=relations, edges=edges)
    q = Question(text=f"{kind} probe", topic_id=0, answer_ids=[depth],
                 relation=relations[edges[0][2]] if kind == "coherent" else None,
                 hop=depth)
    return g, [q]
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest rgdb-eval/tests/test_synthetic.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add rgdb-eval/rgdb_eval/synthetic.py rgdb-eval/tests/test_synthetic.py
git commit -m "eval: add deterministic synthetic probe suite"
```

---

### Task 8: MetaQA loader

**Files:**
- Create: `rgdb-eval/rgdb_eval/metaqa.py`
- Create: `rgdb-eval/tests/test_metaqa.py`
- Create: `rgdb-eval/data/README.md`

**Interfaces:**
- Consumes: `TypedGraph`, `Question`.
- Produces:
  - `load_kb(kb_path: str) -> TypedGraph`
  - `load_questions(qa_path: str, hop: int, graph: TypedGraph, limit: int | None = None) -> list[Question]`
  - Parsers `parse_kb_line` / `parse_qa_line` (pure, unit-tested without files).

**Background:** MetaQA `kb.txt` lines are `head|relation|tail`. QA lines are `question with [topic entity]\tans1|ans2|...`. The topic entity is the bracketed span. We keep questions whose topic and ≥1 answer resolve to KB entities.

- [ ] **Step 1: Write the failing test**

Create `rgdb-eval/tests/test_metaqa.py`:

```python
from rgdb_eval.metaqa import parse_kb_line, parse_qa_line, load_kb_from_lines


def test_parse_kb_line():
    assert parse_kb_line("Blade Runner|directed_by|Ridley Scott") == (
        "Blade Runner", "directed_by", "Ridley Scott"
    )


def test_parse_qa_line():
    line = "what films did [Ridley Scott] direct\tBlade Runner|Alien"
    topic, answers = parse_qa_line(line)
    assert topic == "Ridley Scott"
    assert answers == ["Blade Runner", "Alien"]


def test_load_kb_from_lines_builds_typed_graph():
    lines = [
        "Blade Runner|directed_by|Ridley Scott",
        "Alien|directed_by|Ridley Scott",
    ]
    g = load_kb_from_lines(lines)
    assert g.num_nodes == 3          # 2 movies + 1 director
    assert "directed_by" in g.relation_to_id
    assert len(g.edges) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rgdb-eval/tests/test_metaqa.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the loader**

Create `rgdb-eval/rgdb_eval/metaqa.py`:

```python
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


def load_kb_from_lines(lines: list[str]) -> TypedGraph:
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
        edges.append((nid(h), nid(t), rid(r)))

    entity_names = [""] * len(names)
    for name, i in names.items():
        entity_names[i] = name
    relations = [""] * len(rels)
    for rel, i in rels.items():
        relations[i] = rel
    return TypedGraph(num_nodes=len(names), entity_names=entity_names,
                      relations=relations, edges=edges)


def load_kb(kb_path: str) -> TypedGraph:
    with open(kb_path, encoding="utf-8") as f:
        return load_kb_from_lines(f.readlines())


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
```

- [ ] **Step 4: Document where to obtain the data**

Create `rgdb-eval/data/README.md`:

```markdown
# MetaQA data

The harness expects MetaQA files placed here (not committed):

```
data/MetaQA/kb.txt
data/MetaQA/qa_test_1hop.txt
data/MetaQA/qa_test_2hop.txt
data/MetaQA/qa_test_3hop.txt
```

- `kb.txt` lines: `head|relation|tail`
- `qa_test_Nhop.txt` lines: `question with [topic entity]<TAB>answer1|answer2`

Source: the MetaQA dataset (movie KG, ~43k entities, ~135k triples, 9 relations).
Download from the public MetaQA release and copy the vanilla KB + test QA files
into the paths above. The runner (`rgdb_eval.run`) reads these paths by default.
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest rgdb-eval/tests/test_metaqa.py -v`
Expected: 3 passed. (Parser/loader tests use in-memory lines — no data download needed.)

- [ ] **Step 6: Commit**

```bash
git add rgdb-eval/rgdb_eval/metaqa.py rgdb-eval/tests/test_metaqa.py rgdb-eval/data/README.md
git commit -m "eval: add MetaQA KB + question loaders"
```

---

### Task 9: Runner CLI + markdown report

**Files:**
- Create: `rgdb-eval/rgdb_eval/report.py`
- Create: `rgdb-eval/rgdb_eval/run.py`
- Create: `rgdb-eval/tests/test_report.py`
- Create: `rgdb-eval/results/.gitkeep`

**Interfaces:**
- Consumes: all rankers, metrics, loaders.
- Produces:
  - `evaluate(ranker, questions) -> dict` — mean metrics keyed `hits@k`, `recall@k`, `mrr`, per hop.
  - `to_markdown(rows: list[dict]) -> str` — renders the results table.
  - `python -m rgdb_eval.run --data data/MetaQA --limit 1000 --out results/metaqa.md`

- [ ] **Step 1: Write the failing test**

Create `rgdb-eval/tests/test_report.py`:

```python
from rgdb_eval import TypedGraph, Question
from rgdb_eval.rankers.ppr import PPRRanker
from rgdb_eval.report import evaluate, to_markdown


def test_evaluate_and_markdown():
    g = TypedGraph(
        num_nodes=4, entity_names=["n0", "n1", "n2", "n3"],
        relations=["r"], edges=[(0, 1, 0), (1, 2, 0), (2, 3, 0)],
    )
    qs = [Question(text="q", topic_id=0, answer_ids=[1], relation="r", hop=1)]
    res = evaluate(PPRRanker(g), qs)
    assert res["ranker"] == "untyped-ppr"
    assert 0.0 <= res["hop1"]["mrr"] <= 1.0
    md = to_markdown([res])
    assert "untyped-ppr" in md and "mrr" in md.lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest rgdb-eval/tests/test_report.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement evaluation + report**

Create `rgdb-eval/rgdb_eval/report.py`:

```python
"""Aggregate per-question metrics into per-hop means and a markdown table."""
from __future__ import annotations
from statistics import mean

from .metrics import hits_at_k, recall_at_k, mrr, K_VALUES


def evaluate(ranker, questions) -> dict:
    by_hop: dict[int, list] = {}
    for q in questions:
        by_hop.setdefault(q.hop, []).append(q)

    out: dict = {"ranker": ranker.name}
    for hop, qs in sorted(by_hop.items()):
        gold = [set(q.answer_ids) for q in qs]
        ranked = [ranker.rank([q.topic_id], q.relation, max(K_VALUES)) for q in qs]
        stats = {}
        for k in K_VALUES:
            stats[f"hits@{k}"] = mean(hits_at_k(r, g, k) for r, g in zip(ranked, gold))
            stats[f"recall@{k}"] = mean(recall_at_k(r, g, k) for r, g in zip(ranked, gold))
        stats["mrr"] = mean(mrr(r, g) for r, g in zip(ranked, gold))
        stats["n"] = len(qs)
        out[f"hop{hop}"] = stats
    return out


def to_markdown(rows: list[dict]) -> str:
    hops = sorted({key for row in rows for key in row if key.startswith("hop")})
    cols = [f"hits@{k}" for k in K_VALUES] + [f"recall@{k}" for k in K_VALUES] + ["mrr"]
    lines: list[str] = []
    for hop in hops:
        lines.append(f"\n### {hop}\n")
        header = "| ranker | n | " + " | ".join(cols) + " |"
        sep = "|" + "---|" * (len(cols) + 2)
        lines += [header, sep]
        for row in rows:
            s = row.get(hop)
            if not s:
                continue
            cells = [f"{s[c]:.3f}" for c in cols]
            lines.append(f"| {row['ranker']} | {s['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Implement the runner CLI**

Create `rgdb-eval/rgdb_eval/run.py`:

```python
"""CLI: evaluate all available contenders on MetaQA and write a markdown table."""
from __future__ import annotations
import argparse
import os

from .metaqa import load_kb, load_questions
from .embeddings import embed_texts
from .rankers.ppr import PPRRanker
from .rankers.vector import VectorRanker, VectorTwoHopRanker
from .report import evaluate, to_markdown


def build_rankers(graph, node_vecs, question_vec):
    rankers = [
        VectorRanker(graph, node_vecs, lambda _r: question_vec),
        VectorTwoHopRanker(graph, node_vecs, lambda _r: question_vec),
        PPRRanker(graph),
    ]
    try:
        from .rankers.rgdb_current import CurrentRgdbRanker
        rankers.append(CurrentRgdbRanker(graph))
    except Exception as exc:  # bindings not installed -> skip, but say so
        print(f"[warn] rgdb-current contender skipped: {exc}")
    return rankers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/MetaQA")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", default="results/metaqa.md")
    args = ap.parse_args()

    graph = load_kb(os.path.join(args.data, "kb.txt"))
    node_vecs = embed_texts(graph.entity_names)

    all_questions = []
    for hop in (1, 2, 3):
        path = os.path.join(args.data, f"qa_test_{hop}hop.txt")
        if os.path.exists(path):
            all_questions += load_questions(path, hop, graph, limit=args.limit)

    # One question embedding per question would be ideal; for a per-run table we
    # score each contender question-by-question, so embed lazily per question.
    # Here we pass a closure that re-embeds the current question text.
    rows = []
    for ranker in build_rankers(graph, node_vecs, question_vec=node_vecs[0]):
        # Rebind the query vector per question for vector contenders.
        if isinstance(ranker, VectorRanker):
            def qvec(_rel, _cache={}):
                return _cache.get("v", node_vecs[0])
            ranker.query_vec_fn = qvec
        rows.append(_evaluate_with_question_embeddings(ranker, all_questions))

    md = to_markdown(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# MetaQA retrieval results\n" + md)
    print(f"wrote {args.out}")


def _evaluate_with_question_embeddings(ranker, questions):
    # For vector contenders, set the per-question embedding before ranking.
    from .rankers.vector import VectorRanker as _VR
    if isinstance(ranker, _VR):
        texts = [q.text for q in questions]
        qvecs = embed_texts(texts)
        idx = {"i": 0}
        def fn(_rel):
            return qvecs[idx["i"]]
        ranker.query_vec_fn = fn
        # evaluate() ranks in question order, so advance the pointer in lockstep.
        orig_rank = ranker.rank
        def ranked(seeds, rel, k):
            r = orig_rank(seeds, rel, k)
            idx["i"] = min(idx["i"] + 1, len(qvecs) - 1)
            return r
        ranker.rank = ranked
    return evaluate(ranker, questions)
```

- [ ] **Step 5: Run report unit test + a synthetic end-to-end run**

Run:
```bash
pytest rgdb-eval/tests/test_report.py -v
python - <<'PY'
from rgdb_eval.synthetic import make_probe
from rgdb_eval.rankers.ppr import PPRRanker
from rgdb_eval.report import evaluate, to_markdown
g, qs = make_probe("coherent", seed=3)
print(to_markdown([evaluate(PPRRanker(g), qs)]))
PY
```
Expected: `test_report.py` 1 passed; the inline run prints a markdown table with a `### hop3` section and an `untyped-ppr` row.

- [ ] **Step 6: Commit**

```bash
git add rgdb-eval/rgdb_eval/report.py rgdb-eval/rgdb_eval/run.py rgdb-eval/tests/test_report.py rgdb-eval/results/.gitkeep
git commit -m "eval: add evaluation aggregation + runner CLI with markdown report"
```

---

### Task 10: Capture the pre-rewrite baseline

**Files:**
- Create: `rgdb-eval/results/baseline-metaqa.md` (generated)

**Interfaces:**
- Consumes: everything above + MetaQA data placed under `rgdb-eval/data/MetaQA/`.
- Produces: the committed baseline table (vector-only, vector+2hop, untyped-ppr, rgdb-current) that sub-project B compares against.

**Background:** This is the "before" measurement the whole sequencing decision exists to capture. Requires MetaQA data on disk and bindings installed.

- [ ] **Step 1: Confirm data is present**

Run: `ls rgdb-eval/data/MetaQA/`
Expected: `kb.txt`, `qa_test_1hop.txt`, `qa_test_2hop.txt`, `qa_test_3hop.txt`. If missing, follow `rgdb-eval/data/README.md` to obtain them before continuing.

- [ ] **Step 2: Ensure bindings are installed (so rgdb-current is not skipped)**

Run: `python rgdb-eval/scripts/smoke_bindings.py`
Expected: `bindings smoke ok: [...]`. If it errors, run `maturin develop -m rgdb-python/Cargo.toml`.

- [ ] **Step 3: Run the full baseline**

Run:
```bash
cd rgdb-eval && python -m rgdb_eval.run --data data/MetaQA --limit 1000 --out results/baseline-metaqa.md && cd ..
```
Expected: `wrote results/baseline-metaqa.md`, and the run prints no `[warn] rgdb-current contender skipped` line (if it does, bindings aren't installed — fix and rerun).

- [ ] **Step 4: Sanity-check the table**

Run: `sed -n '1,40p' rgdb-eval/results/baseline-metaqa.md`
Expected: `### hop1/hop2/hop3` sections, each with rows `vector-only`, `vector+2hop`, `untyped-ppr`, `rgdb-current`, and numeric metric cells.

- [ ] **Step 5: Commit the baseline**

```bash
git add rgdb-eval/results/baseline-metaqa.md
git commit -m "eval: capture pre-rewrite MetaQA baseline for all contenders"
```

---

## Self-Review

**Spec coverage (sub-project C section):**
- `rgdb-eval/` Python package via bindings → Tasks 1–2. ✓
- MetaQA dataset, ~1,000/hop, exact seeding, bypass intent classifier → Task 8 (`relation=None`, `topic_id` seed), Task 9 (`--limit 1000`). ✓
- Synthetic probe suite, coherent + incoherent families → Task 7. ✓
- Contenders 1–4 behind one interface → Tasks 4 (ppr), 5 (vector, vector+2hop), 6 (rgdb-current); protocol in Task 4. ✓
- Metrics Hits@k / recall@k / MRR at k∈{1,5,10,20} by hop → Task 3 + Task 9 aggregation. ✓
- One command → markdown table under `results/` → Task 9; committed baseline → Task 10. ✓
- maturin-on-Windows prerequisite (flagged in spec review) → Task 1 up front. ✓
- Contender 5 (new RGDB + ablation) is explicitly deferred to sub-project B (spec: "added after B"). ✓

**Placeholder scan:** No TBD/TODO. Every code step is complete and runnable. The `data/README.md` documents an out-of-band download (data is licensed separately and not committed) rather than leaving a code gap. ✓

**Type consistency:** `Ranker.rank(seeds, query_relation, k) -> list[int]` is identical across `PPRRanker`, `VectorRanker`, `VectorTwoHopRanker`, `CurrentRgdbRanker`. `TypedGraph`/`Question` field names match every consumer. `K_VALUES` used identically in metrics and report. ✓

**Note for executor:** Task 9's per-question vector embedding relies on `evaluate()` ranking questions in list order; the runner advances the query-vector pointer in lockstep. If sub-project B reworks `evaluate()` to reorder questions, update the lockstep logic accordingly.
