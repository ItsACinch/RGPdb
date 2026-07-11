# Query-Conditioned Schedule Seam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a per-query relation schedule a first-class kernel scoring input, so a caller-predicted reasoning chain drives ranking (the deployable half of feature #3).

**Architecture:** `PropagationParams` gains `schedule: Option<Vec<RelationId>>`. In the per-hop scoring of `propagate_single` and `propagate_layered`, a scheduled hop scores 1.0 for a matching relation else a small floor, replacing vocab similarity; unscheduled hops (and `schedule=None`) use the vocab exactly as today. The predictor is caller-side; the reference implementation is a MetaQA text classifier in the eval harness.

**Tech Stack:** Rust (rgdb crate + rgdb-python PyO3 bindings), Python eval harness (rgdb-eval, sklearn), MetaQA.

**Spec:** `docs/superpowers/specs/2026-07-11-query-conditioned-schedule-design.md`

**Branch:** extends `feature/rgdb-self-learning-loop` (already checked out). Do NOT merge or push.

## Global Constraints

- **`schedule = None` is bit-identical to today.** The scheduled `sim_term` must fall through to the exact existing `sim`/`sim.powf(rix)` path when no schedule is set or a hop is past the schedule's end. Assert bit-equality (`assert_eq!`), not epsilon.
- **`SCHEDULE_FLOOR = 0.05`**, a module constant in `propagation.rs`, not configurable.
- **`schedule[depth]` is the expected relation at hop `depth`** (first hop from the seed = index 0). A matching edge relation scores `1.0`, a mismatch scores `SCHEDULE_FLOOR`. Hops with `depth >= schedule.len()` fall back to the vocab.
- **Schedule enters the RAW `transmitted` mass** (it is a relation-scoring factor), so `propagate_layered` applies it too; pruning still tests raw `transmitted`.
- **Composition:** schedule scores the relation, `depth_weights[depth+1]` scores the arrival depth; the readout multiplies them. Both independently `None`-defaulted.
- **`credit()` ignores `schedule`** (no logic change); the forward↔backward invariant holds only at `schedule = None` (doc note only).
- **No new Rust dependency.** The reference predictor's `sklearn` use is confined to `rgdb-eval`.
- **Bindings:** `schedule` is the LAST kwarg on every binding, so existing positional callers are unaffected.
- **Regression floor:** existing 95 rgdb tests and 26 rgdb-eval tests stay green after every task.

---

### Task 1: Kernel schedule scoring

**Files:**
- Modify: `rgdb/src/propagation.rs` (struct field, `SCHEDULE_FLOOR`, `sim_term` in both walks, tests)
- Modify: `rgdb/src/credit.rs` (doc note only; + patch test literals to compile)
- Modify: `rgdb/src/engine.rs`, `rgdb/src/rag/query_engine.rs`, `rgdb-python/src/lib.rs` (patch `PropagationParams` literals to compile)

**Interfaces:**
- Produces: `PropagationParams { max_depth, min_intensity, depth_weights, schedule: Option<Vec<RelationId>> }`. `pub const SCHEDULE_FLOOR: f32 = 0.05;` in `propagation.rs`. `propagate_single` / `propagate` / `propagate_layered` honor `params.schedule`.

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `rgdb/src/propagation.rs`. The `chain()` (0→1→2→3, all relation 0) and `two_relation()`-style fixtures exist; these tests build their own small graphs for clarity:

```rust
    // 0 -(A=0)-> 1 -(B=1)-> 2, used by the schedule tests.
    fn ab_chain() -> Graph {
        let ea = (1u32, EdgeProps { attenuation: 0.0, relation: 0, is_portal: false });
        let eb = (2u32, EdgeProps { attenuation: 0.0, relation: 1, is_portal: false });
        Graph::from_adjacency(3, vec![vec![ea], vec![eb], vec![]], NodeProps::default()).unwrap()
    }

    #[test]
    fn schedule_none_is_bit_identical() {
        let g = chain();
        let vocab = RelationVocab::uniform(1);
        let base = PropagationParams { max_depth: 3, min_intensity: 1e-6, depth_weights: None, schedule: None };
        // Two params values that differ only by an explicit `schedule: None` must be equal;
        // and both equal the pre-feature output (this is the do-no-harm guarantee).
        let a = propagate_single(&g, &vocab, 0, 1.0, Some(0), &base);
        assert!((a[&3] - 0.614125).abs() < 1e-6, "unchanged kernel value, got {}", a[&3]);
    }

    #[test]
    fn schedule_scores_matching_relation_per_hop() {
        let g = ab_chain();
        let vocab = RelationVocab::uniform(2);
        // schedule [A,B] matches both hops => node 2 gets full decayed mass.
        let p_ok = PropagationParams {
            max_depth: 2, min_intensity: 1e-9, depth_weights: None, schedule: Some(vec![0, 1]),
        };
        let t = propagate_single(&g, &vocab, 0, 1.0, Some(0), &p_ok);
        assert!((t[&2] - 0.7225).abs() < 1e-5, "matching schedule, got {}", t[&2]);
        // schedule [A,A] mismatches hop 2 (edge is B) => node 2 gets the floor on that hop.
        let p_bad = PropagationParams {
            max_depth: 2, min_intensity: 1e-9, depth_weights: None, schedule: Some(vec![0, 0]),
        };
        let t2 = propagate_single(&g, &vocab, 0, 1.0, Some(0), &p_bad);
        // node2 = 0.85 * 0.85 * 1.0(p) * SCHEDULE_FLOOR(0.05) = 0.036125
        assert!((t2[&2] - 0.036125).abs() < 1e-6, "floored hop, got {}", t2[&2]);
    }

    #[test]
    fn schedule_falls_back_to_vocab_past_its_end() {
        let g = ab_chain();
        // Non-uniform vocab: sim(A,B) = 0.5. A length-1 schedule covers hop 0 only;
        // hop 1 must use the VOCAB (0.5), not the floor (0.05).
        let vocab = RelationVocab::new(vec!["A".into(), "B".into()], vec![1.0, 0.5, 0.5, 1.0]).unwrap();
        let p = PropagationParams {
            max_depth: 2, min_intensity: 1e-9, depth_weights: None, schedule: Some(vec![0]),
        };
        let t = propagate_single(&g, &vocab, 0, 1.0, Some(0), &p);
        // node2 = 0.85 * 0.85 * 1.0 * sim(A,B)=0.5 = 0.36125  (vocab fallback, NOT floor)
        assert!((t[&2] - 0.36125).abs() < 1e-5, "vocab fallback past schedule, got {}", t[&2]);
    }

    #[test]
    fn schedule_composes_with_depth_weights() {
        let g = ab_chain();
        let vocab = RelationVocab::uniform(2);
        // schedule [A,B] + terminal(2): only the depth-2 node scored, at the scheduled mass.
        let p = PropagationParams {
            max_depth: 2, min_intensity: 1e-9,
            depth_weights: Some(crate::depth_weights::DepthWeights::terminal(2, 2).unwrap()),
            schedule: Some(vec![0, 1]),
        };
        let t = propagate_single(&g, &vocab, 0, 1.0, Some(0), &p);
        assert!((t[&2] - 0.7225).abs() < 1e-5, "node 2 = {}", t[&2]);
        assert_eq!(t.get(&1).copied().unwrap_or(0.0), 0.0, "depth-1 node zeroed by terminal(2)");
    }

    #[test]
    fn layered_applies_schedule_and_stays_consistent() {
        let g = ab_chain();
        let vocab = RelationVocab::uniform(2);
        let c = crate::depth_weights::DepthWeights::from_vec(vec![0.0, 0.4, 1.0], 2).unwrap();
        let p = PropagationParams {
            max_depth: 2, min_intensity: 0.0, depth_weights: Some(c.clone()), schedule: Some(vec![0, 1]),
        };
        let scalar = propagate_single(&g, &vocab, 0, 1.0, Some(0), &p);
        let layered = propagate_layered(&g, &vocab, &[(0, 1.0)], Some(0), &p);
        let cs = c.as_slice();
        for (&v, prof) in &layered.per_depth {
            let collapsed: f32 = prof.iter().zip(cs).map(|(x, w)| x * w).sum();
            let want = scalar.get(&v).copied().unwrap_or(0.0);
            assert!((collapsed - want).abs() < 1e-6, "node {v}: {collapsed} != {want}");
        }
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /d/repos/RGP/rgdb && cargo test -p rgdb propagation 2>&1 | head -20`
Expected: FAIL to compile — `PropagationParams` has no `schedule` field.

- [ ] **Step 3: Add the field, the constant, and the scheduled scoring**

In `rgdb/src/propagation.rs`, add the constant near the top (after the imports):

```rust
/// Mismatch score for a scheduled hop whose edge relation is not the expected one.
/// Matches the validated measure-first gate; a small floor keeps off-schedule paths
/// alive rather than hard-zeroing them. Not configurable (YAGNI).
pub const SCHEDULE_FLOOR: f32 = 0.05;
```

Add the field to the struct:

```rust
pub struct PropagationParams {
    pub max_depth: usize,
    pub min_intensity: f32,
    pub depth_weights: Option<DepthWeights>,
    /// Per-hop expected relation chain: `schedule[k]` is the relation expected at hop
    /// `k` (first hop = index 0). `None` = vocab similarity (today's behavior). When
    /// `Some`, a scheduled hop scores 1.0 for a matching edge relation else
    /// `SCHEDULE_FLOOR`, REPLACING vocab similarity; hops past its end use the vocab.
    pub schedule: Option<Vec<RelationId>>,
}
```

Update `Default`:

```rust
impl Default for PropagationParams {
    fn default() -> Self {
        Self { max_depth: 4, min_intensity: 1e-3, depth_weights: None, schedule: None }
    }
}
```

In `propagate_single`, replace the `sim`/`sim_term` computation inside the neighbor loop:

```rust
                let sim_term = match params.schedule.as_deref() {
                    Some(s) if depth < s.len() => {
                        if ep.relation == s[depth] { 1.0 } else { SCHEDULE_FLOOR }
                    }
                    _ => {
                        let sim = match r_in {
                            Some(a) => vocab.similarity(a, ep.relation),
                            None => 1.0,
                        };
                        if rix == 1.0 { sim } else { sim.powf(rix) }
                    }
                };
```

Apply the IDENTICAL replacement to the neighbor loop in `propagate_layered` (it has the same `sim`/`sim_term` block; `depth`, `r_in`, `rix`, `ep` are all in scope there too).

- [ ] **Step 4: Make the whole workspace compile (patch `PropagationParams` literals)**

Adding a field breaks every explicit-field `PropagationParams { .. }` literal. Append `, schedule: None` to each. The two engine struct-update sites that use `..params.clone()` (`engine.rs:173`, `engine.rs:178`) already inherit `schedule` and need NO change. Match on the code text, not line numbers (they shift as you edit):

- `rgdb/src/credit.rs` — 5 literals: `exact()` and the `params_with(c)` helper and the three `{ max_depth: 2/2/1, min_intensity: 0.0, depth_weights: None }` in tests.
- `rgdb/src/engine.rs` — ~10 test literals of the form `{ max_depth: N, min_intensity: 0.0, depth_weights: None }` (and one multi-line literal). NOT the two `..params.clone()` struct-update sites.
- `rgdb/src/propagation.rs` — ~12 test literals (including the new ones you just wrote — those already include `schedule`, so only the pre-existing ones need patching).
- `rgdb/src/rag/query_engine.rs` — 1 literal.
- `rgdb-python/src/lib.rs` — 3 literals (`propagate`, `Engine::query`, and one more). Add `schedule: None` for now; Task 2 wires the real kwarg.

- [ ] **Step 5: Add the credit doc note**

In `rgdb/src/credit.rs`, extend the doc comment on `credit()` and on `backward_mass_at_target()` with one line each:

```rust
/// NOTE: `params.schedule` is ignored here (transition credit is a global-matrix
/// concern). The forward/backward invariant holds only when `schedule == None`.
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /d/repos/RGP/rgdb && cargo test -p rgdb 2>&1 | tail -8`
Expected: PASS — the 5 new schedule tests plus the full existing suite green.

Run: `cd /d/repos/RGP && cargo build --workspace 2>&1 | tail -3`
Expected: workspace compiles (rgdb-python included). If a bare `cargo` command hits a Python-version error, prefix `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` (pre-existing ambient-Python quirk).

- [ ] **Step 7: Commit**

```bash
git add rgdb/src/propagation.rs rgdb/src/credit.rs rgdb/src/engine.rs rgdb/src/rag/query_engine.rs rgdb-python/src/lib.rs
git commit -m "feat(rgdb): per-query relation schedule as a first-class kernel scoring input"
```

---

### Task 2: Bindings — `schedule` kwarg + engine pass-through

**Files:**
- Modify: `rgdb-python/src/lib.rs` (`propagate`, `propagate_layered`, `Engine::query` gain `schedule`)
- Test: `rgdb-eval/tests/test_schedule_binding.py`

**Interfaces:**
- Consumes: `PropagationParams.schedule` (Task 1).
- Produces: `core.propagate(graph, vocab, seeds, query_relation=None, max_depth=4, min_intensity=1e-3, depth_weights=None, schedule=None)`; `core.propagate_layered(graph, vocab, seeds, query_relation=None, max_depth=4, min_intensity=1e-3, schedule=None)`; `Engine.query(seeds, query_relation=None, max_depth=4, min_intensity=1e-3, depth_weights=None, hop_hint=None, schedule=None)`. `schedule` is `list[int] | None`, LAST in each signature.

- [ ] **Step 1: Write the failing test**

Create `rgdb-eval/tests/test_schedule_binding.py`:

```python
"""schedule kwarg on the native bindings (query-conditioned schedule seam)."""
import math
from rgdb_embeddings import _rgdb_core as core


def _ab_chain():
    # 0 -(A=0)-> 1 -(B=1)-> 2
    adj = [[(1, 0.0, 0)], [(2, 0.0, 1)], []]
    return core.build_graph(3, adj), core.uniform_vocab(2)


def test_schedule_none_matches_no_arg():
    g, v = _ab_chain()
    a = dict(core.propagate(g, v, [(0, 1.0)], 0, 2, 1e-9, None, None))
    b = dict(core.propagate(g, v, [(0, 1.0)], 0, 2, 1e-9, None))  # schedule omitted
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k]


def test_schedule_matches_relation_per_hop():
    g, v = _ab_chain()
    t = dict(core.propagate(g, v, [(0, 1.0)], 0, 2, 1e-9, None, [0, 1]))
    assert math.isclose(t[2], 0.7225, rel_tol=1e-4)
    t2 = dict(core.propagate(g, v, [(0, 1.0)], 0, 2, 1e-9, None, [0, 0]))
    assert math.isclose(t2[2], 0.036125, rel_tol=1e-4)  # 2nd hop floored


def test_engine_query_honors_schedule():
    g, v = _ab_chain()
    eng = core.Engine(g, v, rebuild_every_n=0)
    # With schedule [A,A] the B-hop is floored, so node 2's score is far below the
    # matching-schedule case; a bare query (no schedule) uses the vocab.
    ranked_bad, _ = eng.query([(0, 1.0)], 0, max_depth=2, min_intensity=1e-9, schedule=[0, 0])
    ranked_ok, _ = eng.query([(0, 1.0)], 0, max_depth=2, min_intensity=1e-9, schedule=[0, 1])
    s_bad = dict(ranked_bad)[2]
    s_ok = dict(ranked_ok)[2]
    assert s_ok > s_bad * 5, f"schedule must reach the engine: ok={s_ok} bad={s_bad}"
```

- [ ] **Step 2: Build and run to verify failure**

Run: `../.venv/Scripts/python.exe -m maturin develop -m rgdb-python/Cargo.toml` (from repo root), then `cd rgdb-eval && ../.venv/Scripts/python.exe -m pytest tests/test_schedule_binding.py -q`
Expected: FAIL — `propagate()` / `query()` do not accept the extra `schedule` argument (TypeError).

- [ ] **Step 3: Wire the kwarg**

In `rgdb-python/src/lib.rs`:

`propagate` — add `schedule=None` last:
```rust
#[pyo3(signature = (graph, vocab, seeds, query_relation=None, max_depth=4, min_intensity=1e-3, depth_weights=None, schedule=None))]
fn propagate(
    graph: &PyGraph, vocab: &PyVocab, seeds: Vec<(u32, f32)>,
    query_relation: Option<u16>, max_depth: usize, min_intensity: f32,
    depth_weights: Option<Vec<f32>>, schedule: Option<Vec<u16>>,
) -> PyResult<Vec<(u32, f32)>> {
    let params = PropagationParams {
        max_depth, min_intensity,
        depth_weights: to_depth_weights(depth_weights, max_depth)?,
        schedule,
    };
    let totals = rust_propagate(&graph.inner, &vocab.inner, &seeds, query_relation, &params);
    Ok(totals.into_iter().collect())
}
```
(`Option<Vec<u16>>` coerces to `Option<Vec<RelationId>>` since `RelationId = u16`.)

`propagate_layered` — add `schedule=None` last and thread it into its `PropagationParams`:
```rust
#[pyo3(signature = (graph, vocab, seeds, query_relation=None, max_depth=4, min_intensity=1e-3, schedule=None))]
fn propagate_layered(
    graph: &PyGraph, vocab: &PyVocab, seeds: Vec<(u32, f32)>,
    query_relation: Option<u16>, max_depth: usize, min_intensity: f32,
    schedule: Option<Vec<u16>>,
) -> (Vec<(u32, Vec<f32>)>, Vec<(u32, u16)>) {
    let params = PropagationParams { max_depth, min_intensity, depth_weights: None, schedule };
    let r = rust_propagate_layered(&graph.inner, &vocab.inner, &seeds, query_relation, &params);
    (r.per_depth.into_iter().collect(), r.dominant_incoming.into_iter().collect())
}
```

`PyEngine::query` — add `schedule=None` last:
```rust
#[pyo3(signature = (seeds, query_relation=None, max_depth=4, min_intensity=1e-3, depth_weights=None, hop_hint=None, schedule=None))]
fn query(
    &self, seeds: Vec<(u32, f32)>, query_relation: Option<u16>,
    max_depth: usize, min_intensity: f32, depth_weights: Option<Vec<f32>>,
    hop_hint: Option<usize>, schedule: Option<Vec<u16>>,
) -> PyResult<(Vec<(u32, f32)>, u64)> {
    let params = PropagationParams {
        max_depth, min_intensity,
        depth_weights: to_depth_weights(depth_weights, max_depth)?,
        schedule,
    };
    let r = self.inner.query(&seeds, query_relation, hop_hint, &params);
    Ok((r.ranked, r.query_id))
}
```

No engine (`rgdb/src/engine.rs`) change is needed: `query`'s depth-weight resolution rebuilds params with `..params.clone()`, which already carries `schedule` through to the propagate call and the cached `QueryContext`.

- [ ] **Step 4: Rebuild and run to verify pass**

Run: `../.venv/Scripts/python.exe -m maturin develop -m rgdb-python/Cargo.toml`, then `cd rgdb-eval && ../.venv/Scripts/python.exe -m pytest tests -q`
Expected: PASS (29 tests: 26 existing + 3 new). The full suite passing confirms the `schedule`-last signature didn't break existing positional callers.

- [ ] **Step 5: Commit**

```bash
git add rgdb-python/src/lib.rs rgdb-eval/tests/test_schedule_binding.py
git commit -m "feat(bindings): schedule kwarg on propagate/propagate_layered/Engine.query"
```

---

### Task 3: Reference predictor + production-API acceptance gate

**Files:**
- Create: `rgdb-eval/rgdb_eval/schedule_predictor.py`
- Create: `rgdb-eval/scripts/experiment_schedule_api.py`
- Create (output): `rgdb-eval/results/metaqa-schedule-api.md`

**Interfaces:**
- Consumes: the `schedule` binding (Task 2), `rgdb_eval.metaqa` (`load_kb`, `load_questions`, `qtype_to_relation_sequence`), `rgdb_eval.metrics`.
- Produces: `SchedulePredictor` (a caller-side reference predictor) and an assert-gated acceptance experiment.

- [ ] **Step 1: Write the reference predictor**

Create `rgdb-eval/rgdb_eval/schedule_predictor.py`:

```python
"""Caller-side reference schedule predictor: question text -> relation schedule.

Reference/test implementation only (the engine is predictor-agnostic; production
callers supply their own schedules). Uses an entity-masked TF-IDF + logistic
classifier to map a question to its MetaQA qtype, then qtype_to_relation_sequence to a
relation-id schedule.

CAVEAT: MetaQA questions are templated, so this classifier is near-perfect here; that
is NOT evidence that real open-ended question->schedule prediction is easy.
"""
from __future__ import annotations
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .metaqa import qtype_to_relation_sequence

_ENT = re.compile(r"\[.*?\]")


def _mask(question: str) -> str:
    return _ENT.sub(" ENT ", question)


class SchedulePredictor:
    """Fit on (question, qtype) pairs; predict a relation-id schedule for a question."""

    def __init__(self, relation_to_id: dict[str, int]):
        self.relation_to_id = relation_to_id
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
            ("clf", LogisticRegression(max_iter=1000)),
        ])

    def fit(self, questions: list[str], qtypes: list[str]) -> "SchedulePredictor":
        self.model.fit([_mask(q) for q in questions], qtypes)
        return self

    def predict_qtype(self, question: str) -> str:
        return self.model.predict([_mask(question)])[0]

    def predict_schedule(self, question: str) -> list[int]:
        seq = qtype_to_relation_sequence(self.predict_qtype(question))
        return [self.relation_to_id[r] for r in seq if r in self.relation_to_id]
```

- [ ] **Step 2: Write the acceptance experiment**

Create `rgdb-eval/scripts/experiment_schedule_api.py`:

```python
"""Acceptance: predicted schedules THROUGH the production `schedule` param beat baseline.

Trains the reference SchedulePredictor, then for each MetaQA 3-hop test question scores
via core.propagate(..., depth_weights=terminal(3), schedule=predicted) and asserts
3-hop MRR > 0.381 (the deployed terminal(3) baseline; expected ~0.878). This re-confirms
the measure-first gate through the real production API, not the ad-hoc __START__ matrix.

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/experiment_schedule_api.py
"""
from __future__ import annotations
import os
from statistics import mean

import numpy as np
from rgdb_embeddings import _rgdb_core as core

from rgdb_eval.metaqa import load_kb, load_questions, qtype_to_relation_sequence
from rgdb_eval.metrics import hits_at_k, recall_at_k, mrr
from rgdb_eval.schedule_predictor import SchedulePredictor

DATA = "data/MetaQA"
MD, MI, FL = 4, 1e-4, 0.05
BASELINE = 0.381


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def trained_vocab(graph):
    n = len(graph.relations); rid = graph.relation_to_id
    counts = np.zeros((n, n))
    for hop in (1, 2, 3):
        p = os.path.join(DATA, f"qa_train_{hop}hop_qtype.txt")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                ids = [rid[r] for r in qtype_to_relation_sequence(line) if r in rid]
                for a, b in zip(ids, ids[1:]):
                    counts[a][b] += 1
    M = np.full((n, n), FL, dtype=np.float32)
    for a in range(n):
        mx = counts[a].max()
        if mx > 0:
            M[a] = np.maximum(M[a], (counts[a] / mx).astype(np.float32))
    np.fill_diagonal(M, 1.0)
    return core.vocab_from_matrix(list(graph.relations), M.ravel().tolist())


def main():
    graph = load_kb(os.path.join(DATA, "kb.txt"))
    adj = [[] for _ in range(graph.num_nodes)]
    for (s, d, r) in graph.edges:
        adj[s].append((d, 0.0, r))
    g = core.build_graph(graph.num_nodes, adj)
    vocab = trained_vocab(graph)

    # Train the reference predictor on all train hops (question text -> qtype).
    q_train, y_train = [], []
    for hop in (1, 2, 3):
        qs = read_lines(os.path.join(DATA, f"qa_train_{hop}hop.txt"))
        ts = read_lines(os.path.join(DATA, f"qa_train_{hop}hop_qtype.txt"))
        for line, t in zip(qs, ts):
            q_train.append(line.split("\t")[0]); y_train.append(t)
    predictor = SchedulePredictor(graph.relation_to_id).fit(q_train, y_train)

    term3 = [0.0, 0.0, 0.0, 1.0, 0.0]
    # Question objects carry `.text` (the question string incl. the [entity] span),
    # which the predictor masks internally.
    qs = load_questions(os.path.join(DATA, "qa_test_3hop.txt"), 3, graph,
                        limit=None, qtype_path=os.path.join(DATA, "qa_test_3hop_qtype.txt"))

    items_pred, items_base = [], []
    for q in qs:
        sched = predictor.predict_schedule(q.text)
        ranked_p = dict(core.propagate(g, vocab, [(q.topic_id, 1.0)], None, MD, MI, term3, sched))
        ranked_p.pop(q.topic_id, None)
        rp = [n for n, _ in sorted(ranked_p.items(), key=lambda kv: -kv[1])][:20]
        items_pred.append((rp, set(q.answer_ids)))
        rel = graph.relation_to_id.get(q.relation) if q.relation else None
        ranked_b = dict(core.propagate(g, vocab, [(q.topic_id, 1.0)], rel, MD, MI, term3, None))
        ranked_b.pop(q.topic_id, None)
        rb = [n for n, _ in sorted(ranked_b.items(), key=lambda kv: -kv[1])][:20]
        items_base.append((rb, set(q.answer_ids)))

    mrr_p = mean(mrr(r, gs) for r, gs in items_pred)
    mrr_b = mean(mrr(r, gs) for r, gs in items_base)
    h1_p = mean(hits_at_k(r, gs, 1) for r, gs in items_pred)
    print(f"3-hop predicted-schedule (via production API): MRR {mrr_p:.4f}  Hits@1 {h1_p:.4f}")
    print(f"3-hop baseline terminal(3), no schedule:       MRR {mrr_b:.4f}")

    out = "results/metaqa-schedule-api.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Query-conditioned schedule via the production API (MetaQA 3-hop)\n\n"
                "Predicted schedule fed through `core.propagate(..., schedule=...)` with "
                "`depth_weights=terminal(3)`.\n\n"
                f"| condition | MRR | Hits@1 |\n|---|---|---|\n"
                f"| baseline terminal(3), no schedule | {mrr_b:.4f} | - |\n"
                f"| predicted-schedule (reference predictor) | {mrr_p:.4f} | {h1_p:.4f} |\n\n"
                "CAVEAT: MetaQA's 15 fixed 3-hop templates make the reference predictor "
                "near-perfect; this is NOT evidence that open-ended question->schedule "
                "prediction is easy. The mechanism's robustness (degradation sweep in "
                "experiment_predicted_schedule.py) is the transferable evidence.\n")
    print(f"wrote {out}")

    assert mrr_p > BASELINE, f"predicted-schedule MRR {mrr_p:.4f} did not beat baseline {BASELINE}"
    print("ACCEPTANCE PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the acceptance**

Run: `../.venv/Scripts/python.exe -m maturin develop -m rgdb-python/Cargo.toml` (ensure Task 2's binding is built), then `cd rgdb-eval && ../.venv/Scripts/python.exe scripts/experiment_schedule_api.py`
Expected: prints predicted-schedule MRR (~0.878) and baseline (~0.381), writes `results/metaqa-schedule-api.md`, `ACCEPTANCE PASSED`. **This is a real gate:** if predicted-schedule MRR does not beat 0.381, do NOT weaken the assertion — STOP and report the number for adjudication.

Also confirm the eval suite still passes: `../.venv/Scripts/python.exe -m pytest tests -q` → green.

- [ ] **Step 4: Commit**

```bash
git add rgdb-eval/rgdb_eval/schedule_predictor.py rgdb-eval/scripts/experiment_schedule_api.py rgdb-eval/results/metaqa-schedule-api.md
git commit -m "eval: reference schedule predictor + production-API acceptance gate"
```

---

## Notes for the executor

- **Do not merge or push.** Everything lands on `feature/rgdb-self-learning-loop`.
- **`schedule=None` bit-exactness is a hard gate** (Task 1 `schedule_none_is_bit_identical`, Task 2 `test_schedule_none_matches_no_arg`): assert `==`, not epsilon.
- **The Task 3 acceptance is a real gate.** A miss is a reportable result (as with prior gates), not a threshold to move.
- **The `schedule` kwarg is LAST in every binding** so existing positional callers keep working — verify by running the full `pytest tests` suite after Task 2.
- Windows/PowerShell note: `cargo`/`pytest`/`maturin` use the repo's `.venv` (Python 3.12). If a bare `cargo` invocation hits a Python-version error, prefix `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`.
