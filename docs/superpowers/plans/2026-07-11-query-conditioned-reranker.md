# Query-Conditioned Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the online `Reranker` a query-conditioned match feature (candidate's dominant incoming relation == the query's expected final relation) so it helps rather than hurts when a schedule is present.

**Architecture:** Add an `expected_relation: Option<RelationId>` input to the reranker's feature extraction and training, appending one match feature. The engine derives the expected relation from the schedule already in `PropagationParams` and passes it through. A small `reranker_enabled` binding lets the eval exercise it. The reranker stays off by default and `w=0` identity; the match feature is `0.0` without a schedule.

**Tech Stack:** Rust (rgdb crate + rgdb-python bindings), Python eval harness (rgdb-eval), MetaQA.

**Spec:** `docs/superpowers/specs/2026-07-11-query-conditioned-reranker-design.md`

**Branch:** extends `feature/rgdb-self-learning-loop` (pushed, not merged). Do NOT merge or push.

## Global Constraints

- **Do-no-harm:** reranker off by default (`EngineConfig.reranker_enabled = false`, unchanged); `w=0` is a strict identity even with an `expected_relation` supplied; the match feature is `0.0` when `expected_relation` is `None` — so the no-schedule path is byte-unchanged from before #3.
- **Match feature:** `1.0` iff `layered.dominant_incoming[node] == expected_relation`, else `0.0`; `0.0` for all nodes when `expected_relation` is `None`. It is the LAST feature (`dim - 1`).
- **`dim` grows by exactly 1** to `(max_depth+1) + 2 + n_relations + 1`, in BOTH `new()` and `load()`; persistence version bumps `1 -> 2`.
- **Ship the gate-validated full+match feature set** (lean pruning is a recorded follow-up).
- **Expected relation = `schedule.last()`** — no new engine `query`/`record_feedback` argument, no new `QueryContext` field.
- **Behavioral validation split:** the match feature's effect is proven in isolation by the Task 1 unit test and distributionally by the Task 4 acceptance gate; the Task 2 engine test is a wiring/integration test (a single fixture cannot cleanly isolate a distributional effect, because a correct schedule already ranks the matching candidate).
- **Regression floor:** existing 100 rgdb tests and 29 rgdb-eval tests stay green.

---

### Task 1: Reranker match feature

**Files:**
- Modify: `rgdb/src/reranker.rs` (`features`/`rerank`/`update` signatures, match feature, `dim`, persistence version, existing + new tests)

**Interfaces:**
- Consumes: `LayeredResult.dominant_incoming`, `RelationId = u16`.
- Produces: `features(&self, node, layered, graph, expected_relation: Option<RelationId>) -> Vec<f32>`; `rerank(&self, candidates, layered, graph, expected_relation: Option<RelationId>)`; `update(&mut self, candidates, target, layered, graph, expected_relation: Option<RelationId>, signal)`. `dim = (max_depth+1)+2+n_relations+1`.

- [ ] **Step 1: Update the 3 existing reranker tests to the new signatures, then add the new tests**

In `rgdb/src/reranker.rs` tests: change `r.rerank(&mut cands, &layered, &g)` → `r.rerank(&mut cands, &layered, &g, None)` (in `cold_start_is_identity`); `r.update(&cands, 1, &layered, &g, 1.0)` → `r.update(&cands, 1, &layered, &g, None, 1.0)` (in `learns_to_prefer_low_degree_answers` and `save_load_roundtrip`); and any `r.features(n, &layered, &g)` → `r.features(n, &layered, &g, None)`.

Then add:

```rust
    #[test]
    fn match_feature_fires_only_on_the_expected_relation() {
        let (g, _l) = fixture();
        let r = Reranker::new(3, 2, cfg());
        let mut per = HashMap::new();
        per.insert(1u32, vec![0.0, 0.0, 0.5]);
        let mut dom = HashMap::new();
        dom.insert(1u32, 2u16); // node 1 arrived via relation 2
        let layered = LayeredResult { per_depth: per, dominant_incoming: dom };
        let last = r.features(1, &layered, &g, Some(2)); // expected == 2 => match
        let miss = r.features(1, &layered, &g, Some(0)); // expected == 0 => no match
        let none = r.features(1, &layered, &g, None);    // no schedule => 0
        let m = last.len() - 1;
        assert_eq!(last[m], 1.0);
        assert_eq!(miss[m], 0.0);
        assert_eq!(none[m], 0.0);
    }

    #[test]
    fn cold_start_identity_holds_with_expected_relation() {
        let (g, layered) = fixture();
        let r = Reranker::new(1, 2, cfg());
        let mut cands = vec![(1u32, 0.9f32), (2, 0.8), (3, 0.7)];
        let before = cands.clone();
        r.rerank(&mut cands, &layered, &g, Some(0)); // w=0 => identity even with a schedule
        assert_eq!(cands, before);
    }

    #[test]
    fn learns_the_match_signal_not_a_global_relation() {
        // Two queries with DIFFERENT expected relations; each query's answer is the
        // candidate whose incoming relation matches THAT query's expected relation. The
        // global one-hot nets to ~0 (each relation is answer once, distractor once), so
        // only the match feature separates them.
        let g = Graph::from_adjacency(3, vec![vec![], vec![], vec![]], NodeProps::default()).unwrap();
        let mut r = Reranker::new(3, 2, cfg());
        let mut per = HashMap::new();
        per.insert(1u32, vec![0.0, 0.0, 0.5]);
        per.insert(2u32, vec![0.0, 0.0, 0.5]);
        let mut dom = HashMap::new();
        dom.insert(1u32, 1u16);
        dom.insert(2u32, 2u16);
        let layered = LayeredResult { per_depth: per, dominant_incoming: dom };
        let cands = vec![(1u32, 0.5f32), (2u32, 0.5f32)];
        for _ in 0..300 {
            r.update(&cands, 1, &layered, &g, Some(1), 1.0); // expect rel 1 -> node 1
            r.update(&cands, 2, &layered, &g, Some(2), 1.0); // expect rel 2 -> node 2
        }
        let mut a = cands.clone();
        r.rerank(&mut a, &layered, &g, Some(1));
        assert_eq!(a.first().map(|x| x.0), Some(1));
        let mut b = cands.clone();
        r.rerank(&mut b, &layered, &g, Some(2));
        assert_eq!(b.first().map(|x| x.0), Some(2));
    }

    #[test]
    fn save_load_roundtrip_with_match_feature() {
        let (g, layered) = fixture();
        let mut r = Reranker::new(1, 2, cfg());
        let cands = vec![(1u32, 0.5f32), (2, 0.9), (3, 0.5)];
        for _ in 0..20 { r.update(&cands, 1, &layered, &g, Some(0), 1.0); }
        let path = "test_reranker_match_roundtrip.bin";
        r.save(path).unwrap();
        let t = Reranker::load(path).unwrap();
        let f = t.features(2, &layered, &g, Some(0));
        assert!((t.score(&f) - r.score(&f)).abs() < 1e-6);
        let _ = std::fs::remove_file(path);
    }
```

(If `HashMap` isn't already imported in the test module, add `use hashbrown::HashMap;` — mirror the existing `fixture()` which already builds `LayeredResult` maps.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /d/repos/RGP/rgdb && cargo test -p rgdb reranker 2>&1 | head -20`
Expected: FAIL to compile — `features`/`rerank`/`update` don't take `expected_relation`.

- [ ] **Step 3: Add the param, match feature, dim bump, version bump**

Change `features` (signature + append the match feature as the last slot):

```rust
    pub fn features(&self, node: NodeId, layered: &LayeredResult, graph: &Graph,
                    expected_relation: Option<RelationId>) -> Vec<f32> {
        let mut f = vec![0.0f32; self.dim];
        let mut i = 0;
        if let Some(prof) = layered.per_depth.get(&node) {
            for d in 0..=self.max_depth {
                f[i] = prof.get(d).copied().unwrap_or(0.0);
                i += 1;
            }
        } else {
            i += self.max_depth + 1;
        }
        let score: f32 = layered.per_depth.get(&node).map(|p| p.iter().sum()).unwrap_or(0.0);
        f[i] = (1.0 + score).ln();
        i += 1;
        let deg = graph.neighbors(node).count() as f32;
        f[i] = (1.0 + deg).ln();
        i += 1;
        if let Some(&r) = layered.dominant_incoming.get(&node) {
            let ri = r as usize;
            if ri < self.n_relations {
                f[i + ri] = 1.0;
            }
        }
        // query-conditioned match feature (LAST slot): does this node's dominant incoming
        // relation equal the query's expected final relation? 0.0 when there is no schedule.
        if let Some(er) = expected_relation {
            if layered.dominant_incoming.get(&node) == Some(&er) {
                f[self.dim - 1] = 1.0;
            }
        }
        f
    }
```

Change `rerank`/`update` to take and forward `expected_relation`:

```rust
    pub fn rerank(&self, candidates: &mut Vec<(NodeId, f32)>, layered: &LayeredResult,
                  graph: &Graph, expected_relation: Option<RelationId>) {
        let scored: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(idx, &(node, _))| (idx, self.score(&self.features(node, layered, graph, expected_relation))))
            .collect();
        let mut order: Vec<usize> = (0..candidates.len()).collect();
        order.sort_by(|&a, &b| scored[b].1.partial_cmp(&scored[a].1).unwrap_or(std::cmp::Ordering::Equal));
        let reordered: Vec<(NodeId, f32)> = order.iter().map(|&i| candidates[i]).collect();
        *candidates = reordered;
    }

    pub fn update(
        &mut self, candidates: &[(NodeId, f32)], target: NodeId, layered: &LayeredResult,
        graph: &Graph, expected_relation: Option<RelationId>, signal: f32,
    ) {
        let lr = self.cfg.learning_rate * signal;
        let k = self.cfg.top_k.min(candidates.len());
        for &(node, _) in &candidates[..k] {
            let feats = self.features(node, layered, graph, expected_relation);
            let y = if node == target { 1.0 } else { 0.0 };
            let p = sigmoid(self.score(&feats));
            let g = y - p;
            for (wj, xj) in self.w.iter_mut().zip(&feats) {
                *wj += lr * (g * xj - self.cfg.l2 * *wj);
            }
            self.b += lr * g;
        }
    }
```

Bump `dim` in BOTH `new()` and `load()` to:
```rust
        let dim = (max_depth + 1) + 2 + n_relations + 1;
```
(update the `new()` comment to mention the `+ 1 query-conditioned match feature`).

Bump the persistence version: in `write_snapshot` change `f.write_u32::<LittleEndian>(1)?;` → `f.write_u32::<LittleEndian>(2)?;`; in `load` change the version check from `!= 1` to `!= 2`.

Ensure `RelationId` is imported: the `use crate::graph::{...}` line in `reranker.rs` must include `RelationId` (add it alongside `NodeId`).

- [ ] **Step 4: Run the reranker tests**

Run: `cd /d/repos/RGP/rgdb && cargo test -p rgdb reranker 2>&1 | tail -8`
Expected: PASS (4 new + 3 updated). NOTE: the full crate will not compile until Task 2 (the engine still calls the old reranker signatures) — verify Task 1 with the reranker tests only.

- [ ] **Step 5: Commit**

```bash
git add rgdb/src/reranker.rs
git commit -m "feat(rgdb): reranker query-conditioned match feature (expected final relation)"
```

---

### Task 2: Engine wiring — derive and pass the expected relation

**Files:**
- Modify: `rgdb/src/engine.rs` (`query`, `record_feedback`, one wiring test)

**Interfaces:**
- Consumes: `Reranker::{rerank,update}` with `expected_relation` (Task 1); `PropagationParams.schedule`.
- Produces: the engine passes `schedule.last()` to the reranker in `query` and `record_feedback`.

- [ ] **Step 1: Write the failing wiring test**

Add to the `tests` module in `rgdb/src/engine.rs`:

```rust
    #[test]
    fn engine_reranker_runs_and_trains_under_a_schedule() {
        // Wiring/integration: reranker_enabled + a schedule -> query returns a valid
        // ranking and feedback trains without error over the schedule path (the engine
        // derives expected_relation = schedule.last() and hands it to the reranker).
        // Behavioral proof of the match feature is in reranker.rs (unit) and the eval gate.
        let a = |d, r| (d as u32, EdgeProps { attenuation: 0.0, relation: r, is_portal: false });
        // 0 -A-> 1 -B-> 2  (node 2 reachable at depth 2)
        let g = Graph::from_adjacency(3, vec![vec![a(1, 0)], vec![a(2, 1)], vec![]], NodeProps::default()).unwrap();
        let prior = RelationVocab::with_names_uniform(vec!["A".into(), "B".into()]);
        let e = RgdbEngine::with_engine_config(
            g, prior,
            TransitionConfig { rebuild_every_n: 0, ..TransitionConfig::default() },
            EngineConfig { reranker_enabled: true, ..EngineConfig::default() });
        let p = PropagationParams {
            max_depth: 2, min_intensity: 0.0, depth_weights: None, schedule: Some(vec![0, 1]),
        };
        let r = e.query(&[(0, 1.0)], Some(0), None, &p);
        assert!(!r.ranked.is_empty());
        assert!(e.record_feedback(r.query_id, 2, 1.0).is_ok(), "feedback trains under the schedule");
        // a second round still runs cleanly
        let r2 = e.query(&[(0, 1.0)], Some(0), None, &p);
        assert!(!r2.ranked.is_empty());
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d/repos/RGP/rgdb && cargo test -p rgdb engine_reranker_runs 2>&1 | tail -15`
Expected: FAIL to compile — `rr.rerank`/`rr.update` are still called with the old signatures in `query`/`record_feedback`.

- [ ] **Step 3: Wire the expected relation through the engine**

In `query`, inside the `if self.engine_cfg.reranker_enabled { ... }` block, derive `expected_relation` from the resolved schedule and pass it to `rerank` (leave the `topk_n`/`ranked_topk` capture above it unchanged):

```rust
        if self.engine_cfg.reranker_enabled {
            let expected_relation = resolved.schedule.as_ref().and_then(|s| s.last().copied());
            let rr = self.reranker.lock().unwrap();
            let mut head: Vec<(NodeId, f32)> = ranked.iter().take(topk_n).copied().collect();
            rr.rerank(&mut head, &layered, &self.graph, expected_relation);
            for (i, item) in head.into_iter().enumerate() {
                ranked[i] = item;
            }
        }
```

In `record_feedback`, inside its `if self.engine_cfg.reranker_enabled { ... }` block:

```rust
        if self.engine_cfg.reranker_enabled {
            let expected_relation = ctx.params.schedule.as_ref().and_then(|s| s.last().copied());
            let mut rr = self.reranker.lock().unwrap();
            rr.update(&ctx.ranked_topk, target, &ctx.layered, &self.graph, expected_relation, signal);
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /d/repos/RGP/rgdb && cargo test -p rgdb 2>&1 | tail -8`
Expected: PASS — the new wiring test plus the full existing suite (no-schedule reranker tests still pass because `expected_relation` is then `None`).

- [ ] **Step 5: Commit**

```bash
git add rgdb/src/engine.rs
git commit -m "feat(rgdb): engine passes the schedule's expected final relation to the reranker"
```

---

### Task 3: `reranker_enabled` binding

**Files:**
- Modify: `rgdb-python/src/lib.rs` (`Engine::new` gains `reranker_enabled`)
- Test: `rgdb-eval/tests/test_reranker_enabled_binding.py`

**Interfaces:**
- Consumes: `RgdbEngine::with_engine_config`, `EngineConfig` (existing).
- Produces: `core.Engine(graph, vocab, prior_strength=10.0, floor=0.05, decay=1.0, rebuild_every_n=64, reranker_enabled=False)` — `reranker_enabled` LAST, default `False` (backward-compatible).

- [ ] **Step 1: Write the failing test**

Create `rgdb-eval/tests/test_reranker_enabled_binding.py`:

```python
"""Engine(reranker_enabled=...) binding: default off (backward-compatible), can flip on."""
from rgdb_embeddings import _rgdb_core as core


def _chain():
    adj = [[(1, 0.0, 0)], [(2, 0.0, 1)], []]
    return core.build_graph(3, adj), core.uniform_vocab(2)


def test_default_construction_still_works():
    g, v = _chain()
    eng = core.Engine(g, v, rebuild_every_n=0)  # no reranker_enabled -> default False
    ranked, qid = eng.query([(0, 1.0)], 0, max_depth=2, min_intensity=1e-9)
    assert qid > 0 and len(ranked) >= 1


def test_reranker_enabled_kwarg_accepted():
    g, v = _chain()
    eng = core.Engine(g, v, rebuild_every_n=0, reranker_enabled=True)
    # cold reranker is identity, so this behaves like a normal query; just confirm it runs.
    ranked, qid = eng.query([(0, 1.0)], 0, max_depth=2, min_intensity=1e-9, schedule=[0, 1])
    assert qid > 0 and len(ranked) >= 1
    assert eng.record_feedback(qid, 2, 1.0) is None  # trains the reranker, returns None on success
```

- [ ] **Step 2: Build and run to verify failure**

Run: `cd /d/repos/RGP && .venv/Scripts/python.exe -m maturin develop -m rgdb-python/Cargo.toml`, then `cd rgdb-eval && ../.venv/Scripts/python.exe -m pytest tests/test_reranker_enabled_binding.py -q`
Expected: FAIL — `Engine()` does not accept `reranker_enabled` (TypeError).

- [ ] **Step 3: Add the kwarg**

In `rgdb-python/src/lib.rs`, the `#[pymethods] impl PyEngine`'s `#[new]` currently builds via `RgdbEngine::new(...)` with `TransitionConfig`. Change it to accept `reranker_enabled` (LAST) and construct via `with_engine_config`:

```rust
    #[new]
    #[pyo3(signature = (graph, vocab, prior_strength=10.0, floor=0.05, decay=1.0, rebuild_every_n=64, reranker_enabled=false))]
    fn new(graph: &PyGraph, vocab: &PyVocab, prior_strength: f32, floor: f32, decay: f32,
           rebuild_every_n: u32, reranker_enabled: bool) -> Self {
        let tcfg = TransitionConfig { prior_strength, floor, decay, rebuild_every_n };
        let ecfg = EngineConfig { reranker_enabled, ..EngineConfig::default() };
        PyEngine {
            inner: RgdbEngine::with_engine_config(graph.inner.clone(), vocab.inner.clone(), tcfg, ecfg),
        }
    }
```
Add `EngineConfig` to the `use rgdb::engine::{...}` import line in `lib.rs`. (Keep the exact existing `TransitionConfig` field values; the only new behavior is passing `reranker_enabled` through an `EngineConfig`.)

- [ ] **Step 4: Rebuild and run**

Run: `cd /d/repos/RGP && .venv/Scripts/python.exe -m maturin develop -m rgdb-python/Cargo.toml`, then `cd rgdb-eval && ../.venv/Scripts/python.exe -m pytest tests -q`
Expected: PASS (31 = 29 existing + 2 new); existing `core.Engine(...)` callers unaffected (kwarg defaults false, added last).

- [ ] **Step 5: Commit**

```bash
git add rgdb-python/src/lib.rs rgdb-eval/tests/test_reranker_enabled_binding.py
git commit -m "feat(bindings): Engine reranker_enabled kwarg (default off)"
```

---

### Task 4: Eval acceptance — reranker+match beats schedule-alone on degraded schedules

**Files:**
- Create: `rgdb-eval/scripts/experiment_reranker_schedule_api.py`
- Create (output): `rgdb-eval/results/metaqa-reranker-schedule-api.md`

**Interfaces:**
- Consumes: `Engine(..., reranker_enabled=True)` + `query(schedule=...)` + `record_feedback` (Tasks 2, 3).

- [ ] **Step 1: Write the acceptance script**

Create `rgdb-eval/scripts/experiment_reranker_schedule_api.py`:

```python
"""Acceptance: the query-conditioned reranker recovers 3-hop Hits@1 on DEGRADED
schedules, THROUGH the production engine (reranker_enabled + schedule).

For each corruption p, corrupt each gold-schedule relation with prob p (a proxy for
imperfect prediction). schedule-alone = engine with the reranker OFF; reranker+match =
engine with the reranker ON, trained online via record_feedback under the corrupted
schedule's expected final relation. Both evaluated on the same corrupted test schedules.
Gate: reranker+match 3-hop Hits@1 beats schedule-alone at p=0.25 (floor 0.50).

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/experiment_reranker_schedule_api.py
"""
from __future__ import annotations
import os
from statistics import mean

import numpy as np
from rgdb_embeddings import _rgdb_core as core

from rgdb_eval.metaqa import load_kb, parse_qa_line, qtype_to_relation_sequence
from rgdb_eval.metrics import hits_at_k

DATA = "data/MetaQA"
MD, MI, FL = 4, 1e-4, 0.05
TRAIN, TESTN = 3000, 2000
TERM3 = [0.0, 0.0, 0.0, 1.0, 0.0]
P_LEVELS = [0.0, 0.25, 0.4]
GATE_P, GATE_FLOOR = 0.25, 0.50


def trained_flat(graph):
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
    return M.ravel().tolist(), n


def load_sched(graph, split, hop, limit):
    """(topic_id, set(answer_ids), schedule) line-aligned with the qtype file."""
    rid = graph.relation_to_id
    qpath = os.path.join(DATA, f"qa_{split}_{hop}hop.txt")
    tpath = os.path.join(DATA, f"qa_{split}_{hop}hop_qtype.txt")
    qtypes = [l.strip() for l in open(tpath, encoding="utf-8")]
    rows = []
    for i, line in enumerate(open(qpath, encoding="utf-8")):
        if len(rows) >= limit:
            break
        if not line.strip():
            continue
        topic, answers = parse_qa_line(line)
        if topic not in graph.name_to_id:
            continue
        ans = {graph.name_to_id[a] for a in answers if a in graph.name_to_id}
        if not ans:
            continue
        seq = qtype_to_relation_sequence(qtypes[i]) if i < len(qtypes) else []
        sched = [rid[r] for r in seq if r in rid]
        if not sched:
            continue
        rows.append((graph.name_to_id[topic], ans, sched))
    return rows


def corrupt(sched, p, n, rng):
    out = []
    for r in sched:
        if rng.random() < p:
            alt = int(rng.integers(0, n - 1))
            if alt >= r:
                alt += 1
            out.append(alt)
        else:
            out.append(r)
    return out


def hits1(eng, rows, p, n, seed):
    rng = np.random.default_rng(seed)
    tot = []
    for topic, gold, sched in rows:
        sc = corrupt(sched, p, n, rng)
        ranked, _ = eng.query([(topic, 1.0)], None, max_depth=MD, min_intensity=MI,
                              depth_weights=TERM3, schedule=sc)
        ranked = [nid for nid, _ in ranked if nid != topic][:20]
        tot.append(hits_at_k(ranked, gold, 1))
    return mean(tot)


def train(eng, rows, p, n, seed):
    rng = np.random.default_rng(seed)
    for topic, gold, sched in rows:
        sc = corrupt(sched, p, n, rng)
        _, qid = eng.query([(topic, 1.0)], None, max_depth=MD, min_intensity=MI,
                           depth_weights=TERM3, schedule=sc)
        try:
            eng.record_feedback(qid, next(iter(gold)), 1.0)
        except ValueError:
            pass


def main():
    graph = load_kb(os.path.join(DATA, "kb.txt"))
    adj = [[] for _ in range(graph.num_nodes)]
    for (s, d, r) in graph.edges:
        adj[s].append((d, 0.0, r))
    g = core.build_graph(graph.num_nodes, adj)
    flat, n = trained_flat(graph)
    names = list(graph.relations)
    train_rows = load_sched(graph, "train", 3, TRAIN)
    test_rows = load_sched(graph, "test", 3, TESTN)
    print(f"{len(train_rows)} train / {len(test_rows)} test 3-hop questions")

    rows = []
    for p in P_LEVELS:
        off = core.Engine(g, core.vocab_from_matrix(names, flat), rebuild_every_n=0)
        h_alone = hits1(off, test_rows, p, n, seed=1)           # reranker off
        on = core.Engine(g, core.vocab_from_matrix(names, flat), rebuild_every_n=0,
                         reranker_enabled=True)
        train(on, train_rows, p, n, seed=2)
        h_rerank = hits1(on, test_rows, p, n, seed=1)           # same corrupted test schedules
        rows.append((p, h_alone, h_rerank))
        print(f"p={p}: schedule-alone Hits@1 {h_alone:.4f}  reranker+match Hits@1 {h_rerank:.4f}")

    out = "results/metaqa-reranker-schedule-api.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Query-conditioned reranker via the production engine (MetaQA 3-hop)\n\n"
                "Degraded schedules (each relation corrupted with prob p). Reranker trained "
                "online through record_feedback under the corrupted schedule's expected final "
                "relation; both variants evaluated on the same corrupted test schedules.\n\n"
                "| p | schedule-alone Hits@1 | reranker+match Hits@1 |\n|---|---|---|\n")
        for p, a, b in rows:
            f.write(f"| {p} | {a:.4f} | {b:.4f} |\n")
        f.write("\nCAVEAT: corruption is synthetic uniform relation substitution, a proxy for "
                "a real predictor's correlated errors; MetaQA is templated. Follow-up #1 "
                "(non-templated eval) is where a real predictor's error pattern is measured. "
                "The lean feature-set pruning (drop net-negative degree/incoming-one-hot) "
                "remains a follow-up.\n")
    print(f"wrote {out}")

    gate = next(b for (p, a, b) in rows if p == GATE_P)
    assert gate > GATE_FLOOR, f"reranker+match Hits@1 at p={GATE_P} = {gate:.4f} did not clear {GATE_FLOOR}"
    print("ACCEPTANCE PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the acceptance**

Run: `cd /d/repos/RGP/rgdb-eval && ../.venv/Scripts/python.exe scripts/experiment_reranker_schedule_api.py`
Expected: prints per-`p` schedule-alone vs reranker+match Hits@1, writes the results doc, `ACCEPTANCE PASSED` (reranker+match at p=0.25 clears 0.50; offline gate saw ~0.556). **Real gate:** if it does not clear the floor, do NOT lower it — STOP and report the numbers (compare to the offline gate's ~0.556 to see whether online training or the through-engine path lost ground).

Also confirm the eval suite still passes: `../.venv/Scripts/python.exe -m pytest tests -q` → green.

- [ ] **Step 3: Commit**

```bash
git add rgdb-eval/scripts/experiment_reranker_schedule_api.py rgdb-eval/results/metaqa-reranker-schedule-api.md
git commit -m "eval: query-conditioned reranker acceptance via the engine on degraded schedules"
```

---

## Notes for the executor

- **Do not merge or push.** Everything lands on `feature/rgdb-self-learning-loop`.
- **Do-no-harm is a hard gate:** Task 1 `cold_start_identity_holds_with_expected_relation` and every existing no-schedule reranker/engine test must stay green — the match feature must be inert without a schedule and at `w=0`.
- **Task 1 leaves the crate non-compiling until Task 2** (engine still calls the old reranker signatures). Verify Task 1 with `cargo test -p rgdb reranker`; the full crate compiles after Task 2.
- **Task 4 is a real acceptance gate.** A miss is a reportable result. Compare a miss to the offline gate's ~0.556 (p=0.25) to localize whether it's online-training or through-engine loss.
- **Ship the full+match feature set** (the lean pruning is a recorded follow-up; the results doc notes it). Do not drop the global features in this plan.
- Windows note: `cargo`/`pytest`/`maturin` use the repo `.venv` (Python 3.12); prefix `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` on a bare `cargo` if a Python-version error appears.
