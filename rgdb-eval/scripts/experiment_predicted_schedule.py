"""Feature #3 measure-first gate: PREDICTED per-hop schedule vs. the deployed baseline.

`docs/superpowers/specs/2026-07-10-deferred-query-conditioned-refraction.md` retired the
original gold-schedule gate (it FAILED as written, because it held depth control at
"none" -- the one setting where a relation model has negative slope) and replaced it with
a three-part gate. Step 1 (depth-aware scoring / `depth_weights`) has already landed and
is measured in `results/metaqa-depth-weights.md`: the deployed trained-matrix +
`terminal(3)` readout scores 3-hop MRR **0.381** on the full test set. That is the
baseline this script must beat.

The doc's remaining open steps for reviving C:

    2. A *predicted* schedule (from question text alone, no gold qtype) must beat 0.381
       under the SAME depth control -- not beat untyped-PPR's 0.279, and not rely on the
       gold chain (that ceiling -- 0.927 under a hard distance mask -- is already
       settled and is not what's being tested here).
    3. Graceful degradation: sweep schedule accuracy from perfect down to random and find
       where 3-hop MRR crosses back below 0.381. If the crossover sits at an accuracy
       higher than a real predictor can hit, C is too brittle to ship regardless of its
       ceiling.

This script runs both. It reuses the `__START__` synthetic-relation trick from
`experiment_gold_schedule.py` (a per-question (n+1)x(n+1) matrix where
`M[__START__][r1]=1`, `M[r_i][r_{i+1}]=1`, else `floor`; seed with
`query_relation=__START__`), but drives the per-hop schedule from a TfidfVectorizer +
LogisticRegression classifier over the (entity-masked) question text instead of from the
gold qtype, and applies `depth_weights=terminal(3)` to all three conditions so only the
schedule varies.

HONESTY NOTE (see also results/metaqa-predicted-schedule.md): MetaQA questions are
machine-templated from a fixed set of qtypes (15 for 3-hop), so text -> qtype prediction
is an easy, near-saturated classification problem here. The classifier accuracy below is
an optimistic upper bound relative to real, open-ended questions where paraphrase and
compositionality make schedule prediction much harder. The degradation sweep is what
actually speaks to robustness under imperfect prediction; read it, not the accuracy
number, as the deployability signal.

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/experiment_predicted_schedule.py
"""
from __future__ import annotations
import os
import re
from statistics import mean

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from rgdb_embeddings import _rgdb_core as core

from rgdb_eval.metaqa import (
    load_kb, parse_qa_line, query_relation_from_qtype, qtype_to_relation_sequence,
)
from rgdb_eval.metrics import hits_at_k, recall_at_k, mrr, K_VALUES

DATA = "data/MetaQA"
MAX_DEPTH = 4
MIN_INTENSITY = 1e-4
FLOOR = 0.05
TERMINAL3 = [0.0, 0.0, 0.0, 1.0, 0.0]  # depth_weights, index 0..MAX_DEPTH; deployed readout
SWEEP_N = 2000
SWEEP_PS = [0.0, 0.1, 0.25, 0.5, 1.0]
BASELINE_REFERENCE = 0.381  # deployed trained-matrix + terminal(3), full test set (metaqa-depth-weights.md)

ENT_RE = re.compile(r"\[.+?\]")


def mask_entity(text: str) -> str:
    """Replace the bracketed topic-entity span with a constant token, so the
    classifier learns the question TEMPLATE, not the entity identity."""
    return ENT_RE.sub(" ENT ", text)


def read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


# ---------------------------------------------------------------------------
# 1. question-text -> qtype classifier
# ---------------------------------------------------------------------------

def train_classifier():
    train_lines = read_lines(os.path.join(DATA, "qa_train_3hop.txt"))
    train_y = read_lines(os.path.join(DATA, "qa_train_3hop_qtype.txt"))
    test_lines = read_lines(os.path.join(DATA, "qa_test_3hop.txt"))
    test_y = read_lines(os.path.join(DATA, "qa_test_3hop_qtype.txt"))

    train_x = [mask_entity(ln.split("\t")[0]) for ln in train_lines]
    test_x = [mask_entity(ln.split("\t")[0]) for ln in test_lines]

    pipe = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
        ])),
        ("clf", LogisticRegression(max_iter=2000, C=10.0)),
    ])
    pipe.fit(train_x, train_y)
    pred_all = list(pipe.predict(test_x))
    acc = mean(1.0 if p == y else 0.0 for p, y in zip(pred_all, test_y))
    print(f"classifier: {len(train_x)} train / {len(test_x)} test questions, "
          f"{len(set(train_y))} qtype classes")
    print(f"classifier qtype accuracy on qa_test_3hop: {acc:.4f}")
    return acc, pred_all  # pred_all is line-aligned with qa_test_3hop.txt / its qtype file


# ---------------------------------------------------------------------------
# trained transition matrix (the deployed no-schedule baseline's relation model)
# ---------------------------------------------------------------------------

def build_trained_matrix(graph) -> np.ndarray:
    n = len(graph.relations)
    rid = graph.relation_to_id
    counts = np.zeros((n, n), dtype=np.float64)
    for hop in (1, 2, 3):
        p = os.path.join(DATA, f"qa_train_{hop}hop_qtype.txt")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                ids = [rid[r] for r in qtype_to_relation_sequence(line) if r in rid]
                for a, b in zip(ids, ids[1:]):
                    counts[a][b] += 1
    M = np.full((n, n), FLOOR, dtype=np.float32)
    for a in range(n):
        mx = counts[a].max()
        if mx > 0:
            M[a] = np.maximum(M[a], (counts[a] / mx).astype(np.float32))
    np.fill_diagonal(M, 1.0)
    return M


def schedule_matrix(n_real: int, sched: list[int], floor: float) -> list[float]:
    """(n_real+1)^2 row-major matrix implementing a per-hop expected-relation schedule.
    Relation id `n_real` is the synthetic __START__ (query-relation only)."""
    n = n_real + 1
    start = n_real
    m = [floor] * (n * n)
    m[start * n + sched[0]] = 1.0
    for a, b in zip(sched, sched[1:]):
        m[a * n + b] = 1.0
    return m


# ---------------------------------------------------------------------------
# question rows
# ---------------------------------------------------------------------------

def load_test_rows(graph, pred_qtypes: list[str]) -> list[dict]:
    """Line-aligned with qa_test_3hop.txt / its qtype file, so `pred_qtypes[i]`
    (the classifier's prediction for line i) attaches correctly even though some
    lines get filtered out below."""
    qpath = os.path.join(DATA, "qa_test_3hop.txt")
    tpath = os.path.join(DATA, "qa_test_3hop_qtype.txt")
    qtypes = read_lines(tpath)
    rows = []
    with open(qpath, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            topic, answers = parse_qa_line(line)
            if topic not in graph.name_to_id:
                continue
            answer_ids = [graph.name_to_id[a] for a in answers if a in graph.name_to_id]
            if not answer_ids:
                continue
            qt = qtypes[i] if i < len(qtypes) else ""
            pqt = pred_qtypes[i] if i < len(pred_qtypes) else ""
            first = query_relation_from_qtype(qt)
            first_rel = graph.relation_to_id.get(first) if first else None
            gold_sched = [graph.relation_to_id[r] for r in qtype_to_relation_sequence(qt)
                          if r in graph.relation_to_id]
            pred_sched = [graph.relation_to_id[r] for r in qtype_to_relation_sequence(pqt)
                          if r in graph.relation_to_id]
            rows.append(dict(idx=i, topic_id=graph.name_to_id[topic], answer_ids=answer_ids,
                              qtype=qt, pred_qtype=pqt, first_rel=first_rel,
                              gold_sched=gold_sched, pred_sched=pred_sched))
    return rows


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_rows(g, rows, get_vocab_and_rel, depth_weights, k_max=20) -> list[tuple[list[int], set[int]]]:
    items = []
    for row in rows:
        vocab, rel_id = get_vocab_and_rel(row)
        gold = set(row["answer_ids"])
        if vocab is None or rel_id is None:
            items.append(([], gold))
            continue
        totals = dict(core.propagate(g, vocab, [(row["topic_id"], 1.0)], rel_id,
                                      MAX_DEPTH, MIN_INTENSITY, depth_weights))
        totals.pop(row["topic_id"], None)
        ranked = [nid for nid, _ in sorted(totals.items(), key=lambda kv: -kv[1])][:k_max]
        items.append((ranked, gold))
    return items


def summarize(items) -> dict:
    s = {"n": len(items)}
    for k in K_VALUES:
        s[f"hits@{k}"] = mean(hits_at_k(r, gs, k) for r, gs in items)
        s[f"recall@{k}"] = mean(recall_at_k(r, gs, k) for r, gs in items)
    s["mrr"] = mean(mrr(r, gs) for r, gs in items)
    return s


def main() -> None:
    graph = load_kb(os.path.join(DATA, "kb.txt"))
    n_real = len(graph.relations)
    start_id = n_real
    names_plus = list(graph.relations) + ["__START__"]
    print(f"graph: {graph.num_nodes} entities, {len(graph.edges)} edges, {n_real} relations")

    adj: list[list] = [[] for _ in range(graph.num_nodes)]
    for (s, d, r) in graph.edges:
        adj[s].append((d, 0.0, r))
    g = core.build_graph(graph.num_nodes, adj)

    # ---- 1. classifier ----
    acc, pred_all = train_classifier()

    # ---- rows (line-aligned with the predicted-qtype array) ----
    rows = load_test_rows(graph, pred_all)
    print(f"3-hop test rows: {len(rows)} (of {len(pred_all)} lines)")
    n_gold_resolved = sum(1 for r in rows if len(r["gold_sched"]) == 3)
    n_pred_resolved = sum(1 for r in rows if len(r["pred_sched"]) == 3)
    n_pred_correct = sum(1 for r in rows if r["pred_qtype"] == r["qtype"])
    print(f"rows with resolved gold schedule: {n_gold_resolved}, "
          f"resolved predicted schedule: {n_pred_resolved}, "
          f"predicted qtype == gold qtype: {n_pred_correct} ({n_pred_correct/len(rows):.4f})")

    trained_M = build_trained_matrix(graph)
    trained_vocab = core.vocab_from_matrix(list(graph.relations), trained_M.ravel().tolist())

    # ---- 2. three conditions on the FULL 3-hop test set ----
    def baseline_vr(row):
        return (trained_vocab, row["first_rel"])

    def gold_vr(row):
        if len(row["gold_sched"]) != 3:
            return (None, None)
        vocab = core.vocab_from_matrix(names_plus, schedule_matrix(n_real, row["gold_sched"], FLOOR))
        return (vocab, start_id)

    def pred_vr(row):
        if len(row["pred_sched"]) != 3:
            return (None, None)
        vocab = core.vocab_from_matrix(names_plus, schedule_matrix(n_real, row["pred_sched"], FLOOR))
        return (vocab, start_id)

    print("\nscoring baseline (trained matrix, gold first-hop relation, terminal(3)) ...")
    baseline_items = score_rows(g, rows, baseline_vr, TERMINAL3)
    baseline_stats = summarize(baseline_items)

    print("scoring gold-schedule (__START__ matrix from GOLD qtype, terminal(3)) ...")
    gold_items = score_rows(g, rows, gold_vr, TERMINAL3)
    gold_stats = summarize(gold_items)

    print("scoring predicted-schedule (__START__ matrix from PREDICTED qtype, terminal(3)) ...")
    pred_items = score_rows(g, rows, pred_vr, TERMINAL3)
    pred_stats = summarize(pred_items)

    print("\n3-hop results (full test set, n = {}):".format(len(rows)))
    print(f"{'condition':30s}{'MRR':>8s}{'Hits@1':>8s}{'recall@20':>12s}")
    for name, s in (("baseline (no schedule)", baseline_stats),
                    ("gold-schedule", gold_stats),
                    ("predicted-schedule", pred_stats)):
        print(f"{name:30s}{s['mrr']:8.4f}{s['hits@1']:8.4f}{s['recall@20']:12.4f}")

    # ---- 3. graceful-degradation sweep (gold schedule corrupted at rate p) ----
    sweep_rows = [r for r in rows if len(r["gold_sched"]) == 3][:SWEEP_N]
    print(f"\ndegradation sweep on {len(sweep_rows)} questions (fixed subset, gold schedule resolved)")

    rng = np.random.default_rng(0)

    def corrupt(sched: list[int], p: float) -> list[int]:
        out = []
        for r in sched:
            if p > 0.0 and rng.random() < p:
                choices = [x for x in range(n_real) if x != r]
                out.append(int(rng.choice(choices)))
            else:
                out.append(r)
        return out

    def corrupted_vr_factory(p: float):
        def vr(row):
            sched = corrupt(row["gold_sched"], p)
            vocab = core.vocab_from_matrix(names_plus, schedule_matrix(n_real, sched, FLOOR))
            return (vocab, start_id)
        return vr

    sweep_results = []
    for p in SWEEP_PS:
        items = score_rows(g, sweep_rows, corrupted_vr_factory(p), TERMINAL3)
        stats = summarize(items)
        sweep_results.append((p, stats["mrr"]))
        print(f"  p={p:4.2f}  3-hop MRR = {stats['mrr']:.4f}")

    # baseline MRR on the SAME sweep subset, for a fair crossover comparison
    baseline_sweep_items = score_rows(g, sweep_rows, baseline_vr, TERMINAL3)
    baseline_sweep_mrr = summarize(baseline_sweep_items)["mrr"]
    print(f"  baseline (trained matrix, terminal(3)) on same subset: MRR = {baseline_sweep_mrr:.4f}")

    crossover_p = None
    for p, m in sweep_results:
        if m < baseline_sweep_mrr:
            crossover_p = p
            break

    # ---- verdict ----
    beats_baseline = pred_stats["mrr"] > baseline_stats["mrr"]
    degrades_gracefully = crossover_p is None or crossover_p >= 0.5
    verdict = "PASS" if (beats_baseline and degrades_gracefully) else "FAIL/marginal"

    print("\n" + "=" * 72)
    print(f"GATE  classifier qtype accuracy: {acc:.4f}")
    print(f"GATE  3-hop MRR: baseline {baseline_stats['mrr']:.4f}  "
          f"predicted-schedule {pred_stats['mrr']:.4f}  gold-schedule {gold_stats['mrr']:.4f}")
    print(f"GATE  degradation crossover (subset baseline {baseline_sweep_mrr:.4f}): "
          f"{'p=' + str(crossover_p) if crossover_p is not None else 'never crosses in {0.0..1.0}'}")
    print(f"GATE  {verdict}")
    print("=" * 72)

    # ---- write results md ----
    lines = []
    lines.append("# Feature #3 measure-first gate: predicted schedule vs. baseline (MetaQA 3-hop)\n")
    lines.append(
        "Per the replacement gate in "
        "`docs/superpowers/specs/2026-07-10-deferred-query-conditioned-refraction.md`: "
        "depth-aware scoring (`depth_weights`) has already landed and is held constant "
        "here (`terminal(3)` on all three conditions). The open question is whether a "
        "schedule **predicted from question text** (no gold qtype) beats the deployed "
        "no-schedule baseline, and how gracefully it degrades as prediction accuracy "
        "drops.\n"
    )
    lines.append(
        "**Honesty note:** MetaQA questions are machine-templated (only 15 distinct "
        "3-hop qtypes), so text -> qtype prediction is an EASY, near-saturated problem "
        "here. The classifier accuracy below is an optimistic upper bound relative to "
        "real, open-ended questions, where paraphrase and compositional phrasing make "
        "schedule prediction substantially harder. The degradation sweep -- not the "
        "accuracy number -- is what actually speaks to deployability, since it measures "
        "what happens as prediction quality falls short of this optimistic ceiling.\n"
    )
    lines.append("## 1. Classifier\n")
    lines.append(
        f"TF-IDF (word 1-2gram + char_wb 3-5gram) + LogisticRegression, entity span "
        f"masked to a constant ` ENT ` token before featurizing. Trained on "
        f"`qa_train_3hop.txt` ({len(read_lines(os.path.join(DATA, 'qa_train_3hop.txt')))} "
        f"questions, {len(set(read_lines(os.path.join(DATA, 'qa_train_3hop_qtype.txt'))))} "
        f"qtype classes), evaluated on `qa_test_3hop.txt`.\n"
    )
    lines.append(f"**qtype accuracy: {acc:.4f}** (n={len(read_lines(os.path.join(DATA, 'qa_test_3hop_qtype.txt')))})\n")
    lines.append("## 2. Three conditions, full 3-hop test set\n")
    lines.append(
        f"All three use `max_depth=4`, `min_intensity=1e-4`, `depth_weights=terminal(3)` "
        f"= `[0,0,0,1,0]`. n = {len(rows)}.\n"
    )
    lines.append("| condition | MRR | Hits@1 | recall@20 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| baseline (trained matrix, gold 1st-hop relation) | {baseline_stats['mrr']:.4f} | "
                  f"{baseline_stats['hits@1']:.4f} | {baseline_stats['recall@20']:.4f} |")
    lines.append(f"| gold-schedule (`__START__` matrix, GOLD qtype) | {gold_stats['mrr']:.4f} | "
                  f"{gold_stats['hits@1']:.4f} | {gold_stats['recall@20']:.4f} |")
    lines.append(f"| predicted-schedule (`__START__` matrix, PREDICTED qtype) | {pred_stats['mrr']:.4f} | "
                  f"{pred_stats['hits@1']:.4f} | {pred_stats['recall@20']:.4f} |")
    lines.append(
        f"\nReference: `results/metaqa-depth-weights.md` full-set baseline MRR = "
        f"{BASELINE_REFERENCE:.3f} (this run's baseline: {baseline_stats['mrr']:.4f}).\n"
    )
    lines.append("## 3. Graceful-degradation sweep\n")
    lines.append(
        f"Starting from the GOLD schedule, each of the 3 relations is independently "
        f"replaced with a uniformly random *different* relation with probability p "
        f"(`numpy.random.default_rng(0)`). Scored on a fixed {len(sweep_rows)}-question "
        f"subset (all questions with a resolved gold schedule, first {SWEEP_N} in file "
        f"order) for runtime; same subset used for every p and for the baseline "
        f"reference line below.\n"
    )
    lines.append("| p (per-relation corruption) | 3-hop MRR |")
    lines.append("|---:|---:|")
    for p, m in sweep_results:
        lines.append(f"| {p:.2f} | {m:.4f} |")
    lines.append(f"| — baseline (same subset) | {baseline_sweep_mrr:.4f} |")
    lines.append(
        f"\nCrossover: MRR first drops below the same-subset baseline "
        f"({baseline_sweep_mrr:.4f}) at "
        f"{'p = ' + str(crossover_p) if crossover_p is not None else 'no p in the sweep (stays above baseline through p=1.0)'}."
        f" The classifier's actual error rate on this test set is {1 - acc:.4f} "
        f"({n_pred_correct}/{len(rows)} correct); this is not directly the same "
        f"quantity as per-relation corruption probability p (a wrong qtype prediction "
        f"can differ from gold in one, two, or three of the three relations), but it "
        f"gives the order of magnitude of real prediction error to compare against the "
        f"crossover.\n"
    )
    lines.append("## Verdict\n")
    lines.append(f"- classifier qtype accuracy: **{acc:.4f}**")
    lines.append(f"- 3-hop MRR: baseline **{baseline_stats['mrr']:.4f}**, "
                 f"predicted-schedule **{pred_stats['mrr']:.4f}**, "
                 f"gold-schedule **{gold_stats['mrr']:.4f}**")
    lines.append(f"- degradation crossover: "
                 f"{'p = ' + str(crossover_p) if crossover_p is not None else 'none observed in [0,1]'}")
    lines.append(f"\n**GATE: {verdict}**\n")
    if verdict == "PASS":
        lines.append(
            "Predicted-schedule 3-hop MRR clearly beats the deployed baseline, and the "
            "degradation sweep shows it stays above baseline down to a corruption rate "
            "at or below what the classifier's real error rate implies. #3 is worth "
            "designing for real (i.e. non-templated) deployments IF a comparably "
            "accurate predictor can be built there -- which this experiment, run on "
            "templated MetaQA text, cannot establish on its own (see honesty note above)."
        )
    else:
        lines.append(
            "Predicted-schedule did not clearly and robustly beat the deployed baseline "
            "under the measured conditions. See the numbers above for exactly where it "
            "falls short (ceiling, degradation slope, or both)."
        )
    md = "\n".join(lines) + "\n"

    out = "results/metaqa-predicted-schedule.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
