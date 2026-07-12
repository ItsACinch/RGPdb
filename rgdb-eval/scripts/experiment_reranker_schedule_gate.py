"""Proposal #3 measure-first gate: does a QUERY-CONDITIONED match feature recover
reranker Hits@1 on DEGRADED (imperfect) schedules?

Background: `experiment_reranker.py` / the design spec documented that the online
reranker (rgdb/src/reranker.rs) regressed 3-hop Hits@1 (0.203 -> 0.1585 in the
acceptance-gate run) despite improving recall@20, because its features are GLOBAL
(query-agnostic): a single learned feature-preference vector shared across all
queries cannot supply the question-specific target relation that separates a correct
same-depth answer from a same-depth distractor.

Proposal #3 (this gate): add ONE query-conditioned feature to that same reranker
design -- "does this candidate's dominant incoming relation equal the query's
EXPECTED FINAL relation (`schedule[last]`)?" -- and measure whether that single bit
recovers Hits@1 specifically in the realistic regime where the schedule driving the
query (whether hand-built, predicted, or learned) is only PARTIALLY correct.

KEY SUBTLETY THIS SCRIPT MUST RESPECT: when the schedule is degraded/corrupted, the
"expected final relation" the match feature compares against is `schedule[last]` OF
THE SAME CORRUPTED SCHEDULE THAT DROVE PROPAGATION -- never the gold schedule. That
is: the match feature can only ever be as good as whatever relation-prediction
component produced `schedule[last]`. It helps only in the sub-case where hop 3 of
the (possibly wrong) schedule happens to survive corruption while earlier hops did
not -- i.e. it is a bet that "last-hop relation accuracy" is higher, or at least
different in its error pattern, than "did every hop match". If we compared against
the GOLD schedule instead, this would be cheating (leaking the answer's relation
type through the back door) and would not measure anything about deployability.

Method
------
For four corruption levels p in {0.0, 0.1, 0.25, 0.4} (probability each of the 3
relations in the gold 3-hop schedule is independently replaced by a uniformly random
*different* relation id, `numpy.random.default_rng(0)`):

  1. TRAIN (~1500 3-hop train questions, disjoint qa_train_3hop.txt file): corrupt
     the gold schedule, propagate with `core.propagate_layered(..., schedule=sched_c)`
     (schedule is the LAST positional/keyword arg), collapse per-depth to depth-3
     ("terminal(3)" semantics == `depth_weights=[0,0,0,1,0]` == `per_depth[3]`), take
     the top-20 candidates by that collapsed score (excluding the seed), build a
     feature row per candidate (5 per-depth values, log1p(score), log1p(out_degree),
     18-dim dominant-incoming-relation one-hot, and `match = 1.0 if
     dom[candidate]==sched_c[-1] else 0.0`), label 1 if the candidate is a gold
     answer. Pool all rows and fit two `StandardScaler`+`LogisticRegression`
     (`class_weight='balanced'`) models: WITH all features, WITHOUT the match column.
  2. TEST (~1500 3-hop test questions, qa_test_3hop.txt -- a physically separate file
     from train, so disjoint by construction): same corruption + propagation +
     candidate generation (independent RNG stream, `rng_test`, so test corruption
     draws never depend on how many train questions there were). Score three
     rankings of the SAME top-20 candidate set: schedule-alone (collapsed-score
     order), reranker-without-match, reranker-with-match. Hits@1 = is the #1-ranked
     candidate a gold answer. recall@20 is reported once per p (it is provably
     identical across all three rankings, since they all rerank the same fixed
     20-item set).

HONESTY NOTE: MetaQA is templated (15 fixed 3-hop qtypes), so its gold schedules are
clean and its corruption is purely synthetic label noise -- a real relation predictor
would make structured, correlated errors (e.g. confusing semantically similar
relations), not uniform-random substitution. This experiment isolates a narrower
question: GIVEN that some fraction of a schedule's hops are wrong, does the
`schedule[last]`-match feature carry signal the rest of the reranker's (global)
features don't already have. It is a proxy for predictor error, not a simulation of
one.

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/experiment_reranker_schedule_gate.py
"""
from __future__ import annotations
import math
import os
from statistics import mean

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from rgdb_embeddings import _rgdb_core as core

from rgdb_eval.metaqa import load_kb, parse_qa_line, qtype_to_relation_sequence
from rgdb_eval.metrics import hits_at_k, recall_at_k

DATA = "data/MetaQA"
MAX_DEPTH = 4
MIN_INTENSITY = 1e-4
FLOOR = 0.05
TOPK = 20
N_TRAIN = 1500
N_TEST = 1500
PS = [0.0, 0.1, 0.25, 0.4]
MARGIN = 0.02  # "meaningful margin" for the PASS verdict: >= 2 Hits@1 percentage points


# ---------------------------------------------------------------------------
# graph + trained relation-similarity vocab (co-occurrence over train qtypes,
# same construction as experiment_predicted_schedule.py / experiment_schedule_api.py)
# ---------------------------------------------------------------------------

def build_trained_vocab(graph):
    n = len(graph.relations)
    rid = graph.relation_to_id
    counts = np.zeros((n, n), dtype=np.float64)
    for hop in (1, 2, 3):
        p = os.path.join(DATA, f"qa_train_{hop}hop_qtype.txt")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                ids = [rid[r] for r in qtype_to_relation_sequence(line) if r in rid]
                for a, b in zip(ids, ids[1:]):
                    counts[a][b] += 1
    M = np.full((n, n), FLOOR, dtype=np.float32)
    for a in range(n):
        mx = counts[a].max()
        if mx > 0:
            M[a] = np.maximum(M[a], (counts[a] / mx).astype(np.float32))
    np.fill_diagonal(M, 1.0)
    return core.vocab_from_matrix(list(graph.relations), M.ravel().tolist())


# ---------------------------------------------------------------------------
# question rows with a fully-resolved GOLD 3-hop schedule
# ---------------------------------------------------------------------------

def load_3hop_rows(qa_path: str, qtype_path: str, graph, limit: int) -> list[dict]:
    rows = []
    with open(qa_path, encoding="utf-8") as fq, open(qtype_path, encoding="utf-8") as ft:
        for qline, tline in zip(fq, ft):
            if not qline.strip():
                continue
            topic, answers = parse_qa_line(qline)
            if topic not in graph.name_to_id:
                continue
            answer_ids = [graph.name_to_id[a] for a in answers if a in graph.name_to_id]
            if not answer_ids:
                continue
            seq = qtype_to_relation_sequence(tline)
            if len(seq) != 3:
                continue
            gold_sched = [graph.relation_to_id[r] for r in seq if r in graph.relation_to_id]
            if len(gold_sched) != 3:
                continue
            rows.append(dict(topic_id=graph.name_to_id[topic], answer_ids=answer_ids,
                              gold_sched=gold_sched))
            if len(rows) >= limit:
                break
    return rows


# ---------------------------------------------------------------------------
# corruption
# ---------------------------------------------------------------------------

def corrupt(sched: list[int], p: float, rng: np.random.Generator, n_relations: int) -> list[int]:
    """Each relation independently replaced, with probability p, by a uniformly
    random DIFFERENT relation id in [0, n_relations)."""
    out = []
    for r in sched:
        if rng.random() < p:
            choices = [x for x in range(n_relations) if x != r]
            out.append(int(rng.choice(choices)))
        else:
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# candidates + features
# ---------------------------------------------------------------------------

def top_candidates(g, vocab, topic_id: int, sched_c: list[int]):
    """-> (list[(cand_id, profile)] sorted desc by collapsed depth-3 score, top TOPK,
    excluding the seed; dominant-incoming-relation dict for the SAME propagation)."""
    per_depth, dominant = core.propagate_layered(
        g, vocab, [(topic_id, 1.0)], None,
        max_depth=MAX_DEPTH, min_intensity=MIN_INTENSITY, schedule=sched_c)
    per_map = dict(per_depth)
    per_map.pop(topic_id, None)
    dom_map = dict(dominant)
    ranked = sorted(per_map.items(), key=lambda kv: -kv[1][3])[:TOPK]
    return ranked, dom_map


def build_features(profile: list[float], dom_rel, outdeg_val: int,
                    sched_c: list[int], n_relations: int) -> list[float]:
    score = profile[3]  # terminal(3) collapsed score: depth_weights = [0,0,0,1,0]
    feats = list(profile)                       # 5 per-depth values (depth 0..4)
    feats.append(math.log1p(max(score, 0.0)))    # log1p(score)
    feats.append(math.log1p(outdeg_val))         # log1p(out_degree)
    onehot = [0.0] * n_relations
    if dom_rel is not None and 0 <= dom_rel < n_relations:
        onehot[dom_rel] = 1.0
    feats.extend(onehot)                          # 18-dim dominant-incoming one-hot
    match = 1.0 if dom_rel is not None and dom_rel == sched_c[-1] else 0.0
    feats.append(match)                            # LAST column: the match feature
    return feats


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    graph = load_kb(os.path.join(DATA, "kb.txt"))
    n_relations = len(graph.relations)
    print(f"graph: {graph.num_nodes} entities, {len(graph.edges)} edges, {n_relations} relations")

    adj: list[list] = [[] for _ in range(graph.num_nodes)]
    for (s, d, r) in graph.edges:
        adj[s].append((d, 0.0, r))
    outdeg = np.array([len(a) for a in adj], dtype=np.int64)
    g = core.build_graph(graph.num_nodes, adj)
    vocab = build_trained_vocab(graph)

    train_rows = load_3hop_rows(os.path.join(DATA, "qa_train_3hop.txt"),
                                 os.path.join(DATA, "qa_train_3hop_qtype.txt"),
                                 graph, N_TRAIN)
    test_rows = load_3hop_rows(os.path.join(DATA, "qa_test_3hop.txt"),
                                os.path.join(DATA, "qa_test_3hop_qtype.txt"),
                                graph, N_TEST)
    print(f"train rows: {len(train_rows)} (from qa_train_3hop.txt)")
    print(f"test rows:  {len(test_rows)} (from qa_test_3hop.txt -- disjoint file, no overlap with train)")

    # Two independent, fixed-seed RNG streams: one drives ALL train-set corruption
    # across the whole p sweep, the other drives ALL test-set corruption. Keeping
    # them separate means test corruption draws never depend on train-set size, and
    # each stream is reproducible end-to-end from numpy.random.default_rng(seed).
    rng_train = np.random.default_rng(0)
    rng_test = np.random.default_rng(0)

    match_idx = None  # set once feature width is known
    results: dict[float, dict] = {}

    for p in PS:
        print(f"\n=== p = {p:.2f} ===")

        # ---- training set ----
        X_rows: list[list[float]] = []
        y_rows: list[float] = []
        for row in train_rows:
            sched_c = corrupt(row["gold_sched"], p, rng_train, n_relations)
            ranked, dom_map = top_candidates(g, vocab, row["topic_id"], sched_c)
            gold = set(row["answer_ids"])
            for cand_id, profile in ranked:
                feats = build_features(profile, dom_map.get(cand_id),
                                        int(outdeg[cand_id]), sched_c, n_relations)
                X_rows.append(feats)
                y_rows.append(1.0 if cand_id in gold else 0.0)
        X_train = np.asarray(X_rows, dtype=np.float64)
        y_train = np.asarray(y_rows, dtype=np.float64)
        match_idx = X_train.shape[1] - 1
        n_pos = int(y_train.sum())
        print(f"  train pool: {X_train.shape[0]} candidate rows from {len(train_rows)} "
              f"questions, {n_pos} positive ({n_pos / len(y_train):.4f})")

        scaler = StandardScaler().fit(X_train)
        Xs_train = scaler.transform(X_train)
        Xs_train_without = np.delete(Xs_train, match_idx, axis=1)

        model_with = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
            Xs_train, y_train)
        model_without = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
            Xs_train_without, y_train)

        # ---- test set ----
        h1_sched, h1_without, h1_with, r20 = [], [], [], []
        for row in test_rows:
            sched_c = corrupt(row["gold_sched"], p, rng_test, n_relations)
            ranked, dom_map = top_candidates(g, vocab, row["topic_id"], sched_c)
            gold = set(row["answer_ids"])
            if not ranked:
                h1_sched.append(0.0); h1_without.append(0.0); h1_with.append(0.0); r20.append(0.0)
                continue
            cand_ids = [c for c, _ in ranked]
            feats = np.asarray(
                [build_features(profile, dom_map.get(c), int(outdeg[c]), sched_c, n_relations)
                 for c, profile in ranked], dtype=np.float64)
            Xs_test = scaler.transform(feats)
            Xs_test_without = np.delete(Xs_test, match_idx, axis=1)

            proba_with = model_with.predict_proba(Xs_test)[:, 1]
            proba_without = model_without.predict_proba(Xs_test_without)[:, 1]

            order_sched = cand_ids  # already sorted by collapsed score, desc
            order_without = [c for c, _ in sorted(zip(cand_ids, proba_without), key=lambda t: -t[1])]
            order_with = [c for c, _ in sorted(zip(cand_ids, proba_with), key=lambda t: -t[1])]

            h1_sched.append(hits_at_k(order_sched, gold, 1))
            h1_without.append(hits_at_k(order_without, gold, 1))
            h1_with.append(hits_at_k(order_with, gold, 1))
            # identical across the three rankings (same 20-item set); computed once
            r20.append(recall_at_k(order_sched, gold, 20))

        stats = dict(
            n=len(test_rows),
            sched_alone=mean(h1_sched),
            without=mean(h1_without),
            with_=mean(h1_with),
            recall20=mean(r20),
        )
        results[p] = stats
        print(f"  Hits@1  schedule-alone={stats['sched_alone']:.4f}  "
              f"reranker-no-match={stats['without']:.4f}  "
              f"reranker-with-match={stats['with_']:.4f}  "
              f"recall@20={stats['recall20']:.4f}")

    # -----------------------------------------------------------------------
    # verdict
    # -----------------------------------------------------------------------
    degraded_ps = [p for p in PS if p in (0.25, 0.4)]
    pass_ps = []
    for p in degraded_ps:
        s = results[p]
        beats_sched = s["with_"] - s["sched_alone"] >= MARGIN
        beats_without = s["with_"] - s["without"] >= MARGIN
        if beats_sched and beats_without:
            pass_ps.append(p)
    verdict = "PASS" if pass_ps else "FAIL/marginal"

    print("\n" + "=" * 88)
    print(f"{'p':>6}{'schedule-alone':>18}{'reranker-no-match':>20}{'reranker-with-match':>22}{'recall@20':>14}")
    for p in PS:
        s = results[p]
        print(f"{p:6.2f}{s['sched_alone']:18.4f}{s['without']:20.4f}{s['with_']:22.4f}{s['recall20']:14.4f}")
    print(f"\nGATE (margin >= {MARGIN:.2f} Hits@1 over BOTH schedule-alone and "
          f"reranker-no-match, at p in {{0.25, 0.4}}): {verdict}")
    if verdict == "PASS":
        print(f"  with-match cleared the margin at p in {pass_ps}")
    print("=" * 88)

    # -----------------------------------------------------------------------
    # write results md
    # -----------------------------------------------------------------------
    lines = []
    lines.append("# Proposal #3 gate: expected-final-relation match feature vs. degraded schedules (MetaQA 3-hop)\n")
    lines.append(
        "The online reranker (`rgdb/src/reranker.rs`, see `results/metaqa-reranker.md`) "
        "regressed 3-hop Hits@1 because its features are GLOBAL (query-agnostic): a "
        "single learned feature-preference vector cannot supply the question-specific "
        "target relation that separates a correct same-depth answer from a same-depth "
        "distractor. Proposal #3 adds ONE query-conditioned feature -- `match = "
        "1.0 if candidate's dominant incoming relation == schedule[last] else 0.0` -- "
        "and asks: does that single bit recover Hits@1 specifically in the realistic, "
        "DEGRADED regime where the schedule driving the query is only partially "
        "correct?\n"
    )
    lines.append(
        "**Key subtlety respected here:** when the schedule is corrupted, "
        "`schedule[last]` used by the match feature is read from the SAME corrupted "
        "schedule that drove propagation -- never the gold schedule. The match "
        "feature can only help when the FINAL hop's relation happens to survive "
        "corruption while earlier hops do not; comparing against the gold schedule "
        "instead would leak the answer's relation type and invalidate the gate.\n"
    )
    lines.append(
        "**Honesty caveat:** MetaQA is templated (15 fixed 3-hop qtypes), so its gold "
        "schedules are clean and corruption here is purely synthetic, uniform-random "
        "label noise -- a real relation predictor would make structured, correlated "
        "errors (e.g. confusing semantically similar relations), not uniform "
        "substitution. This experiment isolates a narrower question -- given that some "
        "fraction of a schedule's hops are wrong, does the `schedule[last]`-match "
        "feature carry signal the rest of the (global) reranker features don't already "
        "have -- as a proxy for predictor error, not a simulation of one.\n"
    )
    lines.append(f"Setup: trained transition-matrix vocab (co-occurrence over "
                 f"`qa_train_{{1,2,3}}hop_qtype.txt`, floor={FLOOR}), "
                 f"`max_depth={MAX_DEPTH}`, `min_intensity={MIN_INTENSITY}`, "
                 f"top-{TOPK} candidates by `propagate_layered(...).per_depth[3]` "
                 f"(terminal(3) collapse). Train: {len(train_rows)} 3-hop questions "
                 f"from `qa_train_3hop.txt`. Test: {len(test_rows)} 3-hop questions "
                 f"from `qa_test_3hop.txt` (disjoint file). Two independent "
                 f"`numpy.random.default_rng(0)` streams drive train- and test-set "
                 f"corruption respectively. Two `LogisticRegression(max_iter=1000, "
                 f"class_weight='balanced')` models fit per p on `StandardScaler`-scaled "
                 f"features pooled over all train candidates: WITH the match feature "
                 f"(28 features: 5 per-depth + log1p(score) + log1p(out_degree) + "
                 f"18-dim dominant-relation one-hot + match) and WITHOUT it "
                 f"(27 features).\n")
    lines.append("## Results\n")
    lines.append("| p | schedule-alone Hits@1 | reranker-no-match Hits@1 | reranker-with-match Hits@1 | recall@20 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for p in PS:
        s = results[p]
        lines.append(f"| {p:.2f} | {s['sched_alone']:.4f} | {s['without']:.4f} | "
                     f"{s['with_']:.4f} | {s['recall20']:.4f} |")
    lines.append("")
    lines.append(f"n = {len(test_rows)} test questions per row. recall@20 is identical "
                 f"across the three rankings by construction (they rerank the same "
                 f"fixed top-20 candidate set; only the ORDER differs), so it is "
                 f"reported once per p as the ceiling on achievable Hits@1.\n")
    lines.append("## Verdict\n")
    lines.append(f"PASS requires reranker-with-match to beat BOTH schedule-alone AND "
                 f"reranker-no-match by >= {MARGIN:.2f} Hits@1 at p in {{0.25, 0.4}} "
                 f"(the degraded, realistic regime).\n")
    for p in degraded_ps:
        s = results[p]
        d_sched = s["with_"] - s["sched_alone"]
        d_without = s["with_"] - s["without"]
        lines.append(f"- p={p:.2f}: with-match {s['with_']:.4f} vs schedule-alone "
                     f"{s['sched_alone']:.4f} (delta {d_sched:+.4f}) vs no-match "
                     f"{s['without']:.4f} (delta {d_without:+.4f})")
    lines.append(f"\n**GATE: {verdict}**\n")
    if verdict == "PASS":
        lines.append(
            f"The match feature recovers Hits@1 over both the schedule-alone ranking "
            f"and a reranker without it, at p in {pass_ps}, by a margin that survives "
            f"the honesty caveat above. Worth building the query-conditioned reranker "
            f"feature for real; the surviving open question is whether a real "
            f"relation predictor's error pattern (correlated, not uniform-random) "
            f"preserves this advantage.")
    else:
        lines.append(
            "The match feature did NOT clearly and robustly recover Hits@1 over both "
            "alternatives at the degraded corruption levels. See the numbers above for "
            "exactly where it falls short (ties, negative deltas, or margins under the "
            f"{MARGIN:.2f} threshold). Do not build the query-conditioned reranker "
            "extension on this evidence alone.")
    md = "\n".join(lines) + "\n"

    out = "results/metaqa-reranker-schedule-gate.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
