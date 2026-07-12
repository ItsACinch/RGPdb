"""Follow-up #2 measure-first gate: soft depth-weight PRIOR SHAPES vs. hard terminal(k),
specifically under hop-count (k) error.

This is a different question from `experiment_soft_depth_weights.py` (which tested a
LEARNED per-depth weight vector via the Engine's feedback loop and found it a dead end
because the 3-hop answer's mass concentrates at the same depth as distractors -- a
per-node problem, not a per-depth one). Here the weight vector is a fixed, hand-specified
SHAPE (terminal / geometric / triangular) peaked at a hop-count guess k_used, and the
question is purely about depth-control robustness: if a hop-count predictor is off by
one, does a soft (spread-out) shape degrade gracefully while the hard terminal(k) collapses
(since terminal(k) scores ONLY arrival depth k -- if k is wrong, it reads out zero mass at
the depth where the answer actually concentrates)?

Setup: MetaQA KB, trained transition-matrix vocab (relation-pair co-occurrence counts from
qa_train_{1,2,3}hop_qtype.txt, same construction as experiment_depth_weights_acceptance.py),
seeded with the gold first-hop relation. Evaluated on 3-hop TEST questions (~2000, fixed
seed slice), true k=3.

Weight shapes (length MAX_DEPTH+1=5, index d = arrival depth), peaked at k:
  terminal(k):      1.0 at d=k, else 0.0                         (today's hard readout)
  geometric(k, r):  w[d] = r**|d-k|  for r in {0.3, 0.5}          (exponential decay from k)
  triangular(k, W): w[d] = max(0, 1 - |d-k|/W), W=2               (linear decay from k)

Core comparison: for each shape, score 3-hop MRR/recall@20 with the peak at
k_used in {2 (wrong, -1), 3 (exact), 4 (wrong, +1)}. Verdict: does some soft shape (a)
match terminal(3) at k_used=3 (within ~0.01 MRR) and (b) clearly beat terminal at both
k_used=2 and k_used=4 (where terminal(2)/terminal(4) collapse because they read out zero
mass at the true answer depth)?

Second, low-risk piece: how accurate is a purely-textual hop-count predictor (1 vs 2 vs 3
hops), i.e. how often would k_used actually be off by one in the deployable path? Trains
an entity-masked TF-IDF + LogisticRegression classifier on ALL train questions across all
three hop counts, evaluates on the corresponding test sets.

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/experiment_soft_depth_prior_gate.py
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

from rgdb_eval.metaqa import load_kb, load_questions, qtype_to_relation_sequence
from rgdb_eval.metrics import hits_at_k, recall_at_k, mrr, K_VALUES

DATA = "data/MetaQA"
MAX_DEPTH = 4
MIN_INTENSITY = 1e-4
FLOOR = 0.05
EVAL_N = 2000  # 3-hop test questions scored per weight/k_used cell
TRUE_K = 3
K_USED_VALUES = (2, 3, 4)
ENT_RE = re.compile(r"\[.+?\]")


# ---------------------------------------------------------------------------
# trained transition matrix (same construction as experiment_depth_weights_acceptance.py)
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


# ---------------------------------------------------------------------------
# weight-vector builders: length MAX_DEPTH+1, index d = arrival depth
# ---------------------------------------------------------------------------

def terminal(k: int) -> list[float]:
    v = [0.0] * (MAX_DEPTH + 1)
    v[k] = 1.0
    return v


def geometric(k: int, r: float) -> list[float]:
    return [r ** abs(d - k) for d in range(MAX_DEPTH + 1)]


def triangular(k: int, w: float) -> list[float]:
    return [max(0.0, 1.0 - abs(d - k) / w) for d in range(MAX_DEPTH + 1)]


WEIGHT_SHAPES = [
    ("terminal", lambda k: terminal(k)),
    ("geometric r=0.3", lambda k: geometric(k, 0.3)),
    ("geometric r=0.5", lambda k: geometric(k, 0.5)),
    ("triangular W=2", lambda k: triangular(k, 2.0)),
]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score(g, vocab, graph, qs, weights, k_max=20) -> dict:
    items = []
    for q in qs:
        rel = graph.relation_to_id.get(q.relation) if q.relation else None
        totals = dict(core.propagate(g, vocab, [(q.topic_id, 1.0)], rel,
                                      MAX_DEPTH, MIN_INTENSITY, weights))
        totals.pop(q.topic_id, None)
        ranked = [nid for nid, _ in sorted(totals.items(), key=lambda kv: -kv[1])][:k_max]
        items.append((ranked, set(q.answer_ids)))
    return {
        "mrr": mean(mrr(r, gs) for r, gs in items),
        "recall@20": mean(recall_at_k(r, gs, 20) for r, gs in items),
        "hits@1": mean(hits_at_k(r, gs, 1) for r, gs in items),
    }


# ---------------------------------------------------------------------------
# hop-count predictor (question text -> 1/2/3 hops)
# ---------------------------------------------------------------------------

def mask_entity(text: str) -> str:
    return ENT_RE.sub(" ENT ", text)


def read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def train_hopcount_classifier():
    train_x, train_y = [], []
    test_x, test_y = [], []
    test_by_hop: dict[int, tuple[list[str], list[int]]] = {}
    for hop in (1, 2, 3):
        tr = read_lines(os.path.join(DATA, f"qa_train_{hop}hop.txt"))
        te = read_lines(os.path.join(DATA, f"qa_test_{hop}hop.txt"))
        tr_x = [mask_entity(ln.split("\t")[0]) for ln in tr if ln.strip()]
        te_x = [mask_entity(ln.split("\t")[0]) for ln in te if ln.strip()]
        train_x.extend(tr_x)
        train_y.extend([hop] * len(tr_x))
        test_x.extend(te_x)
        test_y.extend([hop] * len(te_x))
        test_by_hop[hop] = (te_x, [hop] * len(te_x))

    pipe = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
        ])),
        ("clf", LogisticRegression(max_iter=2000, C=10.0)),
    ])
    pipe.fit(train_x, train_y)

    overall_acc = pipe.score(test_x, test_y)
    per_hop_acc = {}
    for hop, (hx, hy) in test_by_hop.items():
        per_hop_acc[hop] = pipe.score(hx, hy)

    print(f"hop-count classifier: {len(train_x)} train / {len(test_x)} test questions "
          f"(1/2/3-hop pooled)")
    print(f"  overall accuracy: {overall_acc:.4f}")
    for hop in (1, 2, 3):
        print(f"  hop{hop} accuracy: {per_hop_acc[hop]:.4f} (n={len(test_by_hop[hop][0])})")
    return overall_acc, per_hop_acc


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    graph = load_kb(os.path.join(DATA, "kb.txt"))
    n = len(graph.relations)
    print(f"graph: {graph.num_nodes} entities, {len(graph.edges)} edges, {n} relations")

    adj: list[list] = [[] for _ in range(graph.num_nodes)]
    for (s, d, r) in graph.edges:
        adj[s].append((d, 0.0, r))
    g = core.build_graph(graph.num_nodes, adj)
    trained_M = build_trained_matrix(graph)
    vocab = core.vocab_from_matrix(list(graph.relations), trained_M.ravel().tolist())

    # ---- core comparison: 3-hop TEST questions, seeded with gold first-hop relation ----
    qs = load_questions(
        os.path.join(DATA, "qa_test_3hop.txt"), 3, graph, limit=EVAL_N,
        qtype_path=os.path.join(DATA, "qa_test_3hop_qtype.txt"),
    )
    print(f"\n3-hop test questions scored: {len(qs)} (true k = {TRUE_K})")

    mrr_table: dict[str, dict[int, float]] = {}
    rec_table: dict[str, dict[int, float]] = {}
    h1_table: dict[str, dict[int, float]] = {}
    for name, builder in WEIGHT_SHAPES:
        mrr_table[name] = {}
        rec_table[name] = {}
        h1_table[name] = {}
        for k_used in K_USED_VALUES:
            weights = builder(k_used)
            stats = score(g, vocab, graph, qs, weights)
            mrr_table[name][k_used] = stats["mrr"]
            rec_table[name][k_used] = stats["recall@20"]
            h1_table[name][k_used] = stats["hits@1"]
            print(f"  {name:18s} k_used={k_used}  weights={['%.3f' % w for w in weights]}  "
                  f"MRR={stats['mrr']:.4f}  recall@20={stats['recall@20']:.4f}")

    # ---- verdict ----
    EXACT_TOL = 0.01
    term_exact = mrr_table["terminal"][TRUE_K]
    term_k2 = mrr_table["terminal"][2]
    term_k4 = mrr_table["terminal"][4]

    soft_names = [name for name, _ in WEIGHT_SHAPES if name != "terminal"]

    def matches_exact(name: str) -> bool:
        return mrr_table[name][TRUE_K] >= term_exact - EXACT_TOL

    def beats_at_wrong_k(name: str, k_used: int, term_val: float) -> float:
        return mrr_table[name][k_used] - term_val

    # candidates that match terminal at exact k
    matching = [name for name in soft_names if matches_exact(name)]
    # among matching candidates, pick the one with the best combined margin at k=2,4
    best_name = None
    best_margin = -1e9
    for name in matching:
        margin = min(beats_at_wrong_k(name, 2, term_k2), beats_at_wrong_k(name, 4, term_k4))
        if margin > best_margin:
            best_margin = margin
            best_name = name

    pass_exact = best_name is not None
    pass_wrong_k = pass_exact and (
        mrr_table[best_name][2] > term_k2 and mrr_table[best_name][4] > term_k4
        and best_margin > EXACT_TOL  # "clearly" beats: more than the exact-k noise tolerance
    )
    verdict = "PASS" if (pass_exact and pass_wrong_k) else "FAIL/marginal"

    print("\n" + "=" * 78)
    print(f"GATE  exact k=3: terminal MRR={term_exact:.4f}; "
          + ", ".join(f"{name} MRR={mrr_table[name][TRUE_K]:.4f}" for name in soft_names))
    print(f"GATE  k_used=2 (wrong): terminal MRR={term_k2:.4f}; "
          + ", ".join(f"{name} MRR={mrr_table[name][2]:.4f}" for name in soft_names))
    print(f"GATE  k_used=4 (wrong): terminal MRR={term_k4:.4f}; "
          + ", ".join(f"{name} MRR={mrr_table[name][4]:.4f}" for name in soft_names))
    if best_name is not None:
        print(f"GATE  best matching-at-exact-k shape: {best_name}  "
              f"(margin over terminal at worst wrong-k = {best_margin:.4f})")
    else:
        print("GATE  no soft shape matched terminal(3) within tolerance at exact k")
    print(f"GATE  {verdict}")
    print("=" * 78)

    # ---- hop-count predictor ----
    print("\ntraining hop-count predictor (1 vs 2 vs 3 hops, all train questions) ...")
    hc_overall_acc, hc_per_hop_acc = train_hopcount_classifier()

    # ---- write results md ----
    lines = []
    lines.append("# Follow-up #2 measure-first gate: soft depth-weight prior shapes vs. hard `terminal(k)` (MetaQA 3-hop)\n")
    lines.append(
        "Question: does a SOFT (spread-out) depth-weight shape, peaked at a hop-count "
        "guess `k_used`, degrade more gracefully than the hard `terminal(k)` readout when "
        "the hop-count predictor is wrong by one? `terminal(k)` reads out only the mass "
        "arriving at exactly depth `k`; if `k` is wrong, it reads out (near-)zero at the "
        "depth where the true answer's mass actually concentrates. A soft shape spreads "
        "some weight to neighboring depths, so it should retain more signal when `k` is "
        "off, at the cost of (possibly) diluting the signal when `k` is exactly right.\n"
    )
    lines.append(
        "This is a different question from `experiment_soft_depth_weights.py` / "
        "`results/metaqa-soft-depth-weights.md`, which tested a *learned* per-depth "
        "weight vector (via the Engine's feedback loop) and found it a dead end -- the "
        "3-hop answer's mass concentrates at the same depth as distractors, which is a "
        "per-node problem no depth-only reweighting can fix. Here the weight vector's "
        "SHAPE is fixed and hand-specified; only its peak position `k_used` varies, "
        "simulating a hop-count predictor that is off by one.\n"
    )
    lines.append(
        f"Setup: MetaQA KB, trained transition-matrix vocab (relation-pair co-occurrence "
        f"from `qa_train_{{1,2,3}}hop_qtype.txt`), `core.build_graph`, `max_depth=4`, "
        f"`min_intensity=1e-4`. 3-hop TEST questions (n={len(qs)}), seeded with the gold "
        f"first-hop relation via `load_questions(..., qtype_path=...)`. True hop count "
        f"k=3. Rankings strip the seed node, top-20.\n"
    )
    lines.append("## Weight shapes\n")
    lines.append(
        "Length `MAX_DEPTH+1=5`, index `d` = arrival depth, peaked at `k`:\n\n"
        "- `terminal(k)`: 1.0 at `d=k`, else 0.0\n"
        "- `geometric(k, r)`: `w[d] = r**|d-k|`, `r` in {0.3, 0.5}\n"
        "- `triangular(k, W)`: `w[d] = max(0, 1 - |d-k|/W)`, `W=2`\n"
    )
    lines.append("## MRR by weight shape x k_used\n")
    lines.append("True k = 3. k_used=2 and k_used=4 simulate a hop-count predictor off by one.\n")
    lines.append("| shape | k_used=2 | k_used=3 (EXACT) | k_used=4 |")
    lines.append("|---|---:|---:|---:|")
    for name, _ in WEIGHT_SHAPES:
        lines.append(f"| {name} | {mrr_table[name][2]:.4f} | {mrr_table[name][3]:.4f} | {mrr_table[name][4]:.4f} |")
    lines.append("\n## recall@20 by weight shape x k_used\n")
    lines.append("| shape | k_used=2 | k_used=3 (EXACT) | k_used=4 |")
    lines.append("|---|---:|---:|---:|")
    for name, _ in WEIGHT_SHAPES:
        lines.append(f"| {name} | {rec_table[name][2]:.4f} | {rec_table[name][3]:.4f} | {rec_table[name][4]:.4f} |")
    lines.append("\n## Hits@1 by weight shape x k_used\n")
    lines.append("| shape | k_used=2 | k_used=3 (EXACT) | k_used=4 |")
    lines.append("|---|---:|---:|---:|")
    for name, _ in WEIGHT_SHAPES:
        lines.append(f"| {name} | {h1_table[name][2]:.4f} | {h1_table[name][3]:.4f} | {h1_table[name][4]:.4f} |")

    lines.append("\n## Verdict\n")
    lines.append(f"- At exact k=3, shapes matching `terminal(3)` (MRR {term_exact:.4f}) within {EXACT_TOL} MRR: "
                  + (", ".join(matching) if matching else "none") + ".")
    if best_name is not None:
        lines.append(
            f"- Winning shape: **{best_name}**. At exact k=3: MRR {mrr_table[best_name][3]:.4f} "
            f"vs terminal {term_exact:.4f} (delta {mrr_table[best_name][3]-term_exact:+.4f}). "
            f"At k_used=2 (wrong): MRR {mrr_table[best_name][2]:.4f} vs terminal {term_k2:.4f} "
            f"(delta {mrr_table[best_name][2]-term_k2:+.4f}). At k_used=4 (wrong): MRR "
            f"{mrr_table[best_name][4]:.4f} vs terminal {term_k4:.4f} (delta "
            f"{mrr_table[best_name][4]-term_k4:+.4f})."
        )
    else:
        lines.append("- No soft shape matched terminal(3) at exact k within tolerance.")
    lines.append(f"\n**GATE: {verdict}**\n")
    if verdict == "PASS":
        lines.append(
            f"`{best_name}` matches `terminal(3)` when k is right and clearly retains more "
            f"MRR than `terminal(k)` when k is off by one in either direction. `terminal(k)` "
            f"collapses at wrong k because it reads out zero mass at the true answer depth; "
            f"the soft shape's neighboring-depth weight recovers a meaningful fraction of "
            f"that signal. Worth carrying a soft depth prior into the deployable path IF the "
            f"hop-count predictor is not already near-perfect (see accuracy below)."
        )
    else:
        lines.append(
            "No soft shape both matched terminal(3) at exact k and clearly beat terminal at "
            "wrong k under the measured conditions. See the numbers above for exactly where "
            "it falls short."
        )

    lines.append("\n## Hop-count predictor accuracy (question text -> 1/2/3 hops)\n")
    lines.append(
        "Entity-masked TF-IDF (word 1-2gram + char_wb 3-5gram) + LogisticRegression, "
        "trained on ALL train questions across 1/2/3-hop, tested on the corresponding "
        "test sets. This tells us how often `k_used` would be exactly right vs. off by "
        "one in the deployable path (predicted schedule, no gold qtype).\n"
    )
    lines.append(f"**Overall accuracy: {hc_overall_acc:.4f}**\n")
    lines.append("| true hop count | accuracy |")
    lines.append("|---:|---:|")
    for hop in (1, 2, 3):
        lines.append(f"| {hop} | {hc_per_hop_acc[hop]:.4f} |")
    lines.append(
        "\n**Honesty note:** MetaQA questions are machine-templated from a small, fixed "
        "set of qtypes, and hop count strongly correlates with surface-level cues (answer "
        "type, question length, template phrasing). Text -> hop-count prediction is "
        "consequently a much easier problem here than in real, open-ended questions, "
        "where paraphrase and compositional structure make it harder to tell from "
        "phrasing alone how many hops are needed. Treat this accuracy as an optimistic "
        "upper bound: it tells us that on templated MetaQA the k-error case is rare, not "
        "that it will be rare in a real deployment. The k-error ROBUSTNESS results above "
        "(the weight-shape comparison at k_used=2/3/4) are the transferable evidence -- "
        "they hold regardless of how often k is actually wrong; this accuracy number only "
        "tells us how much that robustness would matter on THIS dataset.\n"
    )

    out = "results/metaqa-soft-depth-prior.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
