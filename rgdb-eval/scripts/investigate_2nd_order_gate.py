"""Measure-first gate for a 2nd-order transition matrix (feature #1).

A 2nd-order matrix T[(r_prev2, r_prev1)][r_next] can only beat the 1st-order marginal
T[r_prev][r_next] if knowing the FIRST relation narrows the THIRD beyond knowing the
second alone -- i.e. if H(r3 | r1, r2) is meaningfully below H(r3 | r2). This is a pure
information-theoretic check on MetaQA's gold 3-hop chains; if the gain is ~0, the
2nd-order model buys nothing and the (invasive) kernel change is not worth building.

Run from rgdb-eval/:  ../.venv/Scripts/python.exe scripts/investigate_2nd_order_gate.py

RESULT (2026-07-11): gain = 0.000 bits. #1 dropped. MetaQA 3-hop chains share a fixed
movie->person->movie skeleton, so r1 is determined by r2; the residual r3 ambiguity is
question-conditioned (only feature #3 / Option C can address it).
"""
from __future__ import annotations
import math
import os
from collections import Counter, defaultdict

from rgdb_eval.metaqa import qtype_to_relation_sequence

DATA = "data/MetaQA"


def entropy(counter: Counter) -> float:
    tot = sum(counter.values())
    if tot == 0:
        return 0.0
    return -sum(v / tot * math.log2(v / tot) for v in counter.values() if v > 0)


def main() -> None:
    chains = []
    with open(os.path.join(DATA, "qa_train_3hop_qtype.txt"), encoding="utf-8") as f:
        for line in f:
            seq = qtype_to_relation_sequence(line)
            if len(seq) == 3:
                chains.append(tuple(seq))
    print(f"{len(chains)} 3-hop training chains, {len(set(chains))} distinct\n")

    by_r2 = defaultdict(Counter)      # r2 -> distribution over r3   (1st-order context)
    by_r1r2 = defaultdict(Counter)    # (r1,r2) -> distribution over r3 (2nd-order)
    for r1, r2, r3 in chains:
        by_r2[r2][r3] += 1
        by_r1r2[(r1, r2)][r3] += 1

    n = len(chains)
    h1 = sum(sum(c.values()) / n * entropy(c) for c in by_r2.values())
    h2 = sum(sum(c.values()) / n * entropy(c) for c in by_r1r2.values())
    print(f"H(r3 | r2)     = {h1:.3f} bits   (1st-order: knows only the 2nd relation)")
    print(f"H(r3 | r1, r2) = {h2:.3f} bits   (2nd-order: knows the 1st too)")
    print(f"information gain from the extra relation = {h1 - h2:.3f} bits\n")

    verdict = "PASS -> spec #1" if (h1 - h2) > 0.10 else "FAIL -> drop #1"
    print(f"GATE ({'>0.10 bits'}): {verdict}")


if __name__ == "__main__":
    main()
