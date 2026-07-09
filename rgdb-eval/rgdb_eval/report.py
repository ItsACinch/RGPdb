"""Aggregate per-question metrics into per-hop means and a markdown table."""
from __future__ import annotations
from statistics import mean

from .metrics import hits_at_k, recall_at_k, mrr, K_VALUES


def evaluate(ranker, questions) -> dict:
    k_max = max(K_VALUES)

    # Rank in ORIGINAL question order so any per-question state in the ranker
    # (e.g. vector rankers' query-embedding pointer in the runner) stays
    # aligned with the questions list. Bucket by hop only for aggregation.
    per_q = []  # (hop, ranked_without_seeds, gold_set)
    for q in questions:
        seeds = [q.topic_id]
        seed_set = set(seeds)
        # Request extra so the post-strip list still has a true top-k_max.
        ranked = ranker.rank(seeds, q.relation, k_max + len(seeds))
        ranked = [n for n in ranked if n not in seed_set]  # uniform seed exclusion
        per_q.append((q.hop, ranked, set(q.answer_ids)))

    by_hop: dict[int, list] = {}
    for hop, ranked, gold in per_q:
        by_hop.setdefault(hop, []).append((ranked, gold))

    out: dict = {"ranker": ranker.name}
    for hop, items in sorted(by_hop.items()):
        stats = {}
        for k in K_VALUES:
            stats[f"hits@{k}"] = mean(hits_at_k(r, g, k) for r, g in items)
            stats[f"recall@{k}"] = mean(recall_at_k(r, g, k) for r, g in items)
        stats["mrr"] = mean(mrr(r, g) for r, g in items)
        stats["n"] = len(items)
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
