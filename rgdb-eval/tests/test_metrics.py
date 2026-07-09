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
