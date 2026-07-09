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
