"""Exercise the learning loop end-to-end through the native bindings."""
from rgdb_embeddings import _rgdb_core as core


def main() -> None:
    # 0 -(A=0)-> 1 -(B=1)-> 2
    g = core.build_graph(3, [[(1, 0.0, 0)], [(2, 0.0, 1)], []])
    vocab = core.vocab_from_matrix(["A", "B"], [1.0, 1.0, 1.0, 1.0])  # uniform prior
    eng = core.Engine(g, vocab, rebuild_every_n=0)  # manual refresh

    # cold start: matrix must be exactly all-ones (typed PPR, do no harm)
    assert eng.matrix() == [1.0, 1.0, 1.0, 1.0], eng.matrix()

    ranked, qid = eng.query([(0, 1.0)], 0, 4, 0.0)
    assert ranked and qid > 0, (ranked, qid)

    eng.record_feedback(qid, 2, 100.0)
    eng.refresh()

    m = eng.matrix()
    assert m[0] == 1.0, "diagonal pinned"
    assert m[3] == 1.0, "diagonal pinned"

    # unknown query id must raise, never silently drop
    try:
        eng.record_feedback(999999, 2, 1.0)
        raise AssertionError("expected ValueError for unknown query id")
    except ValueError:
        pass

    print("engine smoke ok; matrix:", [round(x, 4) for x in m])


if __name__ == "__main__":
    main()
