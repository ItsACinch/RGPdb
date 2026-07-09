from rgdb_eval import TypedGraph, Question


def test_typed_graph_indexes():
    g = TypedGraph(
        num_nodes=3,
        entity_names=["a", "b", "c"],
        relations=["r0", "r1"],
        edges=[(0, 1, 0), (0, 2, 1)],
    )
    assert g.name_to_id["b"] == 1
    assert g.relation_to_id["r1"] == 1
    assert sorted(g.out_neighbors(0)) == [(1, 0), (2, 1)]


def test_question_fields():
    q = Question(text="what?", topic_id=0, answer_ids=[2], relation="r1", hop=1)
    assert q.hop == 1 and q.answer_ids == [2]
