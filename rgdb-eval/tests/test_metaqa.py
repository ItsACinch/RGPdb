from rgdb_eval.metaqa import (
    parse_kb_line, parse_qa_line, load_kb_from_lines, query_relation_from_qtype,
)


def test_query_relation_from_qtype():
    # 1-hop: forward and inverse directions
    assert query_relation_from_qtype("movie_to_director") == "directed_by"
    assert query_relation_from_qtype("actor_to_movie") == "starred_actors_inv"
    # multi-hop: first pair drives the first-hop relation
    assert query_relation_from_qtype("movie_to_actor_to_movie_to_director") == "starred_actors"
    # unresolvable
    assert query_relation_from_qtype("nonsense") is None


def test_parse_kb_line():
    assert parse_kb_line("Blade Runner|directed_by|Ridley Scott") == (
        "Blade Runner", "directed_by", "Ridley Scott"
    )


def test_parse_qa_line():
    line = "what films did [Ridley Scott] direct\tBlade Runner|Alien"
    topic, answers = parse_qa_line(line)
    assert topic == "Ridley Scott"
    assert answers == ["Blade Runner", "Alien"]


def test_load_kb_from_lines_builds_typed_graph():
    lines = [
        "Blade Runner|directed_by|Ridley Scott",
        "Alien|directed_by|Ridley Scott",
    ]
    # default add_inverse=True: each triple yields a forward + an inverse edge
    g = load_kb_from_lines(lines)
    assert g.num_nodes == 3          # 2 movies + 1 director (inverse edges add no nodes)
    assert "directed_by" in g.relation_to_id
    assert "directed_by_inv" in g.relation_to_id  # inverse relation added
    assert len(g.edges) == 4         # 2 forward + 2 inverse

    # add_inverse=False restores forward-only loading
    g_fwd = load_kb_from_lines(lines, add_inverse=False)
    assert len(g_fwd.edges) == 2
    assert "directed_by_inv" not in g_fwd.relation_to_id
