from rgdb_eval.metaqa import parse_kb_line, parse_qa_line, load_kb_from_lines


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
    g = load_kb_from_lines(lines)
    assert g.num_nodes == 3          # 2 movies + 1 director
    assert "directed_by" in g.relation_to_id
    assert len(g.edges) == 2
