import pytest

from assistlabel.core.ontology import load_ontology


def test_load_valid_mapping(tmp_path):
    p = tmp_path / "onto.yaml"
    p.write_text(
        "classes:\n"
        "  car: {prompt: 'car'}\n"
        "  pedestrian: {prompt: 'pedestrian walking'}\n",
        encoding="utf-8",
    )
    classes = load_ontology(p)
    assert [c.name for c in classes] == ["car", "pedestrian"]
    assert classes[0].prompt == "car"


def test_load_valid_shorthand(tmp_path):
    p = tmp_path / "onto.yaml"
    p.write_text("classes:\n  car: car\n", encoding="utf-8")
    classes = load_ontology(p)
    assert classes[0] .name == "car" and classes[0].prompt == "car"


def test_duplicate_class_rejected(tmp_path):
    p = tmp_path / "onto.yaml"
    p.write_text("classes:\n  car: car\n  car: truck\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_ontology(p)


def test_missing_prompt_rejected(tmp_path):
    p = tmp_path / "onto.yaml"
    p.write_text("classes:\n  car: {}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_ontology(p)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ontology(tmp_path / "nope.yaml")
