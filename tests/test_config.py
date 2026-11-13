import pytest

from magsim.config import get_path, load_scenario, parse_path, set_path

YAML = """
name: demo
temperature_C: 20
magnets:
  - name: A
    pose: {position: [0.0, 0.0, 0.0]}
  - name: B
    pose: {position: [0.0, 0.0, 0.012]}
    material: {grade: N42, temperature_C: 80}
coil:
  ambient_temperature_C: [20, 25]
"""


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(YAML)
    return load_scenario(p)


def test_temperature_conversion(cfg):
    assert cfg["temperature"] == pytest.approx(293.15)
    assert "temperature_C" not in cfg
    assert cfg["magnets"][1]["material"]["temperature"] == pytest.approx(353.15)
    assert cfg["coil"]["ambient_temperature"] == pytest.approx([293.15, 298.15])


def test_reject_both_temperature_keys(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("magnet: {temperature_C: 20, temperature: 300}\n")
    with pytest.raises(ValueError, match="magnet.temperature_C"):
        load_scenario(p)


def test_get_path(cfg):
    assert get_path(cfg, "magnets[1].pose.position[2]") == 0.012
    assert get_path(cfg, "magnets[-1].name") == "B"
    assert parse_path("a.b[3].c") == ["a", "b", 3, "c"]


def test_get_path_errors(cfg):
    with pytest.raises(KeyError, match="magnets\\[1\\].pos.position"):
        get_path(cfg, "magnets[1].pos.position")
    with pytest.raises(IndexError, match="magnets\\[5\\]"):
        get_path(cfg, "magnets[5].name")
    with pytest.raises(KeyError):
        get_path(cfg, "name[0]")
    with pytest.raises(ValueError):
        parse_path("a..b")


def test_set_path_copies(cfg):
    new = set_path(cfg, "magnets[1].pose.position[2]", 0.02)
    assert get_path(new, "magnets[1].pose.position[2]") == 0.02
    assert get_path(cfg, "magnets[1].pose.position[2]") == 0.012
    new2 = set_path(cfg, "coil.turns", 5)
    assert new2["coil"]["turns"] == 5 and "turns" not in cfg["coil"]
    with pytest.raises(IndexError):
        set_path(cfg, "magnets[1].pose.position[3]", 1.0)
    with pytest.raises(KeyError):
        set_path(cfg, "missing.key", 1.0)
