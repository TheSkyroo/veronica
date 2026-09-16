import json

from veronica import prefs


def test_load_missing_file_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(prefs, "_PREFS_PATH", tmp_path / "nonexistent" / "prefs.json")
    assert prefs.load() == {}


def test_save_then_load_roundtrips(monkeypatch, tmp_path):
    path = tmp_path / ".veronica" / "prefs.json"
    monkeypatch.setattr(prefs, "_PREFS_PATH", path)
    prefs.save({"hud_mode": "mini"})
    assert prefs.load() == {"hud_mode": "mini"}
    assert path.is_file()


def test_save_merges_over_existing_keys(monkeypatch, tmp_path):
    path = tmp_path / ".veronica" / "prefs.json"
    monkeypatch.setattr(prefs, "_PREFS_PATH", path)
    prefs.save({"hud_mode": "mini"})
    prefs.save({"hud_pos": [10.0, 20.0]})
    assert prefs.load() == {"hud_mode": "mini", "hud_pos": [10.0, 20.0]}


def test_save_overwrites_same_key(monkeypatch, tmp_path):
    path = tmp_path / ".veronica" / "prefs.json"
    monkeypatch.setattr(prefs, "_PREFS_PATH", path)
    prefs.save({"hud_mode": "mini"})
    prefs.save({"hud_mode": "full"})
    assert prefs.load() == {"hud_mode": "full"}


def test_load_corrupt_file_returns_empty(monkeypatch, tmp_path, caplog):
    path = tmp_path / ".veronica" / "prefs.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    monkeypatch.setattr(prefs, "_PREFS_PATH", path)
    assert prefs.load() == {}


def test_save_creates_parent_dirs(monkeypatch, tmp_path):
    path = tmp_path / "a" / "b" / "prefs.json"
    monkeypatch.setattr(prefs, "_PREFS_PATH", path)
    prefs.save({"hud_mode": "mini"})
    assert path.is_file()
    assert json.loads(path.read_text()) == {"hud_mode": "mini"}
