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


def test_get_returns_default_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(prefs, "_PREFS_PATH", tmp_path / "prefs.json")
    assert prefs.get("hud_mode", "full") == "full"


def test_get_returns_value_when_present(monkeypatch, tmp_path):
    path = tmp_path / "prefs.json"
    monkeypatch.setattr(prefs, "_PREFS_PATH", path)
    prefs.save({"hud_mode": "mini"})
    assert prefs.get("hud_mode", "full") == "mini"


def test_save_settings_override_merges_under_settings_key(monkeypatch, tmp_path):
    path = tmp_path / "prefs.json"
    monkeypatch.setattr(prefs, "_PREFS_PATH", path)
    prefs.save_settings_override("effort", "high")
    assert prefs.load() == {"settings": {"effort": "high"}}
    prefs.save_settings_override("memory_enabled", False)
    assert prefs.load() == {"settings": {"effort": "high", "memory_enabled": False}}


def test_save_settings_override_does_not_clobber_other_prefs(monkeypatch, tmp_path):
    path = tmp_path / "prefs.json"
    monkeypatch.setattr(prefs, "_PREFS_PATH", path)
    prefs.save({"hud_mode": "mini"})
    prefs.save_settings_override("effort", "high")
    assert prefs.load() == {"hud_mode": "mini", "settings": {"effort": "high"}}


def test_clear_settings_override_removes_only_that_key(monkeypatch, tmp_path):
    path = tmp_path / "prefs.json"
    monkeypatch.setattr(prefs, "_PREFS_PATH", path)
    prefs.save_settings_override("effort", "high")
    prefs.save_settings_override("memory_enabled", False)
    prefs.clear_settings_override("effort")
    assert prefs.load() == {"settings": {"memory_enabled": False}}


def test_clear_settings_override_on_missing_key_is_noop(monkeypatch, tmp_path):
    path = tmp_path / "prefs.json"
    monkeypatch.setattr(prefs, "_PREFS_PATH", path)
    prefs.clear_settings_override("effort")
    assert prefs.load() == {}
