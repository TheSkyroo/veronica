import json
import os
import threading

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


def test_save_is_atomic_no_temp_left_and_replace_used(monkeypatch, tmp_path):
    path = tmp_path / "prefs.json"
    monkeypatch.setattr(prefs, "_PREFS_PATH", path)
    replaced = []
    real_replace = os.replace

    def spy(src, dst):
        replaced.append((os.path.basename(src), dst))
        real_replace(src, dst)

    monkeypatch.setattr(prefs.os, "replace", spy)
    prefs.save({"hud_mode": "mini"})
    assert replaced and str(replaced[0][1]) == str(path)
    assert replaced[0][0] != "prefs.json"          # written to a temp file first
    assert sorted(p.name for p in tmp_path.iterdir()) == ["prefs.json"]   # temp file gone
    assert json.loads(path.read_text()) == {"hud_mode": "mini"}


def test_save_failure_mid_write_leaves_old_file_intact(monkeypatch, tmp_path):
    path = tmp_path / "prefs.json"
    monkeypatch.setattr(prefs, "_PREFS_PATH", path)
    prefs.save({"hud_mode": "mini"})

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(prefs.os, "replace", boom)
    prefs.save({"hud_mode": "full"})    # logged, not raised
    assert json.loads(path.read_text()) == {"hud_mode": "mini"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["prefs.json"]   # temp cleaned up


def test_concurrent_saves_lose_nothing(monkeypatch, tmp_path):
    path = tmp_path / "prefs.json"
    monkeypatch.setattr(prefs, "_PREFS_PATH", path)
    n = 8
    start = threading.Barrier(n)

    def worker(i):
        start.wait()
        for j in range(20):
            prefs.save_settings_override(f"k{i}", j)
            prefs.save({f"p{i}": j})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    data = prefs.load()
    assert data["settings"] == {f"k{i}": 19 for i in range(n)}
    assert all(data[f"p{i}"] == 19 for i in range(n))
