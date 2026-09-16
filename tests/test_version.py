import json
from pathlib import Path

from tests.fakes import FakeRun
from veronica import version

GIT_OK = {
    "git rev-parse --short HEAD": (0, "a517483\n", ""),
    "git log -1 --format=%cI": (0, "2026-09-17T00:00:48+05:30\n", ""),
    "git status --porcelain": (0, "", ""),
}


def test_app_version_is_a_string():
    assert isinstance(version.APP_VERSION, str)
    assert version.APP_VERSION == "0.1.0"


def test_build_info_reads_bundle_file_without_touching_git(tmp_path):
    build_json = tmp_path / "build.json"
    build_json.write_text(json.dumps({"sha": "deadbee", "built_at": "2026-09-16T10:00:00+05:30", "dirty": True}))
    run = FakeRun(GIT_OK)
    info = version.build_info(run=run, env={"VERONICA_BUNDLE_BUILD": str(build_json)}, repo=tmp_path)
    assert info == {"sha": "deadbee", "built_at": "2026-09-16T10:00:00+05:30", "dirty": True, "source": "bundle"}
    assert run.calls == []


def test_build_info_falls_back_to_git_when_bundle_file_missing(tmp_path):
    run = FakeRun(GIT_OK)
    info = version.build_info(run=run, env={"VERONICA_BUNDLE_BUILD": str(tmp_path / "nope.json")}, repo=tmp_path)
    assert info["source"] == "git"
    assert info["sha"] == "a517483"


def test_build_info_falls_back_to_git_when_bundle_file_invalid(tmp_path):
    bad = tmp_path / "build.json"
    bad.write_text("{not json")
    run = FakeRun(GIT_OK)
    info = version.build_info(run=run, env={"VERONICA_BUNDLE_BUILD": str(bad)}, repo=tmp_path)
    assert info["source"] == "git"


def test_build_info_from_git_clean(tmp_path):
    run = FakeRun(GIT_OK)
    info = version.build_info(run=run, env={}, repo=tmp_path)
    assert info == {"sha": "a517483", "built_at": "2026-09-17T00:00:48+05:30", "dirty": False, "source": "git"}
    assert run.argv_strings == [
        "git rev-parse --short HEAD",
        "git log -1 --format=%cI",
        "git status --porcelain",
    ]
    assert all(kw.get("cwd") == tmp_path for kw in run.kwargs)


def test_build_info_from_git_dirty(tmp_path):
    run = FakeRun({**GIT_OK, "git status --porcelain": (0, " M veronica/version.py\n", "")})
    info = version.build_info(run=run, env={}, repo=tmp_path)
    assert info["dirty"] is True


def test_build_info_unknown_when_git_fails(tmp_path):
    run = FakeRun({**GIT_OK, "git rev-parse --short HEAD": (128, "", "fatal: not a git repository")})
    info = version.build_info(run=run, env={}, repo=tmp_path)
    assert info == {"sha": "", "built_at": "", "dirty": False, "source": "unknown"}


def test_build_info_unknown_when_git_missing(tmp_path):
    run = FakeRun({"git rev-parse --short HEAD": FileNotFoundError("git")})
    info = version.build_info(run=run, env={}, repo=tmp_path)
    assert info["source"] == "unknown"
    assert info["sha"] == ""


def test_describe_formats_sha_and_day():
    info = {"sha": "a517483", "built_at": "2026-09-17T00:00:48+05:30", "dirty": False, "source": "git"}
    assert version.describe(info) == "Veronica 0.1.0 (a517483, 17 Sep)"


def test_describe_without_date():
    info = {"sha": "a517483", "built_at": "", "dirty": False, "source": "git"}
    assert version.describe(info) == "Veronica 0.1.0 (a517483)"


def test_describe_unknown_build():
    info = {"sha": "", "built_at": "", "dirty": False, "source": "unknown"}
    assert version.describe(info) == "Veronica 0.1.0 (unknown build)"


def test_describe_defaults_to_build_info(monkeypatch):
    monkeypatch.setattr(
        version,
        "build_info",
        lambda *a, **k: {"sha": "abc1234", "built_at": "2026-01-02T03:04:05+00:00", "dirty": False, "source": "git"},
    )
    assert version.describe() == "Veronica 0.1.0 (abc1234, 2 Jan)"


def test_repo_constant_points_at_repo_root():
    assert version.REPO == Path(__file__).resolve().parents[1]
