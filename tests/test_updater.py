import subprocess
from pathlib import Path

import pytest

from tests.fakes import FakeRun
from veronica import updater
from veronica.updater import UpdateError, UpdateStatus

REPO = Path("/fake/repo")

HEAD = "a517483"
REMOTE = "b000001"

GIT_BASE = {
    "git rev-parse --short HEAD": (0, f"{HEAD}\n", ""),
    "git remote get-url origin": (0, "git@github.com:manik/veronica.git\n", ""),
    "git fetch --quiet origin": (0, "", ""),
    "git rev-parse --abbrev-ref HEAD": (0, "batch-d\n", ""),
    "git rev-parse --short origin/batch-d": (0, f"{HEAD}\n", ""),
    "git rev-list --count HEAD..origin/batch-d": (0, "0\n", ""),
}
REMOTE_AHEAD = {
    "git rev-parse --short origin/batch-d": (0, f"{REMOTE}\n", ""),
    "git rev-list --count HEAD..origin/batch-d": (0, "2\n", ""),
}


def _info(sha: str) -> dict:
    return {"sha": sha, "built_at": "2026-09-17T00:00:48+05:30", "dirty": False, "source": "bundle"}


# --- check --------------------------------------------------------------------


def test_check_remote_ahead():
    run = FakeRun({**GIT_BASE, **REMOTE_AHEAD})
    st = updater.check(REPO, run=run, info=_info(HEAD))
    assert st == UpdateStatus(
        available=True,
        kind="remote",
        detail=f"A newer version is on origin ({REMOTE}).",
        running_sha=HEAD,
        head_sha=HEAD,
        remote_sha=REMOTE,
    )
    assert run.argv_strings == [
        "git rev-parse --short HEAD",
        "git remote get-url origin",
        "git fetch --quiet origin",
        "git rev-parse --abbrev-ref HEAD",
        "git rev-parse --short origin/batch-d",
        "git rev-list --count HEAD..origin/batch-d",
    ]
    assert all(kw.get("cwd") == REPO for kw in run.kwargs)
    fetch_kwargs = run.kwargs[2]
    assert fetch_kwargs.get("timeout") == 10


def test_check_local_when_running_sha_behind_head():
    run = FakeRun(GIT_BASE)
    st = updater.check(REPO, run=run, info=_info("0ld0000"))
    assert st.available is True
    assert st.kind == "local"
    assert st.detail == f"Restart to run the latest code ({HEAD})."
    assert st.running_sha == "0ld0000"
    assert st.head_sha == HEAD
    assert st.remote_sha == HEAD


def test_check_none_when_everything_matches():
    run = FakeRun(GIT_BASE)
    st = updater.check(REPO, run=run, info=_info(HEAD))
    assert st.available is False
    assert st.kind == "none"
    assert st.detail == f"You're on the latest ({HEAD})."
    assert st.remote_sha == HEAD


def test_check_remote_wins_over_local():
    run = FakeRun({**GIT_BASE, **REMOTE_AHEAD})
    st = updater.check(REPO, run=run, info=_info("0ld0000"))
    assert st.kind == "remote"


def test_check_head_ahead_of_origin_is_not_an_update():
    # origin/<branch> differs from HEAD but has nothing HEAD lacks (local
    # commits not pushed yet): not "remote"
    run = FakeRun({**GIT_BASE, "git rev-parse --short origin/batch-d": (0, f"{REMOTE}\n", ""),
                   "git rev-list --count HEAD..origin/batch-d": (0, "0\n", "")})
    st = updater.check(REPO, run=run, info=_info(HEAD))
    assert st.kind == "none" and st.available is False
    assert st.remote_sha == REMOTE
    st = updater.check(REPO, run=run, info=_info("0ld0000"))
    assert st.kind == "local"


def test_check_no_remote_branch_falls_through_to_local():
    # origin exists but this branch was never pushed: rev-parse origin/<branch> fails
    run = FakeRun({**GIT_BASE, "git rev-parse --short origin/batch-d": (128, "", "fatal: unknown revision")})
    st = updater.check(REPO, run=run, info=_info("0ld0000"))
    assert st.kind == "local" and st.available is True
    assert st.remote_sha is None
    st = updater.check(REPO, run=run, info=_info(HEAD))
    assert st.kind == "none" and st.available is False
    assert st.detail == "Couldn't check: fatal: unknown revision"


def test_check_fetch_failure_still_reports_local_staleness():
    run = FakeRun({**GIT_BASE, "git fetch --quiet origin": (128, "", "fatal: unable to access origin\n")})
    st = updater.check(REPO, run=run, info=_info("0ld0000"))
    assert st.kind == "local" and st.available is True


def test_check_no_remote_still_computes_local():
    run = FakeRun({**GIT_BASE, "git remote get-url origin": (2, "", "error: No such remote 'origin'")})
    st = updater.check(REPO, run=run, info=_info("0ld0000"))
    assert st.kind == "local"
    assert st.available is True
    assert st.remote_sha is None
    assert "git fetch --quiet origin" not in run.argv_strings


def test_check_no_remote_and_up_to_date_is_none():
    run = FakeRun({**GIT_BASE, "git remote get-url origin": (2, "", "error: No such remote 'origin'")})
    st = updater.check(REPO, run=run, info=_info(HEAD))
    assert st.kind == "none"
    assert st.available is False
    assert st.remote_sha is None


def test_check_unknown_build_with_no_remote_is_none():
    # source=unknown -> sha "" -> can't claim local staleness
    run = FakeRun({**GIT_BASE, "git remote get-url origin": (2, "", "no remote")})
    st = updater.check(REPO, run=run, info={"sha": "", "built_at": "", "dirty": False, "source": "unknown"})
    assert st.kind == "none"
    assert st.running_sha == ""


def test_check_fetch_failure_reports_error():
    run = FakeRun({**GIT_BASE, "git fetch --quiet origin": (128, "", "fatal: unable to access origin\n")})
    st = updater.check(REPO, run=run, info=_info(HEAD))
    assert st.available is False
    assert st.kind == "none"
    assert st.detail == "Couldn't check: fatal: unable to access origin"
    assert st.head_sha == HEAD
    assert st.remote_sha is None


def test_check_fetch_timeout_reports_error():
    run = FakeRun({**GIT_BASE, "git fetch --quiet origin": subprocess.TimeoutExpired(["git", "fetch"], 10)})
    st = updater.check(REPO, run=run, info=_info(HEAD))
    assert st.kind == "none"
    assert st.detail.startswith("Couldn't check:")


def test_check_head_failure_reports_error():
    run = FakeRun({**GIT_BASE, "git rev-parse --short HEAD": (128, "", "fatal: not a git repository")})
    st = updater.check(REPO, run=run, info=_info(HEAD))
    assert st.kind == "none"
    assert st.detail == "Couldn't check: fatal: not a git repository"
    assert st.head_sha == ""


def test_check_git_missing_reports_error():
    run = FakeRun({"git rev-parse --short HEAD": FileNotFoundError("git")})
    st = updater.check(REPO, run=run, info=_info(HEAD))
    assert st.kind == "none"
    assert st.detail.startswith("Couldn't check:")


def test_check_defaults_info_to_build_info(monkeypatch):
    seen = {}

    def fake_build_info(run=None, env=None, repo=None):
        seen["run"] = run
        seen["repo"] = repo
        return _info("0ld0000")

    monkeypatch.setattr(updater.version, "build_info", fake_build_info)
    run = FakeRun(GIT_BASE)
    st = updater.check(REPO, run=run)
    assert st.kind == "local"
    assert seen == {"run": run, "repo": REPO}


# --- update -------------------------------------------------------------------


def _status(kind: str) -> UpdateStatus:
    return UpdateStatus(
        available=kind != "none",
        kind=kind,
        detail="",
        running_sha=HEAD,
        head_sha=HEAD,
        remote_sha=REMOTE if kind == "remote" else None,
    )


UPDATE_OK = {
    "git pull --ff-only --quiet": (0, "", ""),
    "uv sync": (0, "", ""),
}


def test_update_remote_pulls_syncs_builds_in_order():
    run = FakeRun(UPDATE_OK)
    built = []
    log = updater.update(REPO, _status("remote"), run=run, build=lambda: built.append(1) or Path("/x/Veronica.app"), which=lambda n: "/usr/local/bin/uv")
    assert run.argv_strings == ["git pull --ff-only --quiet", "uv sync"]
    assert built == [1]
    assert all(kw.get("cwd") == REPO for kw in run.kwargs)
    assert "git pull" in log
    assert "uv sync" in log
    assert "Veronica.app" in log


def test_update_local_skips_pull():
    run = FakeRun(UPDATE_OK)
    built = []
    updater.update(REPO, _status("local"), run=run, build=lambda: built.append(1), which=lambda n: "/usr/local/bin/uv")
    assert run.argv_strings == ["uv sync"]
    assert built == [1]


def test_update_skips_uv_sync_when_uv_missing():
    run = FakeRun(UPDATE_OK)
    built = []
    log = updater.update(REPO, _status("remote"), run=run, build=lambda: built.append(1), which=lambda n: None)
    assert run.argv_strings == ["git pull --ff-only --quiet"]
    assert built == [1]
    assert "uv" in log  # says it skipped


def test_update_pull_failure_raises_and_stops():
    run = FakeRun({**UPDATE_OK, "git pull --ff-only --quiet": (1, "", "fatal: Not possible to fast-forward\n")})
    built = []
    with pytest.raises(UpdateError, match="Not possible to fast-forward"):
        updater.update(REPO, _status("remote"), run=run, build=lambda: built.append(1), which=lambda n: "/uv")
    assert run.argv_strings == ["git pull --ff-only --quiet"]
    assert built == []


def test_update_sync_failure_raises_and_stops():
    run = FakeRun({**UPDATE_OK, "uv sync": (2, "", "error: lockfile out of date\n")})
    built = []
    with pytest.raises(UpdateError, match="lockfile out of date"):
        updater.update(REPO, _status("local"), run=run, build=lambda: built.append(1), which=lambda n: "/uv")
    assert built == []


def test_update_build_failure_raises_update_error():
    run = FakeRun(UPDATE_OK)

    def boom():
        raise RuntimeError("claude CLI not found")

    with pytest.raises(UpdateError, match="claude CLI not found"):
        updater.update(REPO, _status("local"), run=run, build=boom, which=lambda n: None)


def test_update_command_exception_raises_update_error():
    run = FakeRun({"git pull --ff-only --quiet": FileNotFoundError("git")})
    with pytest.raises(UpdateError, match="git"):
        updater.update(REPO, _status("remote"), run=run, build=lambda: None, which=lambda n: None)


def test_update_default_build_loads_build_app_script(monkeypatch):
    # the lazy default resolves scripts/build_app.py's build_app without importing it as a package
    fn = updater._default_build()
    assert callable(fn)
    assert fn.__name__ == "build_app"
