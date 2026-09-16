import importlib.util
import plistlib
import stat
from pathlib import Path

import pytest

from tests.fakes import FakeRun

REPO = Path(__file__).resolve().parent.parent
FAKE_CLAUDE = Path("/fake/claude/bin/claude")

GIT_SCRIPT = {
    "git rev-parse --short HEAD": (0, "a517483\n", ""),
    "git log -1 --format=%cI": (0, "2026-09-17T00:00:48+05:30\n", ""),
    "git status --porcelain": (0, "", ""),
}


def _fake_run() -> FakeRun:
    return FakeRun(GIT_SCRIPT)


def _load_build_app():
    spec = importlib.util.spec_from_file_location("build_app", REPO / "scripts" / "build_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_app_structure_and_plist(tmp_path):
    build_app = _load_build_app()
    app = build_app.build_app(
        repo=REPO,
        dist_dir=tmp_path,
        venv_python=Path("/fake/.venv/bin/python"),
        codesign_enabled=False,
        claude_bin=FAKE_CLAUDE,
        run=_fake_run(),
    )

    assert app == tmp_path / "Veronica.app"
    launcher = app / "Contents" / "MacOS" / "Veronica"
    plist_path = app / "Contents" / "Info.plist"
    pkginfo = app / "Contents" / "PkgInfo"

    assert launcher.is_file()
    assert plist_path.is_file()
    assert pkginfo.is_file()
    assert pkginfo.read_text() == "APPL????"

    # launcher is executable and points at the given repo/python
    mode = launcher.stat().st_mode
    assert mode & stat.S_IXUSR
    text = launcher.read_text()
    assert str(REPO) in text
    assert "/fake/.venv/bin/python" in text
    assert "-m veronica" in text

    # PATH is exported with the resolved claude dir before exec
    assert 'export PATH="/fake/claude/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"' in text
    assert 'export LANG="${LANG:-en_US.UTF-8}"' in text
    path_idx = text.index("export PATH=")
    exec_idx = text.index("exec ")
    assert path_idx < exec_idx

    with open(plist_path, "rb") as f:
        plist = plistlib.load(f)
    assert plist["CFBundleName"] == "Veronica"
    assert plist["CFBundleIdentifier"] == "io.manik.veronica"
    assert plist["CFBundleExecutable"] == "Veronica"
    assert plist["CFBundleIconFile"] == "Veronica"
    assert plist["LSUIElement"] is True
    assert plist["LSMinimumSystemVersion"] == "13.0"
    assert "NSMicrophoneUsageDescription" in plist
    desc = plist["NSAppleEventsUsageDescription"]
    for app_name in ("Calendar", "Mail", "Reminders", "Notes", "Music", "Chrome", "Safari", "System Events"):
        assert app_name in desc
    assert plist["NSHighResolutionCapable"] is True
    assert plist["CFBundleVersion"] == plist["CFBundleShortVersionString"]


def test_build_app_copies_icon_when_present(tmp_path):
    build_app = _load_build_app()
    icns_src = REPO / "assets" / "Veronica.icns"
    if not icns_src.exists():
        pytest.skip("assets/Veronica.icns not built")
    app = build_app.build_app(repo=REPO, dist_dir=tmp_path, codesign_enabled=False, claude_bin=FAKE_CLAUDE, run=_fake_run())
    assert (app / "Contents" / "Resources" / "Veronica.icns").is_file()


def test_build_app_is_idempotent(tmp_path):
    build_app = _load_build_app()
    app1 = build_app.build_app(repo=REPO, dist_dir=tmp_path, codesign_enabled=False, claude_bin=FAKE_CLAUDE, run=_fake_run())
    marker = app1 / "stray_file"
    marker.write_text("leftover")
    app2 = build_app.build_app(repo=REPO, dist_dir=tmp_path, codesign_enabled=False, claude_bin=FAKE_CLAUDE, run=_fake_run())
    assert app1 == app2
    assert not marker.exists()


def test_build_app_skips_codesign_when_missing(tmp_path, monkeypatch):
    build_app = _load_build_app()
    monkeypatch.setattr(build_app.shutil, "which", lambda name: None)
    # should not raise even though codesign_enabled=True (claude_bin passed
    # explicitly so the claude-resolution check isn't what's being tested here)
    app = build_app.build_app(repo=REPO, dist_dir=tmp_path, codesign_enabled=True, claude_bin=FAKE_CLAUDE, run=_fake_run())
    assert app.is_dir()


def test_build_app_fails_clearly_when_claude_not_found(tmp_path, monkeypatch):
    build_app = _load_build_app()
    monkeypatch.setattr(build_app.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="claude"):
        build_app.build_app(repo=REPO, dist_dir=tmp_path, codesign_enabled=False, run=_fake_run())


def test_build_app_resolves_claude_via_which(tmp_path, monkeypatch):
    build_app = _load_build_app()
    monkeypatch.setattr(build_app.shutil, "which", lambda name: "/opt/homebrew/bin/claude" if name == "claude" else None)
    app = build_app.build_app(repo=REPO, dist_dir=tmp_path, codesign_enabled=False, run=_fake_run())
    launcher = app / "Contents" / "MacOS" / "Veronica"
    assert "/opt/homebrew/bin" in launcher.read_text()


def test_build_app_writes_build_json_and_launcher_exports_it(tmp_path):
    import json

    build_app = _load_build_app()
    run = _fake_run()
    app = build_app.build_app(repo=REPO, dist_dir=tmp_path, codesign_enabled=False, claude_bin=FAKE_CLAUDE, run=run)

    build_json = app / "Contents" / "Resources" / "build.json"
    assert build_json.is_file()
    data = json.loads(build_json.read_text())
    assert data["sha"] == "a517483"
    assert data["built_at"] == "2026-09-17T00:00:48+05:30"
    assert data["dirty"] is False
    assert data["source"] == "git"
    # git ran against the repo, not the cwd
    assert all(kw.get("cwd") == REPO for kw in run.kwargs)

    text = (app / "Contents" / "MacOS" / "Veronica").read_text()
    assert f'export VERONICA_BUNDLE_BUILD="{build_json}"' in text
    assert text.index("export VERONICA_BUNDLE_BUILD=") < text.index("exec ")
    # the launcher runs `python -m veronica` (argv0 = __main__.py), so it must
    # tell the process where the .app is for relaunch / Start at Login
    assert f'export VERONICA_APP_BUNDLE="{app}"' in text
    assert text.index("export VERONICA_APP_BUNDLE=") < text.index("exec ")
