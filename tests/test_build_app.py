import importlib.util
import plistlib
import stat
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


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

    with open(plist_path, "rb") as f:
        plist = plistlib.load(f)
    assert plist["CFBundleName"] == "Veronica"
    assert plist["CFBundleIdentifier"] == "io.manik.veronica"
    assert plist["CFBundleExecutable"] == "Veronica"
    assert plist["CFBundleIconFile"] == "Veronica"
    assert plist["LSUIElement"] is True
    assert plist["LSMinimumSystemVersion"] == "13.0"
    assert "NSMicrophoneUsageDescription" in plist
    assert "NSAppleEventsUsageDescription" in plist
    assert plist["NSHighResolutionCapable"] is True
    assert plist["CFBundleVersion"] == plist["CFBundleShortVersionString"]


def test_build_app_copies_icon_when_present(tmp_path):
    build_app = _load_build_app()
    icns_src = REPO / "assets" / "Veronica.icns"
    if not icns_src.exists():
        import pytest
        pytest.skip("assets/Veronica.icns not built")
    app = build_app.build_app(repo=REPO, dist_dir=tmp_path, codesign_enabled=False)
    assert (app / "Contents" / "Resources" / "Veronica.icns").is_file()


def test_build_app_is_idempotent(tmp_path):
    build_app = _load_build_app()
    app1 = build_app.build_app(repo=REPO, dist_dir=tmp_path, codesign_enabled=False)
    marker = app1 / "stray_file"
    marker.write_text("leftover")
    app2 = build_app.build_app(repo=REPO, dist_dir=tmp_path, codesign_enabled=False)
    assert app1 == app2
    assert not marker.exists()


def test_build_app_skips_codesign_when_missing(tmp_path, monkeypatch):
    build_app = _load_build_app()
    monkeypatch.setattr(build_app.shutil, "which", lambda name: None)
    # should not raise even though codesign_enabled=True
    app = build_app.build_app(repo=REPO, dist_dir=tmp_path, codesign_enabled=True)
    assert app.is_dir()
