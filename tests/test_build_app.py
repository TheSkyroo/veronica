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


def _fake_compiler(source: str, out, defines: dict) -> None:
    """Stand-in for clang: writes the C source plus the -D defines as text so
    tests can assert what got baked into the launcher."""
    out.write_text(source + "\n" + "\n".join(f"#define {k} \"{v}\"" for k, v in defines.items()) + "\n")


FAKE_LIBPYTHON = Path("/fake/python/lib/libpython3.12.dylib")


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
        compiler=_fake_compiler,
        libpython=FAKE_LIBPYTHON,
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
    assert f'#define REPO_DIR "{REPO}"' in text
    assert '#define VENV_PYTHON "/fake/.venv/bin/python"' in text
    assert '#define LIBPYTHON "/fake/python/lib/libpython3.12.dylib"' in text
    assert 'args[n++] = "-m";' in text and 'args[n++] = "veronica";' in text

    # PATH is baked in with the resolved claude dir first, and set before Python starts
    assert '#define PATH_PREFIX "/fake/claude/bin:' in text
    assert "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin" in text
    assert 'setenv("PATH", PATH_PREFIX, 1);' in text
    assert text.index('setenv("PATH"') < text.index("py_main(n, args)")

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
    app = build_app.build_app(repo=REPO, dist_dir=tmp_path, codesign_enabled=False, claude_bin=FAKE_CLAUDE, compiler=_fake_compiler, libpython=FAKE_LIBPYTHON, run=_fake_run())
    assert (app / "Contents" / "Resources" / "Veronica.icns").is_file()


def test_build_app_is_idempotent(tmp_path):
    build_app = _load_build_app()
    app1 = build_app.build_app(repo=REPO, dist_dir=tmp_path, codesign_enabled=False, claude_bin=FAKE_CLAUDE, compiler=_fake_compiler, libpython=FAKE_LIBPYTHON, run=_fake_run())
    marker = app1 / "stray_file"
    marker.write_text("leftover")
    app2 = build_app.build_app(repo=REPO, dist_dir=tmp_path, codesign_enabled=False, claude_bin=FAKE_CLAUDE, compiler=_fake_compiler, libpython=FAKE_LIBPYTHON, run=_fake_run())
    assert app1 == app2
    assert not marker.exists()


def test_build_app_skips_codesign_when_missing(tmp_path, monkeypatch):
    build_app = _load_build_app()
    monkeypatch.setattr(build_app.shutil, "which", lambda name: None)
    # should not raise even though codesign_enabled=True (claude_bin passed
    # explicitly so the claude-resolution check isn't what's being tested here)
    app = build_app.build_app(repo=REPO, dist_dir=tmp_path, codesign_enabled=True, claude_bin=FAKE_CLAUDE, compiler=_fake_compiler, libpython=FAKE_LIBPYTHON, run=_fake_run())
    assert app.is_dir()


def test_build_app_fails_clearly_when_claude_not_found(tmp_path, monkeypatch):
    build_app = _load_build_app()
    monkeypatch.setattr(build_app.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="claude"):
        build_app.build_app(repo=REPO, dist_dir=tmp_path, codesign_enabled=False, run=_fake_run())


def test_build_app_resolves_claude_via_which(tmp_path, monkeypatch):
    build_app = _load_build_app()
    monkeypatch.setattr(build_app.shutil, "which", lambda name: "/opt/homebrew/bin/claude" if name == "claude" else None)
    app = build_app.build_app(repo=REPO, dist_dir=tmp_path, codesign_enabled=False, run=_fake_run(), compiler=_fake_compiler, libpython=FAKE_LIBPYTHON)
    launcher = app / "Contents" / "MacOS" / "Veronica"
    assert "/opt/homebrew/bin" in launcher.read_text()


def test_build_app_writes_build_json_and_launcher_exports_it(tmp_path):
    import json

    build_app = _load_build_app()
    run = _fake_run()
    app = build_app.build_app(repo=REPO, dist_dir=tmp_path, codesign_enabled=False, claude_bin=FAKE_CLAUDE, compiler=_fake_compiler, libpython=FAKE_LIBPYTHON, run=run)

    build_json = tmp_path / "veronica-build.json"   # outside the bundle: keeps the cdhash (and TCC grants) stable
    assert build_json.is_file()
    data = json.loads(build_json.read_text())
    assert data["sha"] == "a517483"
    assert data["built_at"] == "2026-09-17T00:00:48+05:30"
    assert data["dirty"] is False
    assert data["source"] == "git"
    # git ran against the repo, not the cwd
    assert all(kw.get("cwd") == REPO for kw in run.kwargs)

    text = (app / "Contents" / "MacOS" / "Veronica").read_text()
    assert f'#define BUILD_JSON "{build_json}"' in text
    assert 'setenv("VERONICA_BUNDLE_BUILD", BUILD_JSON, 1);' in text
    # the launcher runs `python -m veronica` (argv0 = __main__.py), so it must
    # tell the process where the .app is for relaunch / Start at Login
    assert f'#define APP_BUNDLE "{app}"' in text
    assert 'setenv("VERONICA_APP_BUNDLE", APP_BUNDLE, 1);' in text


def test_build_app_fails_clearly_without_clang(tmp_path, monkeypatch):
    build_app = _load_build_app()
    monkeypatch.setattr(build_app.shutil, "which", lambda name: None if name == "clang" else "/fake/claude/bin/claude")
    with pytest.raises(RuntimeError, match="clang not found"):
        build_app.build_app(repo=REPO, dist_dir=tmp_path, codesign_enabled=False, claude_bin=FAKE_CLAUDE,
                            libpython=FAKE_LIBPYTHON, run=_fake_run())


def test_libpython_for_resolves_symlinked_venv_python(tmp_path):
    build_app = _load_build_app()
    root = tmp_path / "cpython"
    (root / "bin").mkdir(parents=True)
    (root / "lib").mkdir()
    (root / "bin" / "python3.12").write_text("")
    (root / "lib" / "libpython3.12.dylib").write_text("")
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(root / "bin" / "python3.12")
    assert build_app._libpython_for(venv_bin / "python") == root / "lib" / "libpython3.12.dylib"
    (root / "lib" / "libpython3.12.dylib").unlink()
    with pytest.raises(RuntimeError, match="libpython"):
        build_app._libpython_for(venv_bin / "python")


def test_bundle_bytes_are_identical_across_rebuilds(tmp_path):
    """Nothing inside the .app may change between two builds of the same
    source (the build stamp lives outside), or TCC forgets every grant."""
    import hashlib

    build_app = _load_build_app()

    def digest(app):
        h = hashlib.sha256()
        for f in sorted(p for p in app.rglob("*") if p.is_file()):
            h.update(str(f.relative_to(app)).encode())
            h.update(f.read_bytes())
        return h.hexdigest()

    kw = dict(repo=REPO, codesign_enabled=False, claude_bin=FAKE_CLAUDE, compiler=_fake_compiler,
              libpython=FAKE_LIBPYTHON, venv_python=Path("/fake/.venv/bin/python"))
    a = build_app.build_app(dist_dir=tmp_path, run=_fake_run(), **kw)
    first = digest(a)
    later = FakeRun({**GIT_SCRIPT, "git log -1 --format=%cI": (0, "2027-01-01T00:00:00+00:00\n", "")})
    b = build_app.build_app(dist_dir=tmp_path, run=later, **kw)   # same dist dir, new build stamp
    assert digest(b) == first


def test_brain_dirs_follow_per_shell_symlinks(tmp_path):
    """A version manager's per-shell bin (fnm/nvm "multishell") disappears
    with the shell that created it, so the baked-in PATH must also carry the
    node installation the shim resolves to."""
    from scripts.build_app import _brain_dirs

    install = tmp_path / "node-versions" / "v26" / "installation"
    (install / "bin").mkdir(parents=True)
    pkg_bin = install / "lib" / "node_modules" / "@openai" / "codex" / "bin"
    pkg_bin.mkdir(parents=True)
    (pkg_bin / "codex.js").write_text("#!/usr/bin/env node\n")
    shell_bin = tmp_path / "fnm_multishells" / "123" / "bin"
    shell_bin.mkdir(parents=True)
    (shell_bin / "codex").symlink_to(pkg_bin / "codex.js")

    dirs = _brain_dirs(which=lambda n: str(shell_bin / "codex") if n == "codex" else None)
    assert str(install / "bin") in dirs      # survives the shell
    assert str(shell_bin) in dirs            # still first choice while it exists


def test_launcher_path_starts_with_the_brain_dirs():
    from scripts.build_app import _launcher_defines
    from pathlib import Path

    defines = _launcher_defines(Path("/repo"), Path("/py"), "/claude/bin", Path("/b.json"),
                                Path("/A.app"), Path("/libpython"), brain_dirs=["/node/bin", "/claude/bin"])
    parts = defines["PATH_PREFIX"].split(":")
    assert parts[0] == "/claude/bin" and "/node/bin" in parts
    assert len(parts) == len(set(parts))     # no duplicates
    assert parts[-4:] == ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]
