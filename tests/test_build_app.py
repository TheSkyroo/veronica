"""scripts/build_app.py: the PyInstaller build, driven with a fake `run`
(no PyInstaller, no git), plus the generated Veronica.exe entry script."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from tests.fakes import FakeRun

REPO = Path(__file__).resolve().parent.parent

GIT_SCRIPT = {
    "git rev-parse --short HEAD": (0, "a517483\n", ""),
    "git log -1 --format=%cI": (0, "2026-09-17T00:00:48+05:30\n", ""),
    "git status --porcelain": (0, "", ""),
}


def _load_build_app():
    spec = importlib.util.spec_from_file_location("build_app", REPO / "scripts" / "build_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakePyInstallerRun(FakeRun):
    """git answers from GIT_SCRIPT; a PyInstaller call "builds" the onedir
    folder it was asked for (or fails, or builds nothing)."""

    def __init__(self, *, rc=0, produce=True, stderr=""):
        super().__init__(GIT_SCRIPT)
        self.rc, self.produce, self.stderr = rc, produce, stderr
        self.pyinstaller = None

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        if argv[1:3] == ["-m", "PyInstaller"]:
            self.calls.append(argv)
            self.kwargs.append(kwargs)
            self.pyinstaller = argv
            if self.produce and self.rc == 0:
                dist = Path(argv[argv.index("--distpath") + 1]) / "Veronica"
                (dist / "_internal").mkdir(parents=True)
                (dist / "Veronica.exe").write_bytes(b"MZ fake")
            import subprocess
            return subprocess.CompletedProcess(argv, self.rc, "", self.stderr)
        return super().__call__(argv, **kwargs)


@pytest.fixture
def repo(tmp_path):
    """A minimal checkout: a venv interpreter and the icon."""
    r = tmp_path / "repo"
    py = r / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    py.parent.mkdir(parents=True)
    py.write_text("")
    (r / "assets").mkdir()
    (r / "assets" / "Veronica.ico").write_bytes(b"\0\0\1\0")
    return r


def test_build_app_runs_pyinstaller_and_installs_dist(repo):
    mod = _load_build_app()
    run = FakePyInstallerRun()
    exe = mod.build_app(repo=repo, run=run)
    assert exe == repo / "dist" / "Veronica" / "Veronica.exe"
    assert exe.read_bytes() == b"MZ fake"
    assert not (repo / "build" / "pyinstaller" / "dist" / "Veronica").exists()   # moved, not copied
    argv = run.pyinstaller
    assert argv[0] == str(mod._venv_python(repo))              # the checkout's own venv
    assert run.kwargs[-1]["cwd"] == repo


def test_pyinstaller_argv_is_a_windowed_onedir_build_with_icon_and_data(repo):
    mod = _load_build_app()
    run = FakePyInstallerRun()
    mod.build_app(repo=repo, run=run)
    argv = run.pyinstaller
    assert {"--onedir", "--windowed", "--noconfirm", "--clean"} <= set(argv)
    assert argv[argv.index("--name") + 1] == "Veronica"
    assert argv[argv.index("--icon") + 1] == str(repo / "assets" / "Veronica.ico")
    assert argv[3] == str(repo / "build" / "pyinstaller" / "veronica_entry.py")
    pairs = list(zip(argv, argv[1:]))
    # every veronica module (the brains run `Veronica.exe -m veronica.tools.serve`)
    # and its web pages / browser extension
    assert ("--collect-submodules", "veronica") in pairs
    assert ("--collect-data", "veronica") in pairs
    assert ("--collect-all", "webview") in pairs and ("--hidden-import", "pystray._win32") in pairs
    build_json = repo / "build" / "pyinstaller" / "build.json"
    assert ("--add-data", f"{build_json}{mod.os.pathsep}.") in pairs
    # models are never bundled
    assert not any("models" in a for a in argv)


def test_build_json_records_the_commit_and_the_checkout(repo):
    mod = _load_build_app()
    mod.build_app(repo=repo, run=FakePyInstallerRun())
    data = json.loads((repo / "build" / "pyinstaller" / "build.json").read_text())
    assert data == {"sha": "a517483", "built_at": "2026-09-17T00:00:48+05:30", "dirty": False,
                    "source": "git", "repo": str(repo)}


def test_build_app_without_icon_still_builds(repo, capsys):
    (repo / "assets" / "Veronica.ico").unlink()
    mod = _load_build_app()
    run = FakePyInstallerRun()
    mod.build_app(repo=repo, run=run)
    assert "--icon" not in run.pyinstaller
    assert "make_icon.py" in capsys.readouterr().out


def test_build_app_is_idempotent(repo):
    mod = _load_build_app()
    mod.build_app(repo=repo, run=FakePyInstallerRun())
    stale = repo / "dist" / "Veronica" / "stale.txt"
    stale.write_text("old")
    exe = mod.build_app(repo=repo, run=FakePyInstallerRun())
    assert exe.exists() and not stale.exists()
    assert sorted(p.name for p in (repo / "dist").iterdir()) == ["Veronica"]   # no .old left behind


def test_build_app_stages_next_to_a_running_install(repo, monkeypatch, capsys):
    mod = _load_build_app()
    mod.build_app(repo=repo, run=FakePyInstallerRun())
    target = repo / "dist" / "Veronica"
    real_rename = Path.rename

    def locked(self, dest):
        if self == target:
            raise PermissionError("in use")
        return real_rename(self, dest)

    monkeypatch.setattr(Path, "rename", locked)
    exe = mod.build_app(repo=repo, run=FakePyInstallerRun())
    assert exe == target / "Veronica.exe"                       # where it'll be after the restart
    staged = repo / "dist" / "Veronica.new"
    assert (staged / "Veronica.exe").exists() and (target / "Veronica.exe").exists()
    assert "staged" in capsys.readouterr().out
    # the relaunch helper looks for it in the same place
    from veronica.ui.relaunch import staged_dir
    assert staged_dir(exe) == staged


def test_build_app_fails_clearly_without_the_venv(tmp_path):
    mod = _load_build_app()
    with pytest.raises(RuntimeError, match="uv sync"):
        mod.build_app(repo=tmp_path, run=FakePyInstallerRun())


def test_build_app_reports_pyinstaller_failure(repo):
    mod = _load_build_app()
    run = FakePyInstallerRun(rc=1, stderr="lots of output\nModuleNotFoundError: No module named 'webview'\n")
    with pytest.raises(RuntimeError, match="No module named 'webview'"):
        mod.build_app(repo=repo, run=run)
    assert not (repo / "dist").exists()


def test_build_app_fails_when_no_exe_was_produced(repo):
    mod = _load_build_app()
    with pytest.raises(RuntimeError, match="did not produce"):
        mod.build_app(repo=repo, run=FakePyInstallerRun(produce=False))


# -- the generated entry script --------------------------------------------------------

@pytest.fixture
def entry(tmp_path, monkeypatch):
    mod = _load_build_app()
    ns = types.ModuleType("veronica_entry")
    exec(compile(mod.ENTRY_SOURCE, "veronica_entry.py", "exec"), ns.__dict__)
    bundle = tmp_path / "_internal"
    bundle.mkdir()
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.delenv("VERONICA_BUNDLE_BUILD", raising=False)
    monkeypatch.delenv("VERONICA_REPO", raising=False)
    monkeypatch.chdir(tmp_path)
    return ns, bundle


def test_entry_exports_build_info_and_enters_the_checkout(entry, tmp_path, monkeypatch):
    ns, bundle = entry
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (bundle / "build.json").write_text(json.dumps({"sha": "a517483", "repo": str(checkout)}))
    ns._setup()
    import os
    assert os.environ["VERONICA_BUNDLE_BUILD"] == str(bundle / "build.json")
    assert os.environ["VERONICA_REPO"] == str(checkout)
    assert Path.cwd() == checkout


def test_entry_tolerates_a_missing_build_json(entry, tmp_path):
    ns, bundle = entry
    ns._setup()
    import os
    assert os.environ["VERONICA_BUNDLE_BUILD"] == str(bundle / "build.json")
    assert "VERONICA_REPO" not in os.environ
    assert Path.cwd() == tmp_path


def test_entry_runs_dash_m_modules_like_python(entry, tmp_path, monkeypatch):
    ns, _ = entry
    pkg = tmp_path / "mods"
    pkg.mkdir()
    (pkg / "entry_probe.py").write_text(
        "import sys\nRESULT = list(sys.argv)\nsys.modules['entry_probe_result'] = RESULT\n")
    monkeypatch.syspath_prepend(str(pkg))
    monkeypatch.setattr(sys, "argv", ["Veronica.exe", "-m", "entry_probe", "serve", "--flag"])
    ns.main()
    assert sys.modules.pop("entry_probe_result")[1:] == ["serve", "--flag"]
