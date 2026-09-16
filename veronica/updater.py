"""Check for and apply updates to the running checkout.

Two kinds of "update available":

- `remote`: `origin/<branch>` has moved past HEAD (needs `git pull`).
- `local`: HEAD has moved past the commit the running process was built from
  (the bundle's build.json sha) — i.e. someone pulled/committed while the app
  was running; a rebuild + restart is all it takes.

`check()` never raises: any git failure (no remote, offline, not a repo)
becomes `kind="none"` with a "Couldn't check: ..." detail, and a missing
`origin` just skips the remote comparison. `update()` does raise
(`UpdateError`) so the caller can report the failure. `run`, `which` and
`build` are injectable so tests never run real git/uv/build_app.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from veronica import version

FETCH_TIMEOUT_S = 10


class UpdateError(RuntimeError):
    pass


class UpdateInProgress(UpdateError):
    """Raised by the app's update hook when another update (spoken, menu or
    settings window) already holds the update slot."""


@dataclass
class UpdateStatus:
    available: bool
    kind: Literal["remote", "local", "none"]
    detail: str
    running_sha: str
    head_sha: str
    remote_sha: str | None


class _GitFailed(Exception):
    pass


def _git(run, repo: Path, *args: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    kwargs: dict = {"cwd": repo, "capture_output": True, "text": True, "check": False}
    if timeout is not None:
        kwargs["timeout"] = timeout
    try:
        return run(["git", *args], **kwargs)
    except subprocess.TimeoutExpired:
        raise _GitFailed(f"git {' '.join(args)} timed out after {timeout}s") from None
    except OSError as e:
        raise _GitFailed(str(e)) from None


def _git_out(run, repo: Path, *args: str, timeout: float | None = None) -> str:
    proc = _git(run, repo, *args, timeout=timeout)
    if proc.returncode != 0:
        raise _GitFailed((proc.stderr or "").strip() or f"git {' '.join(args)} failed")
    return (proc.stdout or "").strip()


def check(repo: Path, run=subprocess.run, info: dict | None = None) -> UpdateStatus:
    if info is None:
        info = version.build_info(run=run, repo=repo)
    running_sha = info.get("sha") or ""
    head_sha = ""
    remote_sha: str | None = None
    try:
        head_sha = _git_out(run, repo, "rev-parse", "--short", "HEAD")
        has_origin = _git(run, repo, "remote", "get-url", "origin").returncode == 0
        if has_origin:
            _git_out(run, repo, "fetch", "--quiet", "origin", timeout=FETCH_TIMEOUT_S)
            branch = _git_out(run, repo, "rev-parse", "--abbrev-ref", "HEAD")
            remote_sha = _git_out(run, repo, "rev-parse", "--short", f"origin/{branch}")
    except _GitFailed as e:
        return UpdateStatus(False, "none", f"Couldn't check: {e}", running_sha, head_sha, None)

    if remote_sha is not None and remote_sha != head_sha:
        return UpdateStatus(
            True, "remote", f"A newer version is on origin ({remote_sha}).", running_sha, head_sha, remote_sha
        )
    if running_sha and running_sha != head_sha:
        return UpdateStatus(
            True, "local", f"Restart to run the latest code ({head_sha}).", running_sha, head_sha, remote_sha
        )
    return UpdateStatus(False, "none", f"You're on the latest ({head_sha}).", running_sha, head_sha, remote_sha)


def _default_build() -> Callable[[], Path]:
    """scripts/build_app.py's build_app, loaded by path (scripts/ isn't a package)."""
    path = version.REPO / "scripts" / "build_app.py"
    spec = importlib.util.spec_from_file_location("veronica_build_app", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_app


def _run_step(run, repo: Path, argv: list[str], log: list[str]) -> None:
    try:
        proc = run(argv, cwd=repo, capture_output=True, text=True, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        raise UpdateError(f"{' '.join(argv)}: {e}") from None
    if proc.returncode != 0:
        raise UpdateError((proc.stderr or "").strip() or f"{' '.join(argv)} failed ({proc.returncode})")
    log.append(f"{' '.join(argv)}: ok")


def update(
    repo: Path,
    status: UpdateStatus,
    run=subprocess.run,
    build: Callable[[], object] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> str:
    """Pull (remote only), `uv sync` (if uv is present; not `--frozen` —
    uv.lock is gitignored, so the lock must be re-resolved after a pull), rebuild the
    .app. Returns a short log; raises UpdateError on the first failure. The
    caller relaunches afterwards."""
    log: list[str] = []
    if status.kind == "remote":
        _run_step(run, repo, ["git", "pull", "--ff-only", "--quiet"], log)
    if which("uv"):
        _run_step(run, repo, ["uv", "sync"], log)
    else:
        log.append("uv not found: skipped sync")
    try:
        built = (build or _default_build())()
    except UpdateError:
        raise
    except Exception as e:
        raise UpdateError(f"build failed: {e}") from e
    log.append(f"build: {built}" if built is not None else "build: ok")
    return "\n".join(log)
