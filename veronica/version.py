"""App version and build provenance.

`APP_VERSION` is the package version from pyproject (via installed metadata).
`build_info()` answers "which commit is actually running?": when launched from
the built .app, the launcher exports `VERONICA_BUNDLE_BUILD` pointing at the
bundle's `Contents/Resources/build.json` (written by scripts/build_app.py at
build time), so we report what was built even if the repo has since moved on.
Outside the bundle (dev runs) it's computed live from git. `run` and `env` are
injectable so tests never touch real git.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP_NAME = "Veronica"


def _package_version() -> str:
    try:
        return metadata.version("veronica")
    except metadata.PackageNotFoundError:
        return "0.1.0"


APP_VERSION: str = _package_version()

UNKNOWN: dict = {"sha": "", "built_at": "", "dirty": False, "source": "unknown"}


def _git(run, repo: Path, *args: str) -> str:
    """stdout of `git <args>` in `repo`, or raise on any failure."""
    proc = run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "").strip() or f"git {' '.join(args)} failed")
    return (proc.stdout or "").strip()


def _from_bundle(env) -> dict | None:
    path = env.get("VERONICA_BUNDLE_BUILD")
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return {
        "sha": str(data.get("sha", "")),
        "built_at": str(data.get("built_at", "")),
        "dirty": bool(data.get("dirty", False)),
        "source": "bundle",
    }


def _from_git(run, repo: Path) -> dict:
    try:
        sha = _git(run, repo, "rev-parse", "--short", "HEAD")
        built_at = _git(run, repo, "log", "-1", "--format=%cI")
        porcelain = _git(run, repo, "status", "--porcelain")
    except (OSError, subprocess.SubprocessError, RuntimeError):
        return dict(UNKNOWN)
    return {"sha": sha, "built_at": built_at, "dirty": bool(porcelain), "source": "git"}


def build_info(run=subprocess.run, env=os.environ, repo: Path = REPO) -> dict:
    """{"sha", "built_at" (ISO 8601), "dirty", "source": "bundle"|"git"|"unknown"}."""
    return _from_bundle(env) or _from_git(run, repo)


def _short_day(iso: str) -> str:
    """'2026-09-17T00:00:48+05:30' -> '17 Sep'; '' on unparsable input."""
    try:
        dt = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return ""
    return f"{dt.day} {dt:%b}"


def describe(info: dict | None = None) -> str:
    """'Veronica 0.1.0 (a517483, 17 Sep)'; '(unknown build)' when the sha is missing."""
    if info is None:
        info = build_info()
    sha = info.get("sha") or ""
    if not sha:
        return f"{APP_NAME} {APP_VERSION} (unknown build)"
    day = _short_day(info.get("built_at") or "")
    inner = f"{sha}, {day}" if day else sha
    return f"{APP_NAME} {APP_VERSION} ({inner})"
