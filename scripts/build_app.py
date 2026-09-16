"""Build dist/Veronica.app: a menu-bar-only .app bundle wrapping `python -m veronica`.

Layout (per docs/superpowers/specs/2026-09-16-veronica-phases-4-6-design.md,
Phase 6):

    Veronica.app/Contents/
        MacOS/Veronica          shell launcher: exports a PATH that includes the `claude` CLI (resolved
                                 at build time, since Finder/launchd only gives the process
                                 /usr/bin:/bin:/usr/sbin:/sbin, which is missing wherever `claude` actually
                                 lives), cd <repo>, exec <repo>/.venv/bin/python -m veronica "$@"
        Info.plist              CFBundleIdentifier io.manik.veronica, LSUIElement, mic/automation usage strings
        Resources/Veronica.icns copied from assets/Veronica.icns (built by scripts/make_icon.py)
        PkgInfo                 "APPL????"

The build resolves `claude` via `shutil.which("claude")` in the build shell and fails with a clear error
if it isn't found — the Agent SDK needs it at runtime, and by the time the bundle is running under
launchd's minimal PATH it's too late to find it dynamically.

Then `codesign --force --deep -s - dist/Veronica.app` (ad-hoc) so TCC
(microphone/automation) permissions stick to the bundle across rebuilds.

Idempotent: removes any existing dist/Veronica.app first. Run directly:

    uv run python scripts/build_app.py
"""
from __future__ import annotations

import argparse
import plistlib
import shutil
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUNDLE_ID = "io.manik.veronica"
APP_NAME = "Veronica"

def _launcher_script(repo: Path, python: Path, claude_dir: str) -> str:
    # Plain (non-f) strings for the lines containing shell variable
    # expansions ($PATH, ${LANG:-...}) so Python's str.format/f-string
    # brace parsing never sees them.
    return (
        "#!/bin/zsh\n"
        f'export PATH="{claude_dir}:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"\n'
        'export LANG="${LANG:-en_US.UTF-8}"\n'
        f'cd "{repo}"\n'
        f'exec "{python}" -m veronica "$@"\n'
    )


def _read_version(repo: Path) -> str:
    data = tomllib.loads((repo / "pyproject.toml").read_text())
    return data["project"]["version"]


def _info_plist(version: str) -> dict:
    return {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleVersion": version,
        "CFBundleShortVersionString": version,
        "CFBundleExecutable": APP_NAME,
        "CFBundleIconFile": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleInfoDictionaryVersion": "6.0",
        "LSUIElement": True,
        "LSMinimumSystemVersion": "13.0",
        "NSMicrophoneUsageDescription": "Veronica listens for the wake word and your voice commands.",
        "NSAppleEventsUsageDescription": (
            "Veronica reads and creates Calendar events, Mail, and Reminders on your behalf."
        ),
        "NSHighResolutionCapable": True,
    }


def build_app(
    *,
    repo: Path = REPO,
    dist_dir: Path | None = None,
    venv_python: Path | None = None,
    codesign_enabled: bool = True,
    claude_bin: Path | str | None = None,
) -> Path:
    """Build dist/Veronica.app (or dist_dir/Veronica.app) and return its path.

    `claude_bin` overrides where the `claude` CLI is resolved from (for
    tests); by default it's `shutil.which("claude")` in the build shell, and
    the build fails loudly if that comes back empty.
    """
    dist_dir = dist_dir or (repo / "dist")
    venv_python = venv_python or (repo / ".venv" / "bin" / "python")

    resolved_claude = claude_bin or shutil.which("claude")
    if not resolved_claude:
        raise RuntimeError(
            "build_app: `claude` CLI not found on PATH. Install/login the Claude Code CLI "
            "before building the app bundle — the Agent SDK needs it at runtime, and the "
            "built .app launches under launchd's minimal PATH "
            "(/usr/bin:/bin:/usr/sbin:/sbin), which won't include it unless it's baked in "
            "at build time."
        )
    claude_dir = str(Path(resolved_claude).resolve().parent)

    app = dist_dir / f"{APP_NAME}.app"
    if app.exists():
        shutil.rmtree(app)

    macos_dir = app / "Contents" / "MacOS"
    resources_dir = app / "Contents" / "Resources"
    macos_dir.mkdir(parents=True)
    resources_dir.mkdir(parents=True)

    # launcher
    launcher = macos_dir / APP_NAME
    launcher.write_text(_launcher_script(repo, venv_python, claude_dir))
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # Info.plist
    version = _read_version(repo)
    with open(app / "Contents" / "Info.plist", "wb") as f:
        plistlib.dump(_info_plist(version), f)

    # PkgInfo
    (app / "Contents" / "PkgInfo").write_text("APPL????")

    # icon
    icns_src = repo / "assets" / f"{APP_NAME}.icns"
    if icns_src.exists():
        shutil.copy(icns_src, resources_dir / f"{APP_NAME}.icns")
    else:
        print(f"build_app: {icns_src} not found — run scripts/make_icon.py first; app will have no icon")

    if codesign_enabled:
        if shutil.which("codesign") is None:
            print("build_app: codesign not available — skipping ad-hoc signing")
        else:
            subprocess.run(
                ["codesign", "--force", "--deep", "-s", "-", str(app)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

    return app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-codesign", action="store_true", help="skip ad-hoc codesign")
    args = p.parse_args(argv)
    app = build_app(codesign_enabled=not args.no_codesign)
    print(f"build_app: wrote {app}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
