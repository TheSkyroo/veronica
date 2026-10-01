.PHONY: app icon test run

# Windows has no `make` out of the box: each target is a single `uv run`
# command, so run it directly in PowerShell/cmd (e.g. `uv run python
# scripts/build_app.py`), or use make from Git Bash/MSYS2/WSL if installed.

# Render assets/Veronica.ico from the HUD orb (headless Playwright if it's
# installed, else the Pillow-drawn orb).
icon:
	uv run python scripts/make_icon.py

# Build dist/Veronica/Veronica.exe with PyInstaller (needs the dev extra:
# `uv sync --extra dev`; uses the committed assets/Veronica.ico — run
# `make icon` first if you want to re-render it).
app:
	uv run python scripts/build_app.py

test:
	uv run pytest

# Dev run: the tray app directly in this shell, no build needed.
run:
	uv run python -m veronica
