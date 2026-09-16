.PHONY: app icon test run

# Render assets/Veronica.icns from the HUD orb (headless Playwright + iconutil).
icon:
	uv run python scripts/make_icon.py

# Build dist/Veronica.app (uses the committed assets/Veronica.icns; run
# `make icon` first if you want to re-render it).
app:
	uv run python scripts/build_app.py

test:
	uv run pytest

# Dev run: menu bar app directly in this shell, no bundle needed.
run:
	uv run python -m veronica
