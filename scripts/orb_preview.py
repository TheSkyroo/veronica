"""Screenshot the HUD orb in every state (full, mini, icon-on-black).

Loads veronica/ui/hud/index.html in headless Chromium (Playwright), drives
each state the way the orchestrator would (mic/voice/tool events) and writes
PNGs of the #orb canvas. A script for eyeballing the renderer, not a test.

    uv run python scripts/orb_preview.py [--out .superpowers/tasks/holo-preview] [--dpr 2]
"""
import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
HUD = REPO / "veronica" / "ui" / "hud" / "index.html"
STATES = ["idle", "listening", "thinking", "speaking", "confirming", "error", "warming"]
VOICE = [0.2, 0.5, 0.8, 1, 0.9, 0.6, 0.3, 0.7, 1, 0.5, 0.2, 0.6, 0.9, 0.4, 0.1] * 20

BLACK = "document.body.style.background='#000'; document.getElementById('card').style.background='#000'"


def drive(page, state, settle_ms):
    page.evaluate("s => window.hud.push({kind:'state', payload:s})", state)
    if state == "listening":
        for _ in range(8):
            page.evaluate("window.hud.push({kind:'mic', payload:0.75})")
            page.wait_for_timeout(60)
    if state == "speaking":
        page.evaluate("v => window.hud.push({kind:'voice', payload:{step_ms:50, levels:v}})", VOICE)
    if state == "confirming":
        page.evaluate("window.hud.push({kind:'tool', payload:{summary:'x', decision:'ask', timeout_ms:60000}})")
    page.wait_for_timeout(settle_ms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / ".superpowers" / "tasks" / "holo-preview"))
    ap.add_argument("--dpr", type=float, default=2.0)
    ap.add_argument("--states", default=",".join(STATES))
    ap.add_argument("--no-mini", action="store_true")
    ap.add_argument("--no-icon", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("orb_preview: playwright not installed", file=sys.stderr)
        return 1
    states = [s for s in args.states.split(",") if s]

    with sync_playwright() as p:
        browser = p.chromium.launch()
        errors = []
        # full mode, 170 px canvas on black at the given dpr
        page = browser.new_page(viewport={"width": 540, "height": 300}, device_scale_factor=args.dpr)
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined && window.__hud !== undefined")
        page.evaluate(BLACK)
        for state in states:
            drive(page, state, 300 if state == "error" else 900)
            page.locator("#orb").screenshot(path=str(out / f"{state}.png"))
            if state == "error":
                page.wait_for_timeout(900)
                page.locator("#orb").screenshot(path=str(out / "error-settled.png"))
        page.evaluate("window.hud.push({kind:'state', payload:'idle'})")
        page.wait_for_timeout(900)
        page.locator("#orb").screenshot(path=str(out / "idle-again.png"))
        page.screenshot(path=str(out / "card-idle.png"))
        page.close()

        if not args.no_mini:
            page = browser.new_page(viewport={"width": 400, "height": 72}, device_scale_factor=5)
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            page.goto(HUD.as_uri())
            page.wait_for_function("window.hud !== undefined")
            page.evaluate("window.hud.setMode('mini')")
            page.evaluate(BLACK)
            for state in states:
                drive(page, state, 300 if state == "error" else 900)
                page.locator("#orb").screenshot(path=str(out / f"mini-{state}.png"))
            page.close()

        if not args.no_icon:
            page = browser.new_page(viewport={"width": 1024, "height": 1024}, device_scale_factor=1)
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            page.goto(HUD.as_uri() + "?icon=1")
            page.wait_for_timeout(2200)
            page.screenshot(path=str(out / "icon-1024.png"), omit_background=True)
            page.evaluate("document.body.style.background='#000'")
            page.screenshot(path=str(out / "icon-1024-on-black.png"))
            page.close()
        browser.close()
    if errors:
        print("orb_preview: page errors:", *errors, sep="\n  ", file=sys.stderr)
        return 1
    print(f"orb_preview: wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
