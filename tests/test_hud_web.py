import pathlib

import pytest

HUD = pathlib.Path(__file__).resolve().parents[1] / "veronica" / "ui" / "hud" / "index.html"
SCREENSHOT_DIR = pathlib.Path(__file__).resolve().parents[1] / ".superpowers"


@pytest.mark.live
def test_hud_dom_and_canvas_react():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 400, "height": 240})
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined")
        # Ensure at least one animation frame has rendered before grabbing the
        # baseline snapshot, otherwise the canvas can still be blank.
        page.wait_for_timeout(100)
        idle_px = page.evaluate("document.getElementById('orb').toDataURL()")
        page.evaluate("window.hud.push({kind:'state', payload:'listening'})")
        page.evaluate("window.hud.push({kind:'heard', payload:'what time is it'})")
        page.evaluate("window.hud.push({kind:'state', payload:'speaking'})")
        page.evaluate("window.hud.push({kind:'voice', payload:{step_ms:50, levels:[1,1,1,1,1,1,1,1,1,1]}})")
        page.evaluate("window.hud.push({kind:'sentence', payload:'It is noon.'})")
        page.evaluate("window.hud.push({kind:'tool', payload:{summary:'Open Safari', decision:'auto'}})")
        page.wait_for_function("document.querySelector('#reply .msg').textContent === 'It is noon.'", timeout=3000)
        assert page.inner_text("#heard .msg") == "what time is it"
        assert page.inner_text("#tool .msg") == "Open Safari"
        assert "auto" in page.get_attribute("#tool .badge", "class")
        page.wait_for_timeout(120)
        speaking_px = page.evaluate("document.getElementById('orb').toDataURL()")
        assert speaking_px != idle_px
        st = page.evaluate("window.hud.state()")
        assert st["state"] == "speaking" and st["reply"] == "It is noon."
        browser.close()


@pytest.mark.live
def test_hud_robust_to_bad_events_and_clear():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 400, "height": 240})
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined")
        page.wait_for_timeout(100)

        # Malformed / unknown events must not throw or wedge the typewriter.
        page.evaluate("window.hud.push({kind:'sentence'})")
        page.evaluate("window.hud.push({kind:'tool', payload:null})")
        page.evaluate("window.hud.push({kind:'nope'})")

        long_sentence = "This is a perfectly valid sentence that should still type out fully."
        page.evaluate(
            "window.hud.push({kind:'sentence', payload:" + repr(long_sentence) + "})"
        )
        page.wait_for_function(
            "document.querySelector('#reply .msg').textContent === " + repr(long_sentence),
            timeout=3000,
        )

        # A fresh turn (state:listening) must cancel the in-flight typewriter
        # so it doesn't keep writing the old sentence into the cleared reply.
        page.evaluate("window.hud.push({kind:'state', payload:'listening'})")
        page.evaluate("window.hud.push({kind:'state', payload:'speaking'})")
        page.evaluate(
            "window.hud.push({kind:'sentence', payload:'This is a long sentence that keeps typing.'})"
        )
        page.evaluate("window.hud.push({kind:'state', payload:'listening'})")
        page.wait_for_timeout(600)
        assert page.inner_text("#reply .msg") == ""

        # Tool summaries longer than 60 chars are truncated with an ellipsis.
        summary_70 = "x" * 70
        page.evaluate(
            "window.hud.push({kind:'tool', payload:{summary:" + repr(summary_70) + ", decision:'auto'}})"
        )
        rendered = page.inner_text("#tool .msg")
        assert rendered.endswith("…")
        assert len(rendered) == 60

        browser.close()


@pytest.mark.live
def test_long_reply_and_followup_stay_in_card():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 400, "height": 240})
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined")
        page.wait_for_timeout(100)

        page.evaluate("window.hud.push({kind:'heard', payload:'tell me a long story about the weather'})")
        sentences = [
            "This is a long first sentence about the weather that goes on for quite a while indeed.",
            "This is a long second sentence about the weather that goes on for quite a while indeed.",
            "This is a long third sentence about the weather that goes on for quite a while indeed.",
        ]
        for s in sentences:
            page.evaluate("window.hud.push({kind:'sentence', payload:" + repr(s) + "})")
        page.wait_for_function(
            "document.querySelector('#reply .msg').textContent.length > 0", timeout=3000
        )
        page.wait_for_timeout(2000)  # let the typewriter catch up

        tool_box = page.eval_on_selector("#tool", "el => el.getBoundingClientRect()")
        heard_box = page.eval_on_selector("#heard", "el => el.getBoundingClientRect()")
        assert tool_box["bottom"] <= 220
        assert heard_box["top"] >= 0

        # A second 'heard' (a follow-up) must clear the previous reply text
        # immediately, before any new sentences arrive.
        page.evaluate("window.hud.push({kind:'heard', payload:'and now a follow-up question'})")
        assert page.inner_text("#reply .msg") == ""

        browser.close()


@pytest.mark.live
def test_confirm_hint_appears_and_clears():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 400, "height": 240})
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined")
        page.wait_for_timeout(100)

        page.evaluate(
            "window.hud.push({kind:'tool', payload:{summary:'Bash: rm x', decision:'ask', timeout_ms:8000}})"
        )
        assert page.inner_text("#hint .msg") == 'say "yes" or "no"'

        page.evaluate(
            "window.hud.push({kind:'tool', payload:{summary:'Bash: rm x', decision:'allowed'}})"
        )
        assert page.inner_text("#hint .msg") == ""

        browser.close()


@pytest.mark.live
def test_orb_screenshot():
    from playwright.sync_api import sync_playwright

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 400, "height": 240}, device_scale_factor=2)
        page.on("pageerror", lambda exc: errors.append(exc))
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined")
        page.wait_for_timeout(100)

        page.evaluate("window.hud.push({kind:'state', payload:'speaking'})")
        page.evaluate(
            "window.hud.push({kind:'voice', payload:{step_ms:50, "
            "levels:[0.2,0.5,0.8,1,0.9,0.6,0.3,0.7,1,0.5,0.2,0.6,0.9,0.4,0.1]}})"
        )
        page.wait_for_timeout(400)

        page.locator("#orb").screenshot(path=str(SCREENSHOT_DIR / "holo-orb.png"))

        assert not errors, f"page errors: {errors}"
        browser.close()


@pytest.mark.live
def test_hud_setvisible_does_not_double_schedule_raf():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 400, "height": 240})
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined")
        page.wait_for_timeout(100)

        # Instrument requestAnimationFrame to count calls over a fixed window.
        page.evaluate(
            """
            () => {
              window.__rafCount = 0;
              const orig = window.requestAnimationFrame.bind(window);
              window.requestAnimationFrame = (cb) => {
                window.__rafCount++;
                return orig(cb);
              };
            }
            """
        )

        page.evaluate("window.__rafCount = 0")
        page.wait_for_timeout(400)
        baseline = page.evaluate("window.__rafCount")

        # Rapid hide/show must not leave two rAF loops running concurrently.
        page.evaluate("window.hud.setVisible(false)")
        page.evaluate("window.hud.setVisible(true)")

        page.evaluate("window.__rafCount = 0")
        page.wait_for_timeout(400)
        toggled = page.evaluate("window.__rafCount")

        assert toggled <= baseline * 1.3, f"toggled={toggled} baseline={baseline}"

        browser.close()
