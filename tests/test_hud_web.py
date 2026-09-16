import pathlib

import pytest

HUD = pathlib.Path(__file__).resolve().parents[1] / "veronica" / "ui" / "hud" / "index.html"


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
