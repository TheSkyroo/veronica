import pathlib

import pytest

HUD = pathlib.Path(__file__).resolve().parents[1] / "veronica" / "ui" / "hud" / "index.html"
SCREENSHOT_DIR = pathlib.Path(__file__).resolve().parents[1] / ".superpowers"


@pytest.mark.live
def test_hud_dom_and_canvas_react():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 540, "height": 300})
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
        page = browser.new_page(viewport={"width": 540, "height": 300})
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
        page = browser.new_page(viewport={"width": 540, "height": 300})
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

        card_box = page.eval_on_selector("#card", "el => el.getBoundingClientRect()")
        reply_box = page.eval_on_selector("#reply", "el => el.getBoundingClientRect()")
        heard_box = page.eval_on_selector("#heard", "el => el.getBoundingClientRect()")
        assert reply_box["bottom"] <= card_box["bottom"]
        assert heard_box["top"] >= card_box["top"]

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
        page = browser.new_page(viewport={"width": 540, "height": 300})
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined")
        page.wait_for_timeout(100)

        page.evaluate("window.hud.push({kind:'sentence', payload:'Something before.'})")
        page.wait_for_function(
            "document.querySelector('#reply .msg').textContent === 'Something before.'",
            timeout=3000,
        )

        # The question is spoken (and rendered) as a 'prompt' event, distinct
        # from the reply row; the 'tool' ask event (carrying summary/detail
        # and the countdown timeout) follows once the question has been
        # spoken. The status label ("Say yes or no") covers the "how to
        # answer" part, so the #hint row is not a duplicate of it.
        page.evaluate("window.hud.push({kind:'state', payload:'confirming'})")
        page.evaluate(
            "window.hud.push({kind:'prompt', payload:'Fetch weather from wttr.in?'})"
        )
        page.evaluate(
            "window.hud.push({kind:'tool', payload:{summary:'Fetch weather from wttr.in', "
            "detail:'Bash: curl -s https://wttr.in', decision:'ask', timeout_ms:8000}})"
        )
        assert page.inner_text("#prompt .msg") == 'Fetch weather from wttr.in?'
        assert page.inner_text("#status .label") == 'Say yes or no'
        assert page.inner_text("#hint .msg") == 'say "yes" or "no"'
        assert page.inner_text("#tool .detail") == 'Bash: curl -s https://wttr.in'
        assert page.inner_text("#reply .msg") == 'Something before.'

        page.evaluate(
            "window.hud.push({kind:'tool', payload:{summary:'Fetch weather from wttr.in', decision:'allowed'}})"
        )
        assert page.inner_text("#hint .msg") == ""
        assert page.inner_text("#prompt .msg") == ""

        browser.close()


@pytest.mark.live
def test_orb_screenshot():
    from playwright.sync_api import sync_playwright

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 540, "height": 300}, device_scale_factor=2)
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
        page = browser.new_page(viewport={"width": 540, "height": 300})
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


@pytest.mark.live
def test_status_label_per_state():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 540, "height": 300})
        errors = []
        page.on("pageerror", lambda exc: errors.append(exc))
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined")
        page.wait_for_timeout(100)

        assert page.inner_text("#status .label") == ""

        cases = [
            ("warming", "Warming up…"),
            ("listening", "Listening…"),
            ("thinking", "Thinking…"),
            ("speaking", "Speaking"),
            ("followup", "Listening…"),
            ("confirming", "Say yes or no"),
            ("error", "Error"),
        ]
        for state, label in cases:
            page.evaluate(f"window.hud.push({{kind:'state', payload:'{state}'}})")
            assert page.inner_text("#status .label") == label, state
            assert page.get_attribute("#status", "data-state") == state

        page.evaluate("window.hud.push({kind:'state', payload:'idle'})")
        assert page.inner_text("#status .label") == ""

        assert not errors, f"page errors: {errors}"
        browser.close()


@pytest.mark.live
def test_level_bar_grows_with_mic_while_listening():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 540, "height": 300})
        errors = []
        page.on("pageerror", lambda exc: errors.append(exc))
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined")
        page.wait_for_timeout(100)

        page.evaluate("window.hud.push({kind:'state', payload:'listening'})")
        page.evaluate("window.hud.push({kind:'mic', payload:0.8})")
        page.wait_for_timeout(500)  # let the smoothed mic level catch up
        width = page.eval_on_selector("#status .level i", "el => el.getBoundingClientRect().width")
        assert width > 40, width  # bar is 120px wide; a strong mic level should fill a good chunk

        page.evaluate("window.hud.push({kind:'state', payload:'idle'})")
        page.wait_for_timeout(50)
        level_display = page.eval_on_selector(
            "#status .level", "el => getComputedStyle(el).display"
        )
        assert level_display == "none"

        assert not errors, f"page errors: {errors}"
        browser.close()


@pytest.mark.live
def test_heard_clamp_keeps_card_in_bounds():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 540, "height": 300})
        errors = []
        page.on("pageerror", lambda exc: errors.append(exc))
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined")
        page.wait_for_timeout(100)

        long_heard = "word " * 60  # ~300 chars
        page.evaluate("window.hud.push({kind:'heard', payload:" + repr(long_heard) + "})")
        page.wait_for_timeout(100)

        card_box = page.eval_on_selector("#card", "el => el.getBoundingClientRect()")
        heard_box = page.eval_on_selector("#heard", "el => el.getBoundingClientRect()")
        assert heard_box["bottom"] <= card_box["bottom"]
        assert card_box["width"] == 540 and card_box["height"] == 300

        assert not errors, f"page errors: {errors}"
        browser.close()


@pytest.mark.live
def test_partial_transcript_then_final():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 540, "height": 300})
        errors = []
        page.on("pageerror", lambda exc: errors.append(exc))
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined")
        page.wait_for_timeout(100)

        page.evaluate("window.hud.push({kind:'state', payload:'listening'})")
        page.evaluate("window.hud.push({kind:'heard_partial', payload:'what time'})")
        assert page.inner_text("#heard .msg") == "what time"
        assert "partial" in page.get_attribute("#heard .msg", "class")

        page.evaluate("window.hud.push({kind:'heard', payload:'what time is it'})")
        assert page.inner_text("#heard .msg") == "what time is it"
        assert "partial" not in (page.get_attribute("#heard .msg", "class") or "")

        # a fresh 'listening' clears any leftover partial styling/text
        page.evaluate("window.hud.push({kind:'heard_partial', payload:'stray'})")
        page.evaluate("window.hud.push({kind:'state', payload:'listening'})")
        assert page.inner_text("#heard .msg") == ""
        assert "partial" not in (page.get_attribute("#heard .msg", "class") or "")

        assert not errors, f"page errors: {errors}"
        browser.close()


@pytest.mark.live
def test_mini_mode_shows_only_orb_and_caption():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 540, "height": 300})
        errors = []
        page.on("pageerror", lambda exc: errors.append(exc))
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined")
        page.wait_for_timeout(100)

        # Full mode by default: orb + text (status/chat) all visible, caption
        # pill never shown.
        page.evaluate("window.hud.push({kind:'state', payload:'listening'})")
        page.evaluate("window.hud.push({kind:'heard', payload:'hi there'})")
        assert page.is_visible("#orb")
        assert page.is_visible("#heard")
        assert page.eval_on_selector("#status .label", "el => getComputedStyle(el).display") != "none"
        assert page.eval_on_selector("#caption", "el => getComputedStyle(el).display") == "none"

        # Switch to mini: the orb (scaled to 64px) plus the caption pill are
        # the only visible pieces; the full card's status/chat/action block
        # is hidden wholesale.
        page.evaluate("window.hud.setMode('mini')")
        assert "mini" in page.evaluate("document.body.className")
        assert page.is_visible("#orb")
        orb_box = page.eval_on_selector("#orb", "el => el.getBoundingClientRect()")
        assert round(orb_box["width"]) == 64 and round(orb_box["height"]) == 64
        assert page.eval_on_selector("#text", "el => getComputedStyle(el).display") == "none"
        # 'hi there' is still the last heard text, so the caption pill shows it.
        assert page.eval_on_selector("#caption", "el => getComputedStyle(el).display") != "none"
        assert page.inner_text("#caption .msg") == "hi there"

        # Switch back to full: everything reappears, caption hides again.
        page.evaluate("window.hud.setMode('full')")
        assert "mini" not in page.evaluate("document.body.className")
        assert page.eval_on_selector("#status .label", "el => getComputedStyle(el).display") != "none"
        assert page.eval_on_selector("#caption", "el => getComputedStyle(el).display") == "none"
        card_box = page.eval_on_selector("#card", "el => el.getBoundingClientRect()")
        assert round(card_box["width"]) == 540 and round(card_box["height"]) == 300

        assert not errors, f"page errors: {errors}"
        browser.close()


@pytest.mark.live
def test_compact_mini_caption():
    from playwright.sync_api import sync_playwright

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 400, "height": 72}, device_scale_factor=2)
        errors = []
        page.on("pageerror", lambda exc: errors.append(exc))
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined")
        page.wait_for_timeout(100)

        page.evaluate("window.hud.setMode('mini')")
        assert page.eval_on_selector("#caption", "el => getComputedStyle(el).display") == "none"

        page.evaluate("window.hud.push({kind:'state', payload:'listening'})")
        page.evaluate("window.hud.push({kind:'heard_partial', payload:'what is'})")
        assert page.eval_on_selector("#caption", "el => getComputedStyle(el).display") != "none"
        assert page.inner_text("#caption .msg") == "what is"
        assert "partial" in page.get_attribute("#caption .msg", "class")

        page.evaluate("window.hud.push({kind:'heard', payload:'what is the weather'})")
        assert page.inner_text("#caption .msg") == "what is the weather"
        assert "partial" not in (page.get_attribute("#caption .msg", "class") or "")

        page.evaluate("window.hud.push({kind:'state', payload:'speaking'})")
        page.evaluate("window.hud.push({kind:'sentence', payload:'It is sunny.'})")
        assert page.inner_text("#caption .msg") == "It is sunny."

        cap_box = page.eval_on_selector("#caption", "el => el.getBoundingClientRect()")
        assert cap_box["left"] >= 0 and cap_box["top"] >= 0
        assert cap_box["right"] <= 400 and cap_box["bottom"] <= 72

        page.locator("#card").screenshot(path=str(SCREENSHOT_DIR / "hud-compact.png"))

        page.evaluate("window.hud.push({kind:'state', payload:'confirming'})")
        page.evaluate("window.hud.push({kind:'prompt', payload:'Fetch weather via curl?'})")
        page.evaluate(
            "window.hud.push({kind:'tool', payload:{summary:'Fetch weather via curl', "
            "decision:'ask', timeout_ms:8000}})"
        )
        caption_text = page.inner_text("#caption .msg")
        assert "Fetch weather via curl?" in caption_text
        assert "say yes or no" in caption_text

        assert not errors, f"page errors: {errors}"
        browser.close()


def _rects_intersect(a, b):
    return not (a["right"] <= b["left"] or b["right"] <= a["left"]
                or a["bottom"] <= b["top"] or b["bottom"] <= a["top"])


@pytest.mark.live
def test_no_overlap_v3():
    from playwright.sync_api import sync_playwright

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 540, "height": 300}, device_scale_factor=2)
        errors = []
        page.on("pageerror", lambda exc: errors.append(exc))
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined")
        page.wait_for_timeout(100)

        long_heard = "word " * 60  # ~300 chars
        page.evaluate("window.hud.push({kind:'state', payload:'confirming'})")
        page.evaluate("window.hud.push({kind:'heard', payload:" + repr(long_heard) + "})")

        sentences = [
            "This is a long first sentence about the weather that goes on for quite a while indeed.",
            "This is a long second sentence about the weather that goes on for quite a while indeed.",
            "This is a long third sentence about the weather that goes on for quite a while indeed.",
        ]
        for s in sentences:
            page.evaluate("window.hud.push({kind:'sentence', payload:" + repr(s) + "})")
        page.wait_for_timeout(2500)  # let the typewriter catch up

        page.evaluate(
            "window.hud.push({kind:'prompt', payload:'Fetch weather from wttr.in?'})"
        )
        page.evaluate(
            "window.hud.push({kind:'tool', payload:{summary:'Fetch weather from wttr.in', "
            "detail:'Bash: curl -s https://wttr.in', decision:'ask', timeout_ms:8000}})"
        )
        page.wait_for_timeout(100)

        card_box = page.eval_on_selector("#card", "el => el.getBoundingClientRect()")
        boxes = {
            sel: page.eval_on_selector(sel, "el => el.getBoundingClientRect()")
            for sel in ("#status", "#heard", "#reply", "#action")
        }

        for sel, box in boxes.items():
            assert box["left"] >= card_box["left"] - 0.5, (sel, box, card_box)
            assert box["top"] >= card_box["top"] - 0.5, (sel, box, card_box)
            assert box["right"] <= card_box["right"] + 0.5, (sel, box, card_box)
            assert box["bottom"] <= card_box["bottom"] + 0.5, (sel, box, card_box)

        names = list(boxes)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = boxes[names[i]], boxes[names[j]]
                assert not _rects_intersect(a, b), (names[i], names[j], a, b)

        page.locator("#card").screenshot(path=str(SCREENSHOT_DIR / "hud-v3.png"))

        assert not errors, f"page errors: {errors}"
        browser.close()


ALL_STATES = ["idle", "warming", "listening", "thinking", "speaking", "followup", "confirming", "error"]


@pytest.mark.live
def test_configure_particles_reported_in_state():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 540, "height": 300})
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined")
        st = page.evaluate("window.hud.state()")
        assert st["particles"] == 4000 and st["intensity"] == 1.0
        page.evaluate("window.hud.configure({particles: 1200, intensity: 1.5})")
        st = page.evaluate("window.hud.state()")
        assert st["particles"] == 1200 and st["intensity"] == 1.5
        # clamped to the settings range; mini mode quarters the active count
        page.evaluate("window.hud.configure({particles: 99999, intensity: 9})")
        st = page.evaluate("window.hud.state()")
        assert st["particles"] == 8000 and st["intensity"] == 2.0
        page.evaluate("window.hud.setMode('mini')")
        assert page.evaluate("window.hud.state().particles") == 2000
        page.evaluate("window.hud.setMode('full')")
        assert page.evaluate("window.hud.state().particles") == 8000
        # garbage is ignored, not thrown
        page.evaluate("window.hud.configure(null); window.hud.configure({particles: 'x', intensity: NaN})")
        assert page.evaluate("window.hud.state().particles") == 8000
        browser.close()


@pytest.mark.live
def test_every_state_renders_without_errors():
    from playwright.sync_api import sync_playwright

    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 540, "height": 300})
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined")
        snaps = {}
        for state in ALL_STATES:
            page.evaluate("s => window.hud.push({kind:'state', payload:s})", state)
            if state in ("listening", "followup"):
                page.evaluate("window.hud.push({kind:'mic', payload:0.8})")
            if state == "speaking":
                page.evaluate("window.hud.push({kind:'voice', payload:{step_ms:50, levels:[1,1,1,1,1,1,1,1,1,1]}})")
            if state == "confirming":
                page.evaluate("window.hud.push({kind:'tool', payload:{summary:'x', decision:'ask', timeout_ms:5000}})")
            page.wait_for_timeout(500)
            snaps[state] = page.evaluate("document.getElementById('orb').toDataURL()")
            assert page.evaluate("window.__hud.frames") > 0
        for mode in ("mini", "full"):
            page.evaluate("m => window.hud.setMode(m)", mode)
            page.wait_for_timeout(200)
        assert not errors, f"page errors: {errors}"
        # every state has its own look
        assert len(set(snaps.values())) == len(ALL_STATES)
        browser.close()


@pytest.mark.live
def test_reduced_motion_freezes_idle_but_state_changes_still_draw():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 540, "height": 300})
        page.emulate_media(reduced_motion="reduce")
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined")
        assert page.evaluate("matchMedia('(prefers-reduced-motion: reduce)').matches")
        page.wait_for_timeout(1000)   # let the idle parameter lerp fully settle
        a = page.evaluate("document.getElementById('orb').toDataURL()")
        page.wait_for_timeout(200)
        b = page.evaluate("document.getElementById('orb').toDataURL()")
        assert a == b, "idle orb must be frozen under prefers-reduced-motion"
        page.evaluate("window.hud.push({kind:'state', payload:'thinking'})")
        page.wait_for_timeout(500)
        c = page.evaluate("document.getElementById('orb').toDataURL()")
        assert c != a, "colour/brightness changes still apply under reduced motion"
        browser.close()
