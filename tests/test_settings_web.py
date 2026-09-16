"""Settings page (index.html + settings.js) in headless Chromium. Without
`window.webkit.messageHandlers.veronica` the page pushes its messages into
`window.__settings.sent`; tests answer them with `window.settings.reply`."""
import json
import pathlib

import pytest

from veronica.config import EDITABLE_SETTINGS

PAGE = pathlib.Path(__file__).resolve().parents[1] / "veronica" / "ui" / "settings" / "index.html"
TABS = ["general", "voice", "listening", "briefings", "brain", "history", "about"]


def fixture_state(**over) -> dict:
    fields = {
        name: {"kind": f.kind, "label": f.label, "help": f.help,
               "choices": list(f.choices) if f.choices else None,
               "min": f.min, "max": f.max, "restart": f.restart}
        for name, f in EDITABLE_SETTINGS.items()
    }
    state = {
        "general": {"language": "en", "start_at_login": False, "ptt_enabled": True, "hud_mode": "full",
                    "hud_hide_after_s": 3.0, "can_start_at_login": True},
        "voice": {"voice": "af_sarah", "hindi_voice": "hf_alpha", "speed": 1.0,
                  "voices": [{"id": "af_sarah", "name": "Sarah", "hindi": False},
                             {"id": "bm_george", "name": "George", "hindi": False},
                             {"id": "hf_alpha", "name": "Alpha", "hindi": True}]},
        "listening": {"followup_window_s": 6, "confirm_listen_s": 8, "vad_silence_ms": 1200,
                      "max_utterance_s": 20, "wake_min_rms": 0.01, "wake_window_s": 1.5, "wake_hop_s": 0.25,
                      "wake_phrases": ["veronica", "hey veronica"]},
        "briefings": {"briefing_enabled": False, "briefing_time": "08:00", "nudges_enabled": True, "nudge_minutes": 5},
        "brain": {"effort": "medium", "memory_enabled": True, "brain_cwd": "/Users/me"},
        "about": {"version": "0.1.0", "build": "a517483", "built_at": "2026-09-17T10:00:00+05:30", "dirty": True,
                  "describe": "Veronica 0.1.0 (a517483, 17 Sep)", "update": {"available": False, "detail": ""},
                  "updating": False, "log_path": "/tmp/veronica.log", "can_restart": True},
        "meta": {"restart_required": False, "fields": fields},
    }
    for k, v in over.items():
        state[k].update(v)
    return state


def open_page(p, state=None):
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 720, "height": 520})
    page.goto(PAGE.as_uri())
    page.wait_for_function("window.settings !== undefined && window.__settings !== undefined")
    page.evaluate("s => window.settings.state(s)", state or fixture_state())
    return browser, page


def sent(page):
    return page.evaluate("window.__settings.sent")


def reply(page, mid, result):
    page.evaluate("([id, r]) => window.settings.reply(id, r)", [mid, result])


@pytest.mark.live
def test_all_tabs_render_from_state():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser, page = open_page(p)
        assert page.locator("#tabs button").count() == 7
        for tab in TABS:
            page.evaluate(f"window.settings.select({json.dumps(tab)})")
            assert page.get_attribute("#pane", "data-tab") == tab
            assert "active" in page.get_attribute(f"#tabs button[data-tab={tab}]", "class")
            assert page.locator("#pane .row, #pane .about, #pane .history").count() > 0
        # state() also accepts a JSON string (what Python pushes is an object literal, but be lenient)
        page.evaluate("window.settings.state(JSON.stringify(window.__settings.state()))")
        assert page.evaluate("window.__settings.state().brain.effort") == "medium"

        page.evaluate("window.settings.select('general')")
        assert page.input_value("#pane select[data-key=language]") == "en"
        assert page.is_checked("#pane input[data-key=ptt_enabled]")
        page.evaluate("window.settings.select('voice')")
        opts = page.locator("#pane select[data-key=voice] option").all_text_contents()
        assert opts == ["Sarah", "George"]          # Hindi voices go to the other select
        assert page.locator("#pane select[data-key=hindi_voice] option").all_text_contents() == ["Alpha"]
        page.evaluate("window.settings.select('listening')")
        assert page.input_value("#pane input[data-key=wake_phrases]") == "veronica, hey veronica"
        assert page.input_value("#pane input[data-key=followup_window_s]") == "6"
        assert page.inner_text("#pane .row[data-key=followup_window_s] .value") == "6"
        page.evaluate("window.settings.select('about')")
        about = page.inner_text("#pane .about")
        assert "0.1.0" in about and "a517483" in about and "modified" in about.lower()
        assert page.inner_text("#pane .about .log-path") == "/tmp/veronica.log"
        assert not page.is_visible("#banner")
        browser.close()


@pytest.mark.live
def test_controls_post_set_and_reply_drives_banner():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser, page = open_page(p)
        page.evaluate("window.settings.select('brain')")
        page.select_option("#pane select[data-key=effort]", "high")
        msg = sent(page)[-1]
        assert msg["cmd"] == "set" and msg["args"] == {"section": "brain", "key": "effort", "value": "high"}
        assert page.inner_text("#pane .row[data-key=effort] .status") == "Applying…"
        reply(page, msg["id"], {"ok": True, "restart_required": True, "message": ""})
        page.wait_for_selector("#banner", state="visible")
        assert "Restart" in page.inner_text("#banner")
        assert page.inner_text("#pane .row[data-key=effort] .status") == ""

        # toggle
        page.click("#pane input[data-key=memory_enabled]")
        msg = sent(page)[-1]
        assert msg["args"] == {"section": "brain", "key": "memory_enabled", "value": False}

        # text commits on Enter (and not on every keystroke)
        before = len(sent(page))
        page.fill("#pane input[data-key=brain_cwd]", "/tmp/work")
        assert len(sent(page)) == before
        page.press("#pane input[data-key=brain_cwd]", "Enter")
        msg = sent(page)[-1]
        assert msg["args"] == {"section": "brain", "key": "brain_cwd", "value": "/tmp/work"}
        # error reply shows the message and reverts the control to state
        reply(page, msg["id"], {"ok": False, "message": "no such folder", "restart_required": True})
        assert page.inner_text("#pane .row[data-key=brain_cwd] .status") == "no such folder"
        assert page.input_value("#pane input[data-key=brain_cwd]") == "/Users/me"

        # range: label follows drag, message on change; list field → array
        page.evaluate("window.settings.select('listening')")
        page.evaluate("""() => {
          const el = document.querySelector('#pane input[data-key=confirm_listen_s]');
          el.value = '12'; el.dispatchEvent(new Event('input', {bubbles: true}));
          el.dispatchEvent(new Event('change', {bubbles: true}));
        }""")
        assert page.inner_text("#pane .row[data-key=confirm_listen_s] .value") == "12"
        msg = sent(page)[-1]
        assert msg["args"] == {"section": "listening", "key": "confirm_listen_s", "value": 12}
        page.fill("#pane input[data-key=wake_phrases]", "veronica, jarvis ,, ")
        page.press("#pane input[data-key=wake_phrases]", "Tab")
        msg = sent(page)[-1]
        assert msg["args"] == {"section": "listening", "key": "wake_phrases", "value": ["veronica", "jarvis"]}

        # general: language select + voice speed + test voice
        page.evaluate("window.settings.select('general')")
        page.select_option("#pane select[data-key=language]", "hi")
        assert sent(page)[-1]["args"] == {"section": "general", "key": "language", "value": "hi"}
        page.evaluate("window.settings.select('voice')")
        page.select_option("#pane select[data-key=voice]", "bm_george")
        assert sent(page)[-1]["args"] == {"section": "voice", "key": "voice", "value": "bm_george"}
        page.click("#pane button[data-cmd=test_voice][data-lang=hi]")
        assert sent(page)[-1] == {"id": sent(page)[-1]["id"], "cmd": "test_voice", "args": {"lang": "hi"}}

        # restart banner button → restart
        page.click("#banner button")
        assert sent(page)[-1]["cmd"] == "restart"
        # a state push with restart_required false clears the banner
        page.evaluate("s => window.settings.state(s)", fixture_state())
        assert not page.is_visible("#banner")
        page.evaluate("s => window.settings.state(s)", fixture_state(meta={"restart_required": True}))
        assert page.is_visible("#banner")
        browser.close()


@pytest.mark.live
def test_history_search_forget_and_clear():
    from playwright.sync_api import sync_playwright

    items = [{"id": 2, "ts": "2026-09-17T14:05:00", "heard": "what time is it", "reply": "It is two."},
             {"id": 1, "ts": "2026-09-16T09:30:00", "heard": "open safari", "reply": "Opening Safari."}]
    with sync_playwright() as p:
        browser, page = open_page(p)
        page.evaluate("window.settings.select('history')")
        msg = sent(page)[-1]
        assert msg["cmd"] == "history" and msg["args"] == {"query": "", "limit": 200, "offset": 0}
        reply(page, msg["id"], {"ok": True, "items": [], "message": ""})
        assert page.inner_text("#pane .history .empty") == "Nothing yet."

        # search is debounced: three keystrokes → one request
        n = len(sent(page))
        page.type("#pane input.search", "saf", delay=20)
        assert len(sent(page)) == n
        page.wait_for_timeout(350)
        assert len(sent(page)) == n + 1
        msg = sent(page)[-1]
        assert msg["cmd"] == "history" and msg["args"]["query"] == "saf"
        reply(page, msg["id"], {"ok": True, "items": items, "message": ""})
        rows = page.locator("#pane .history .turn")
        assert rows.count() == 2
        first = rows.nth(0).inner_text()
        assert "14:05" in first and "You: what time is it" in first and "Veronica: It is two." in first

        # Forget → forget_turn, then the list reloads
        rows.nth(1).locator("button.forget").click()
        msg = sent(page)[-1]
        assert msg["cmd"] == "forget_turn" and msg["args"] == {"id": 1}
        reply(page, msg["id"], {"ok": True, "message": ""})
        msg = sent(page)[-1]
        assert msg["cmd"] == "history"
        reply(page, msg["id"], {"ok": True, "items": items[:1], "message": ""})
        assert page.locator("#pane .history .turn").count() == 1

        # Clear all → in-page confirm; No cancels, Yes posts clear_history
        page.click("#pane button.clear")
        assert page.is_visible("#pane .confirm")
        n = len(sent(page))
        page.click("#pane .confirm button.no")
        assert not page.is_visible("#pane .confirm") and len(sent(page)) == n
        page.click("#pane button.clear")
        page.click("#pane .confirm button.yes")
        msg = sent(page)[-1]
        assert msg["cmd"] == "clear_history" and not page.is_visible("#pane .confirm")
        reply(page, msg["id"], {"ok": True, "count": 1, "message": ""})
        msg = sent(page)[-1]
        assert msg["cmd"] == "history"
        reply(page, msg["id"], {"ok": True, "items": [], "message": ""})
        assert page.inner_text("#pane .history .empty") == "Nothing yet."
        browser.close()


@pytest.mark.live
def test_about_buttons_and_status_line():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser, page = open_page(p)
        page.evaluate("window.settings.select('about')")
        page.click("#pane button[data-cmd=check_update]")
        msg = sent(page)[-1]
        assert msg["cmd"] == "check_update"
        assert page.inner_text("#pane .about .status") == "Checking…"
        reply(page, msg["id"], {"ok": True, "available": True, "kind": "remote", "detail": "2 commits behind", "message": ""})
        assert page.inner_text("#pane .about .status") == "2 commits behind"
        reply_id = msg["id"]
        page.click("#pane button[data-cmd=update_now]")
        msg = sent(page)[-1]
        assert msg["cmd"] == "update_now" and msg["id"] != reply_id
        reply(page, msg["id"], {"ok": False, "message": "Busy, try again in a moment."})
        assert page.inner_text("#pane .about .status") == "Busy, try again in a moment."
        for cmd in ("restart", "open_logs"):
            page.click(f"#pane button[data-cmd={cmd}]")
            assert sent(page)[-1]["cmd"] == cmd
        # updating state disables Update & restart
        page.evaluate("s => window.settings.state(s)", fixture_state(about={"updating": True}))
        assert page.is_disabled("#pane button[data-cmd=update_now]")
        # keyboard: tabs are real buttons — Enter on a focused tab selects it
        page.focus("#tabs button[data-tab=voice]")
        page.keyboard.press("Enter")
        assert page.get_attribute("#pane", "data-tab") == "voice"
        browser.close()


@pytest.mark.live
def test_state_push_keeps_typing_and_focus_without_posting():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser, page = open_page(p)
        page.evaluate("window.settings.select('brain')")
        page.click("#pane input[data-key=brain_cwd]")
        page.keyboard.press("End")
        page.keyboard.type("/proj")
        n = len(sent(page))
        page.evaluate("s => window.settings.state(s)", fixture_state())
        assert page.input_value("#pane input[data-key=brain_cwd]") == "/Users/me/proj"
        assert page.evaluate("document.activeElement === document.querySelector('#pane input[data-key=brain_cwd]')")
        assert page.evaluate("document.activeElement.selectionStart") == len("/Users/me/proj")
        assert len(sent(page)) == n                      # the half-typed value was not posted
        page.keyboard.press("Enter")                     # committing afterwards still works
        assert sent(page)[-1]["args"] == {"section": "brain", "key": "brain_cwd", "value": "/Users/me/proj"}
        # a focused select survives a push too
        page.focus("#pane select[data-key=effort]")
        page.evaluate("s => window.settings.state(s)", fixture_state())
        assert page.evaluate("document.activeElement === document.querySelector('#pane select[data-key=effort]')")
        browser.close()


@pytest.mark.live
def test_every_control_has_an_accessible_name():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser, page = open_page(p)
        assert page.get_attribute("#tabs", "role") == "tablist"
        for tab in TABS:
            page.evaluate(f"window.settings.select({json.dumps(tab)})")
            assert page.get_attribute(f"#tabs button[data-tab={tab}]", "role") == "tab"
            assert page.get_attribute(f"#tabs button[data-tab={tab}]", "aria-selected") == "true"
            assert page.locator("#tabs button[aria-selected=true]").count() == 1
            unnamed = page.evaluate("""() => {
              const out = [];
              for (const c of document.querySelectorAll('#pane input, #pane select, #pane button')) {
                let name = c.getAttribute('aria-label') || c.textContent.trim();
                const by = c.getAttribute('aria-labelledby');
                if (by) name = (document.getElementById(by) || {}).textContent || '';
                if (!name.trim()) out.push(c.outerHTML.slice(0, 80));
              }
              return out;
            }""")
            assert unnamed == [], (tab, unnamed)
        page.evaluate("window.settings.select('listening')")
        assert page.get_by_label("Wake phrases", exact=False).input_value() == "veronica, hey veronica"
        assert page.get_by_label("Search history").count() == 0
        page.evaluate("window.settings.select('history')")
        assert page.get_by_label("Search history").count() == 1
        browser.close()


@pytest.mark.live
def test_forget_error_is_shown_and_about_status_resets():
    from playwright.sync_api import sync_playwright

    items = [{"id": 5, "ts": "2026-09-17T14:05:00", "heard": "hi", "reply": "Hello."}]
    with sync_playwright() as p:
        browser, page = open_page(p)
        page.evaluate("window.settings.select('history')")
        reply(page, sent(page)[-1]["id"], {"ok": True, "items": items, "message": ""})
        page.click("#pane .turn button.forget")
        reply(page, sent(page)[-1]["id"], {"ok": False, "message": "That one's already gone."})
        msg = sent(page)[-1]
        assert msg["cmd"] == "history"
        reply(page, msg["id"], {"ok": True, "items": items, "message": ""})
        assert page.inner_text("#pane .history .notice") == "That one's already gone."
        assert page.locator("#pane .history .turn").count() == 1
        page.click("#pane .turn button.forget")            # next round trip clears the notice
        reply(page, sent(page)[-1]["id"], {"ok": True, "message": ""})
        reply(page, sent(page)[-1]["id"], {"ok": True, "items": [], "message": ""})
        assert page.locator("#pane .history .notice").count() == 0

        page.evaluate("window.settings.select('about')")
        page.click("#pane button[data-cmd=update_now]")
        reply(page, sent(page)[-1]["id"], {"ok": True, "message": "Updating, back in a moment."})
        assert page.inner_text("#pane .about .status") == "Updating, back in a moment."
        page.evaluate("s => window.settings.state(s)", fixture_state(about={"updating": True}))
        assert page.inner_text("#pane .about .status") == ""          # the flag flipped: stale text gone
        page.evaluate("s => window.settings.state(s)",
                      fixture_state(about={"updating": False, "update": {"available": True, "detail": "The update failed, check the log."}}))
        assert page.inner_text("#pane .about .status") == "The update failed, check the log."
        page.click("#pane button[data-cmd=check_update]")
        assert page.inner_text("#pane .about .status") == "Checking…"
        page.evaluate("s => window.settings.state(s)",
                      fixture_state(about={"update": {"available": True, "detail": "The update failed, check the log."}}))
        assert page.inner_text("#pane .about .status") == "Checking…"  # unchanged detail: reply text stays
        browser.close()
