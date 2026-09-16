import json

import pytest

from veronica.tools import browser as b


def _ok(text):
    return {"content": [{"type": "text", "text": text}]}


def _err(text):
    return {"content": [{"type": "text", "text": text}], "is_error": True}


@pytest.fixture
def scripts(monkeypatch):
    """Capture every AppleScript sent; reply from a queue of canned results.
    An empty queue answers 'complete' so the post-navigation readyState poll
    (which never fails a tool) ends immediately; the poll's sleep is stubbed."""
    sent = []
    replies = []

    async def fake_osascript(script):
        sent.append(script)
        return replies.pop(0) if replies else _ok("complete")

    async def no_sleep(_s):
        pass

    monkeypatch.setattr(b, "_osascript", fake_osascript)
    monkeypatch.setattr(b, "_sleep", no_sleep)
    return sent, replies


def ready_polls(sent):
    return [s for s in sent if "document.readyState" in s]


def frontmost(replies, name):
    replies.append(_ok(name))


async def test_target_browser_prefers_frontmost(scripts):
    sent, replies = scripts
    frontmost(replies, "Safari")
    assert await b.target_browser() == "Safari"
    assert "frontmost" in sent[0]


async def test_target_browser_falls_back_to_running_chrome(scripts):
    sent, replies = scripts
    frontmost(replies, "Finder")
    replies.append(_ok("true"))          # Chrome running?
    assert await b.target_browser() == "Google Chrome"
    assert 'application "Google Chrome"' in sent[1] or "Google Chrome" in sent[1]


async def test_target_browser_none_running(scripts):
    _sent, replies = scripts
    frontmost(replies, "Finder")
    replies.append(_ok("false"))         # Chrome
    replies.append(_ok("false"))         # Safari
    with pytest.raises(b.BrowserUnavailable):
        await b.target_browser()


async def test_tabs_lists_and_marks_current(scripts):
    sent, replies = scripts
    frontmost(replies, "Google Chrome")
    replies.append(_ok("2\nGitHub\thttps://github.com\nDocs\thttps://docs.example\n"))
    res = await b.browser_tabs.handler({})
    assert res["content"][0]["text"] == "1. GitHub — https://github.com\n2. * Docs — https://docs.example"
    assert 'tell application "Google Chrome"' in sent[1]


async def test_open_rejects_non_http(scripts):
    res = await b.browser_open.handler({"url": "file:///etc/passwd"})
    assert res.get("is_error") and "http" in res["content"][0]["text"]


async def test_open_new_tab_chrome(scripts):
    sent, replies = scripts
    frontmost(replies, "Google Chrome")
    replies.append(_ok("ok"))
    res = await b.browser_open.handler({"url": "https://example.com", "new_tab": True})
    assert not res.get("is_error")
    assert "make new tab" in sent[1] and "https://example.com" in sent[1]


async def test_read_caps_and_truncates(scripts):
    sent, replies = scripts
    frontmost(replies, "Safari")
    replies.append(_ok(json.dumps({"title": "T", "url": "https://x", "text": "a " * 5000})))
    res = await b.browser_read.handler({"max_chars": 100})
    text = res["content"][0]["text"]
    assert text.startswith("T\nhttps://x\n")
    assert text.endswith("…[truncated]")
    assert len(text) < 200
    assert "do JavaScript" in sent[1] and "innerText" in sent[1]


async def test_find_returns_numbered_lines(scripts):
    _sent, replies = scripts
    frontmost(replies, "Google Chrome")
    replies.append(_ok(json.dumps({"lines": [[3, "Pricing plans"], [9, "See pricing"]]})))
    res = await b.browser_find.handler({"text": "pricing"})
    assert res["content"][0]["text"] == "3: Pricing plans\n9: See pricing"


async def test_find_not_found(scripts):
    _sent, replies = scripts
    frontmost(replies, "Google Chrome")
    replies.append(_ok(json.dumps({"lines": []})))
    res = await b.browser_find.handler({"text": "zzz"})
    assert res["content"][0]["text"] == "not found"


async def test_click_reports_element(scripts):
    sent, replies = scripts
    frontmost(replies, "Google Chrome")
    replies.append(_ok(json.dumps({"clicked": "BUTTON Log in"})))
    res = await b.browser_click.handler({"target": "Log in"})
    assert res["content"][0]["text"] == "Clicked BUTTON Log in"
    assert "execute active tab" in sent[1] and "aria-label" in sent[1]


async def test_click_no_match(scripts):
    _sent, replies = scripts
    frontmost(replies, "Google Chrome")
    replies.append(_ok(json.dumps({"clicked": None})))
    res = await b.browser_click.handler({"target": "Nope"})
    assert res.get("is_error") and "no element matching" in res["content"][0]["text"]


async def test_type_sets_value_and_submits(scripts):
    sent, replies = scripts
    frontmost(replies, "Safari")
    replies.append(_ok(json.dumps({"typed": "INPUT search"})))
    res = await b.browser_type.handler({"target": "search", "text": "hello", "submit": True})
    assert res["content"][0]["text"] == "Typed into INPUT search"
    js = sent[1]
    assert "placeholder" in js and "dispatchEvent" in js and "Enter" in js


async def test_scroll_and_back(scripts):
    sent, replies = scripts
    frontmost(replies, "Google Chrome"); replies.append(_ok("ok"))
    assert not (await b.browser_scroll.handler({"direction": "bottom"})).get("is_error")
    assert "scrollTo" in sent[1]
    frontmost(replies, "Google Chrome"); replies.append(_ok("ok"))
    assert not (await b.browser_back.handler({})).get("is_error")
    assert "history.back" in sent[3]
    res = await b.browser_scroll.handler({"direction": "sideways"})
    assert res.get("is_error")


async def test_js_error_mapping(scripts):
    _sent, replies = scripts
    frontmost(replies, "Google Chrome")
    replies.append(_err("execution error: Google Chrome got an error: Executing JavaScript through AppleScript is turned off. To turn it on, from the menu bar, go to View > Developer > Allow JavaScript from Apple Events. (12)"))
    res = await b.browser_read.handler({})
    assert res.get("is_error")
    assert "Allow JavaScript from Apple Events" in res["content"][0]["text"] and "View" in res["content"][0]["text"]

    frontmost(replies, "Safari")
    replies.append(_err("execution error: Not authorized to send Apple events to Safari. (-1743)"))
    res = await b.browser_read.handler({})
    assert "Automation" in res["content"][0]["text"]


def test_js_string_is_escaped_for_applescript():
    s = b._wrap_js("Google Chrome", 'alert("x\\y")')
    assert '\\"' in s and "\\\\" in s


async def test_target_browser_reports_automation_denial(scripts):
    _sent, replies = scripts
    replies.append(_err("execution error: Not authorized to send Apple events to System Events. (-1743)"))
    with pytest.raises(b.BrowserUnavailable) as exc:
        await b.target_browser()
    assert "Automation" in str(exc.value) and "System Events" in str(exc.value)


async def test_tool_surfaces_automation_denial(scripts):
    _sent, replies = scripts
    replies.append(_err("execution error: Not authorized to send Apple events to System Events. (-1743)"))
    res = await b.browser_read.handler({})
    assert res.get("is_error") and "Automation" in res["content"][0]["text"]


async def test_click_prefers_innermost_match(scripts):
    sent, replies = scripts
    frontmost(replies, "Google Chrome")
    replies.append(_ok(json.dumps({"clicked": "BUTTON Log in"})))
    await b.browser_click.handler({"target": "Log in"})
    assert "contains(" in sent[1]


def test_wrap_js_roundtrip():
    needle = 'pri"cing\\x'  # a quote and a backslash, the two chars _q must escape
    js = f"var q=norm({json.dumps(needle)}); q"
    wrapped = b._wrap_js("Google Chrome", js)
    inner = wrapped.split(' javascript "', 1)[1]
    assert inner.endswith('"')
    unescaped = inner[:-1].replace('\\"', '"').replace("\\\\", "\\")
    assert unescaped == js


async def test_type_submit_guards_against_double_submit(scripts):
    """If a keydown handler already handled Enter (preventDefault), don't
    also call form.requestSubmit() — that would submit twice."""
    sent, replies = scripts
    frontmost(replies, "Safari")
    replies.append(_ok(json.dumps({"typed": "INPUT search"})))
    await b.browser_type.handler({"target": "search", "text": "hello", "submit": True})
    js = sent[1]
    assert "var ok=el.dispatchEvent(new KeyboardEvent('keydown',opts))" in js
    assert "if(ok&&el.form&&document.activeElement===el)" in js


async def test_open_waits_for_page_to_load(scripts):
    sent, replies = scripts
    frontmost(replies, "Google Chrome")
    replies.append(_ok("ok"))                       # the open itself
    frontmost(replies, "Google Chrome"); replies.append(_ok("loading"))
    frontmost(replies, "Google Chrome"); replies.append(_ok("loading"))
    frontmost(replies, "Google Chrome"); replies.append(_ok("complete"))
    res = await b.browser_open.handler({"url": "https://example.com"})
    assert res["content"][0]["text"] == "Opened https://example.com"
    assert len(ready_polls(sent)) == 3


async def test_click_waits_for_page_to_load(scripts):
    sent, replies = scripts
    frontmost(replies, "Google Chrome")
    replies.append(_ok(json.dumps({"clicked": "A Docs"})))
    frontmost(replies, "Google Chrome"); replies.append(_ok("loading"))
    frontmost(replies, "Google Chrome"); replies.append(_ok("complete"))
    res = await b.browser_click.handler({"target": "Docs"})
    assert res["content"][0]["text"] == "Clicked A Docs"
    assert len(ready_polls(sent)) == 2


async def test_back_and_submit_wait_for_page_to_load(scripts):
    sent, replies = scripts
    frontmost(replies, "Safari"); replies.append(_ok("ok"))
    frontmost(replies, "Safari"); replies.append(_ok("complete"))
    assert (await b.browser_back.handler({}))["content"][0]["text"] == "Went back"
    assert len(ready_polls(sent)) == 1
    sent.clear()
    frontmost(replies, "Safari"); replies.append(_ok(json.dumps({"typed": "INPUT q"})))
    frontmost(replies, "Safari"); replies.append(_ok("complete"))
    res = await b.browser_type.handler({"target": "q", "text": "x", "submit": True})
    assert res["content"][0]["text"] == "Typed into INPUT q"
    assert len(ready_polls(sent)) == 1
    sent.clear()
    frontmost(replies, "Safari"); replies.append(_ok(json.dumps({"typed": "INPUT q"})))
    res = await b.browser_type.handler({"target": "q", "text": "x", "submit": False})
    assert not res.get("is_error")
    assert ready_polls(sent) == []                   # no submit: nothing to wait for


async def test_nav_wait_error_does_not_fail_tool(scripts):
    """A readyState poll error (page mid-navigation, JS refused) ends the
    wait but never turns the tool's own success into an error."""
    sent, replies = scripts
    frontmost(replies, "Google Chrome")
    replies.append(_ok("ok"))
    frontmost(replies, "Google Chrome"); replies.append(_ok("loading"))
    frontmost(replies, "Google Chrome"); replies.append(_err("execution error: page is unloading"))
    frontmost(replies, "Google Chrome"); replies.append(_ok("complete"))   # must not be consumed
    res = await b.browser_open.handler({"url": "https://example.com"})
    assert not res.get("is_error")
    assert res["content"][0]["text"] == "Opened https://example.com"
    assert len(ready_polls(sent)) == 2
    assert len(replies) == 2


async def test_nav_wait_gives_up_after_deadline(scripts, monkeypatch):
    sent, replies = scripts
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(b, "_sleep", fake_sleep)
    frontmost(replies, "Google Chrome")
    replies.append(_ok("ok"))
    for _ in range(40):
        frontmost(replies, "Google Chrome"); replies.append(_ok("loading"))
    res = await b.browser_open.handler({"url": "https://example.com"})
    assert res["content"][0]["text"] == "Opened https://example.com"
    n = len(ready_polls(sent))
    assert n == int(b.NAV_WAIT_S / b.NAV_POLL_S) and n < 40
    assert slept and all(s == b.NAV_POLL_S for s in slept)
