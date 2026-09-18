"""`computer` MCP tools with every framework seam faked: nothing here
posts a CGEvent, calls Vision/AX, or reads the real screen."""
import pytest

from veronica.tools import computer as c
from veronica.tools import computer_events as ev
from veronica.tools import ocr, screen
from veronica.tools.computer_events import DangerousCombo, Front
from veronica.tools.ocr import Word
from veronica.tools.screen import Geometry

FRONT = Front(app="Finder", bundle_id="com.apple.finder", window_title="Downloads", pid=42)
DIALOG = Front(app="SecurityAgent", bundle_id="com.apple.SecurityAgent", window_title="", pid=7)


def _geometry(age: float = 5.0, scale: float = 2.0, ox: float = 100.0, oy: float = 50.0) -> Geometry:
    return Geometry(
        region="screen", image_w=1568, image_h=1019, origin_x=ox, origin_y=oy,
        width_pt=784.0, height_pt=509.5, scale=scale, captured_at=1000.0 - age, window=None,
    )


def text(res: dict) -> str:
    return res["content"][0]["text"]


@pytest.fixture
def fakes(monkeypatch):
    """Recording fakes for every primitive; fresh geometry; trusted; Finder in front."""
    calls: list[tuple] = []
    state = {
        "trusted": True, "prompted": [], "front": FRONT, "secure": False,
        "geometry": _geometry(), "words": [], "slept": [],
    }

    def rec(name):
        def f(*a, **kw):
            calls.append((name, *a, *([kw] if kw else [])))
        return f

    def trusted(prompt=False):
        state["prompted"].append(prompt)
        return state["trusted"]

    async def sleep(s):
        state["slept"].append(s)

    monkeypatch.setattr(screen, "_now", lambda: 1000.0)
    for name in ("move", "click", "drag", "scroll", "type_text", "key"):
        monkeypatch.setattr(ev, name, rec(name))
    monkeypatch.setattr(ev, "accessibility_trusted", trusted)
    monkeypatch.setattr(ev, "frontmost", lambda: state["front"])
    monkeypatch.setattr(ev, "focused_is_secure", lambda: state["secure"])
    monkeypatch.setattr(screen, "load_geometry", lambda path=None: state["geometry"])
    monkeypatch.setattr(ocr, "recognize_text", lambda p, **kw: state["words"])
    monkeypatch.setattr(c, "_sleep", sleep)
    c.reset_prompt()
    return calls, state


# --- permission / geometry gates --------------------------------------------

ACTION_TOOLS = [
    (c.computer_move, {"x": 10, "y": 10}),
    (c.computer_scroll, {"x": 10, "y": 10, "dy": 5}),
    (c.computer_click, {"x": 10, "y": 10}),
    (c.computer_click_text, {"text": "Save"}),
    (c.computer_drag, {"x1": 1, "y1": 1, "x2": 2, "y2": 2}),
    (c.computer_find, {"text": "Save"}),
]
ALL_TOOLS = ACTION_TOOLS + [(c.computer_type, {"text": "hi"}), (c.computer_key, {"combo": "enter"})]


@pytest.mark.parametrize("tool,args", ALL_TOOLS)
async def test_permission_hint_when_not_trusted(fakes, tool, args):
    calls, state = fakes
    state["trusted"] = False
    res = await tool.handler(args)
    assert res.get("is_error") and text(res) == ev.PERMISSION_HINT
    assert state["prompted"] == [True]     # macOS prompt requested once
    assert calls == []


async def test_permission_prompt_only_once_per_process(fakes):
    _, state = fakes
    state["trusted"] = False
    for _ in range(3):
        res = await c.computer_key.handler({"combo": "enter"})
        assert text(res) == ev.PERMISSION_HINT
    assert state["prompted"] == [True, False, False]
    state["trusted"] = True                  # granted later: no prompt needed, works
    res = await c.computer_key.handler({"combo": "enter"})
    assert not res.get("is_error")
    assert state["prompted"][-1] is False


@pytest.mark.parametrize("tool,args", ACTION_TOOLS)
@pytest.mark.parametrize("geometry", [None, _geometry(age=121.0)])
async def test_stale_or_missing_geometry(fakes, tool, args, geometry):
    calls, state = fakes
    state["geometry"] = geometry
    state["words"] = [Word("Save", 0, 0, 10, 10, 0.9)]
    res = await tool.handler(args)
    assert res.get("is_error") and text(res) == c.STALE_HINT
    assert calls == []


@pytest.mark.parametrize("tool,args", [(c.computer_type, {"text": "hi"}), (c.computer_key, {"combo": "enter"})])
async def test_type_and_key_need_no_geometry(fakes, tool, args):
    calls, state = fakes
    state["geometry"] = None
    res = await tool.handler(args)
    assert not res.get("is_error")
    assert len(calls) == 1


async def test_geometry_at_max_age_is_still_fresh(fakes):
    _, state = fakes
    state["geometry"] = _geometry(age=screen.GEOMETRY_MAX_AGE_S)
    res = await c.computer_move.handler({"x": 0, "y": 0})
    assert not res.get("is_error")


# --- coordinates -------------------------------------------------------------

def test_coords_image_default_converts_via_geometry():
    assert c._coords({"x": 200, "y": 100}, _geometry()) == (200.0, 100.0)   # 100 + 200/2, 50 + 100/2


def test_coords_screen_space_passthrough():
    assert c._coords({"x": 200, "y": 100, "space": "screen"}, _geometry()) == (200.0, 100.0)
    assert c._coords({"x": 107, "y": 59, "space": "screen"}, _geometry()) == (107.0, 59.0)


@pytest.mark.parametrize("x,y", [(-1, 10), (10, -0.5), (1569, 10), (10, 1020), (99999, 99999)])
def test_coords_image_out_of_bounds(x, y):
    with pytest.raises(ValueError, match="outside the last screenshot"):
        c._coords({"x": x, "y": y}, _geometry())


def test_coords_image_edges_are_inside():
    assert c._coords({"x": 0, "y": 0}, _geometry()) == (100.0, 50.0)
    assert c._coords({"x": 1568, "y": 1019}, _geometry()) == (884.0, 559.5)


@pytest.mark.parametrize("x,y", [(99, 60), (110, 49), (885, 60), (110, 560), (7, 9)])
def test_coords_screen_space_outside_captured_area(x, y):
    with pytest.raises(ValueError, match="outside the last screenshot"):
        c._coords({"x": x, "y": y, "space": "screen"}, _geometry())


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", float("nan"), float("inf"), "NaN"])
def test_coords_reject_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        c._coords({"x": bad, "y": 1}, _geometry())
    with pytest.raises(ValueError, match="finite"):
        c._coords({"x": 1, "y": bad, "space": "screen"}, _geometry())


def test_coords_custom_keys_and_bad_values():
    assert c._coords({"x2": 20, "y2": 40}, _geometry(), "x2", "y2") == (110.0, 70.0)
    with pytest.raises(ValueError):
        c._coords({"x": "abc", "y": 1}, _geometry())
    with pytest.raises(ValueError):
        c._coords({"x": 1}, _geometry())


async def test_move_converts_image_to_screen(fakes):
    calls, _ = fakes
    res = await c.computer_move.handler({"x": 200, "y": 100})
    assert calls == [("move", 200.0, 100.0)]
    assert text(res) == "done — frontmost: Finder — Downloads"


async def test_move_screen_space(fakes):
    calls, _ = fakes
    await c.computer_move.handler({"x": 200, "y": 100, "space": "screen"})
    assert calls == [("move", 200.0, 100.0)]
    await c.computer_move.handler({"x": 107, "y": 59, "space": "screen"})
    assert calls[-1] == ("move", 107.0, 59.0)


OUTSIDE = "(-5, 10) is outside the last screenshot — take a new one or pick a point on it."


@pytest.mark.parametrize("tool,args", [
    (c.computer_move, {"x": -5, "y": 10}),
    (c.computer_scroll, {"x": -5, "y": 10, "dy": 5}),
    (c.computer_click, {"x": -5, "y": 10}),
    (c.computer_drag, {"x1": -5, "y1": 10, "x2": 2, "y2": 2}),
    (c.computer_drag, {"x1": 2, "y1": 2, "x2": -5, "y2": 10}),
])
async def test_positional_tools_reject_points_off_the_screenshot(fakes, tool, args):
    calls, _ = fakes
    res = await tool.handler(args)
    assert res.get("is_error") and text(res).endswith(OUTSIDE)
    assert calls == []


@pytest.mark.parametrize("bad", ["nan", "inf", 1e400])
async def test_positional_tools_reject_non_finite(fakes, bad):
    calls, _ = fakes
    res = await c.computer_click.handler({"x": bad, "y": 10})
    assert res.get("is_error") and "finite" in text(res)
    assert calls == []


async def test_screen_space_click_outside_window_capture(fakes):
    calls, state = fakes
    state["geometry"] = Geometry(
        region="window", image_w=800, image_h=600, origin_x=300, origin_y=200,
        width_pt=400, height_pt=300, scale=2.0, captured_at=995.0,
        window={"id": 1, "app": "Notes", "title": "n", "x": 300, "y": 200, "w": 400, "h": 300},
    )
    res = await c.computer_click.handler({"x": 50, "y": 50, "space": "screen"})
    assert res.get("is_error") and "outside the last screenshot" in text(res)
    res = await c.computer_click.handler({"x": 350, "y": 250, "space": "screen"})
    assert not res.get("is_error") and calls == [("click", 350.0, 250.0, "left", False)]


async def test_bad_coordinates_are_an_error(fakes):
    calls, _ = fakes
    res = await c.computer_click.handler({"x": "left", "y": 3})
    assert res.get("is_error")
    assert calls == []


# --- click / drag / scroll ---------------------------------------------------

async def test_click_defaults_and_options(fakes):
    calls, state = fakes
    res = await c.computer_click.handler({"x": 200, "y": 100})
    assert calls == [("click", 200.0, 100.0, "left", False)]
    assert text(res) == "done — frontmost: Finder — Downloads"
    assert state["slept"] == [c.SETTLE_S]
    await c.computer_click.handler({"x": 0, "y": 0, "button": "right", "double": True})
    assert calls[-1] == ("click", 100.0, 50.0, "right", True)


async def test_click_bad_button(fakes):
    calls, _ = fakes
    res = await c.computer_click.handler({"x": 0, "y": 0, "button": "side"})
    assert res.get("is_error") and "button" in text(res)
    assert calls == []


async def test_drag_converts_both_ends(fakes):
    calls, _ = fakes
    res = await c.computer_drag.handler({"x1": 20, "y1": 40, "x2": 200, "y2": 100})
    assert calls == [("drag", 110.0, 70.0, 200.0, 100.0)]
    assert text(res).startswith("done — frontmost: Finder")


async def test_scroll_negates_content_direction(fakes):
    calls, _ = fakes
    await c.computer_scroll.handler({"x": 200, "y": 100, "dx": 3, "dy": 5})
    assert calls == [("scroll", 200.0, 100.0, -3.0, -5.0)]
    await c.computer_scroll.handler({"x": 0, "y": 0, "dy": -10})
    assert calls[-1] == ("scroll", 100.0, 50.0, 0.0, 10.0)


async def test_scroll_needs_a_direction(fakes):
    calls, _ = fakes
    res = await c.computer_scroll.handler({"x": 0, "y": 0})
    assert res.get("is_error")
    assert calls == []


async def test_done_without_window_title(fakes):
    _, state = fakes
    state["front"] = Front(app="Finder", bundle_id="com.apple.finder", window_title="", pid=1)
    res = await c.computer_move.handler({"x": 0, "y": 0})
    assert text(res) == "done — frontmost: Finder"


# --- find / click_text -------------------------------------------------------

WORDS = [
    Word("Cancel", 700, 420, 70, 22, 0.97),
    Word("Save", 782, 420, 60, 22, 0.98),
    Word("Save As", 782, 460, 90, 22, 0.91),
]


async def test_find_lists_matches_with_integer_centers(fakes, monkeypatch):
    _, state = fakes
    state["words"] = WORDS
    seen = {}

    def recognize(path, **kw):
        seen.update(path=path, **kw)
        return WORDS

    monkeypatch.setattr(ocr, "recognize_text", recognize)
    res = await c.computer_find.handler({"text": "save"})
    assert text(res) == (
        "1. 'Save' at (812, 431) size 60×22 (conf 0.98)\n"
        "2. 'Save As' at (827, 471) size 90×22 (conf 0.91)"
    )
    assert seen["path"] == screen.latest_screenshot_path()
    assert seen["image_size"] == (1568, 1019)   # no sips spawn to size the PNG


async def test_find_not_found_and_cap(fakes):
    _, state = fakes
    state["words"] = WORDS
    assert text(await c.computer_find.handler({"text": "Quit"})) == "not found"
    state["words"] = [Word("row", 0, i * 10, 10, 10, 0.5) for i in range(15)]
    lines = text(await c.computer_find.handler({"text": "row"})).splitlines()
    assert len(lines) == c.FIND_MAX and lines[-1].startswith("10. ")


async def test_find_requires_text(fakes):
    res = await c.computer_find.handler({"text": "  "})
    assert res.get("is_error")


async def test_click_text_clicks_match_center(fakes):
    calls, state = fakes
    state["words"] = WORDS
    res = await c.computer_click_text.handler({"text": "Save"})
    # center (812, 431) image px → (100 + 406, 50 + 215.5)
    assert calls == [("click", 506.0, 265.5, "left", False)]
    assert text(res) == "done — frontmost: Finder — Downloads"


async def test_click_text_index_and_double(fakes):
    calls, state = fakes
    state["words"] = WORDS
    await c.computer_click_text.handler({"text": "Save", "index": 1, "double": True})
    assert calls == [("click", 100 + 827 / 2, 50 + 471 / 2, "left", True)]


async def test_click_text_no_match_and_bad_index(fakes):
    calls, state = fakes
    state["words"] = WORDS
    res = await c.computer_click_text.handler({"text": "Quit"})
    assert res.get("is_error") and text(res) == "no match for 'Quit'"
    res = await c.computer_click_text.handler({"text": "Save", "index": 5})
    assert res.get("is_error") and "2 match" in text(res)
    assert calls == []


@pytest.mark.parametrize("target", ["Allow", "always allow", "OK", "Open System Settings", "continue", "Install", "TRUST"])
async def test_click_text_refuses_system_dialog_buttons(fakes, target):
    calls, state = fakes
    state["front"] = DIALOG
    state["words"] = [Word(target, 0, 0, 10, 10, 0.9)]
    res = await c.computer_click_text.handler({"text": target})
    assert res.get("is_error") and text(res) == c.DIALOG_HINT
    assert calls == []


@pytest.mark.parametrize("query", ["Allo", "Alow", "always", "llow", "ALLOW ", "ok"])
async def test_click_text_refuses_when_the_matched_label_is_allow(fakes, query):
    """OCR matching is contains/fuzzy: a near-miss query must not slip
    through to the Allow button — the label actually clicked is checked."""
    calls, state = fakes
    state["front"] = DIALOG
    state["words"] = [Word("Allow", 0, 0, 10, 10, 0.9), Word("Always Allow", 50, 0, 10, 10, 0.9),
                      Word("OK", 100, 0, 10, 10, 0.9), Word("Don't Allow", 150, 0, 10, 10, 0.9)]
    res = await c.computer_click_text.handler({"text": query})
    assert res.get("is_error") and text(res) == c.DIALOG_HINT, query
    assert calls == []


async def test_click_text_near_miss_resolving_to_dont_allow_still_clicks(fakes):
    calls, state = fakes
    state["front"] = DIALOG
    state["words"] = [Word("Don't Allow", 0, 0, 10, 10, 0.9), Word("Allow", 50, 0, 10, 10, 0.9)]
    res = await c.computer_click_text.handler({"text": "Allo"})     # first match is Don't Allow
    assert not res.get("is_error") and calls == [("click", 102.5, 52.5, "left", False)]


async def test_click_text_index_into_allow_is_refused(fakes):
    calls, state = fakes
    state["front"] = DIALOG
    state["words"] = [Word("Don't Allow", 0, 0, 10, 10, 0.9), Word("Always Allow", 50, 0, 10, 10, 0.9)]
    res = await c.computer_click_text.handler({"text": "allow", "index": 1})
    assert res.get("is_error") and text(res) == c.DIALOG_HINT
    assert calls == []


def test_dialog_target_labels():
    assert c._dialog_target("Allow") and c._dialog_target("  always   ALLOW ") and c._dialog_target("Always Allow on this Mac")
    assert c._dialog_target("OK") and c._dialog_target("Open System Settings")
    assert not c._dialog_target("Don't Allow") and not c._dialog_target("Cancel") and not c._dialog_target("Allow once?")


async def test_click_refused_on_system_dialog(fakes):
    calls, state = fakes
    state["front"] = DIALOG
    res = await c.computer_click.handler({"x": 10, "y": 10})
    assert res.get("is_error") and text(res) == c.DIALOG_HINT
    assert calls == []
    state["front"] = Front(app="System Settings", bundle_id="com.apple.systempreferences",
                           window_title="Privacy & Security", pid=3)
    res = await c.computer_click.handler({"x": 10, "y": 10})
    assert res.get("is_error") and text(res) == c.DIALOG_HINT
    assert calls == []


async def test_click_text_allows_other_buttons_in_dialog_and_allow_elsewhere(fakes):
    calls, state = fakes
    state["front"] = DIALOG
    state["words"] = [Word("Don't Allow", 0, 0, 10, 10, 0.9), Word("Allow", 50, 0, 10, 10, 0.9)]
    res = await c.computer_click_text.handler({"text": "Don't Allow"})
    assert not res.get("is_error") and len(calls) == 1
    state["front"] = FRONT
    res = await c.computer_click_text.handler({"text": "Allow"})
    assert not res.get("is_error") and len(calls) == 2


# --- type / key --------------------------------------------------------------

async def test_type_text_and_submit(fakes):
    calls, _ = fakes
    res = await c.computer_type.handler({"text": "hello"})
    assert calls == [("type_text", "hello")]
    assert text(res) == "done — frontmost: Finder — Downloads"
    await c.computer_type.handler({"text": "world", "submit": True})
    assert calls[1:] == [("type_text", "world"), ("key", "enter")]


async def test_type_refuses_secure_field(fakes):
    calls, state = fakes
    state["secure"] = True
    res = await c.computer_type.handler({"text": "hunter2"})
    assert res.get("is_error") and text(res) == c.SECURE_HINT
    assert calls == []


@pytest.mark.parametrize("combo", ["enter", "Return", "space", "spacebar", "ENTER"])
async def test_key_refuses_accept_keys_on_system_dialog(fakes, combo):
    calls, state = fakes
    state["front"] = DIALOG
    res = await c.computer_key.handler({"combo": combo})
    assert res.get("is_error") and text(res) == c.DIALOG_HINT
    assert calls == []


@pytest.mark.parametrize("combo", ["esc", "tab", "cmd+enter", "shift+space"])
async def test_key_other_keys_still_allowed_on_system_dialog(fakes, combo):
    calls, state = fakes
    state["front"] = DIALOG
    res = await c.computer_key.handler({"combo": combo})
    assert not res.get("is_error")
    assert calls == [("key", combo)]


async def test_key_enter_allowed_outside_dialogs(fakes):
    calls, _ = fakes
    res = await c.computer_key.handler({"combo": "enter"})
    assert not res.get("is_error") and calls == [("key", "enter")]


async def test_type_submit_refused_on_system_dialog(fakes):
    calls, state = fakes
    state["front"] = DIALOG
    res = await c.computer_type.handler({"text": "hi", "submit": True})
    assert res.get("is_error") and text(res) == c.DIALOG_HINT
    assert calls == []
    res = await c.computer_type.handler({"text": "hi"})     # plain typing is still fine
    assert not res.get("is_error") and calls == [("type_text", "hi")]


async def test_type_requires_text(fakes):
    calls, _ = fakes
    res = await c.computer_type.handler({"text": ""})
    assert res.get("is_error")
    assert calls == []


async def test_key_presses_combo(fakes):
    calls, _ = fakes
    res = await c.computer_key.handler({"combo": "cmd+shift+s"})
    assert calls == [("key", "cmd+shift+s")]
    assert text(res) == "done — frontmost: Finder — Downloads"


async def test_key_dangerous_and_unknown(fakes, monkeypatch):
    calls, _ = fakes

    def key(combo):
        if combo == "cmd+q":
            raise DangerousCombo("refusing")
        raise ValueError("unknown key 'zz'")

    monkeypatch.setattr(ev, "key", key)
    res = await c.computer_key.handler({"combo": "cmd+q"})
    assert res.get("is_error") and text(res) == c.DANGEROUS_HINT
    res = await c.computer_key.handler({"combo": "zz"})
    assert res.get("is_error") and text(res).startswith("Unknown key: ")
    assert calls == []


async def test_key_requires_combo(fakes):
    res = await c.computer_key.handler({"combo": ""})
    assert res.get("is_error")


# --- failures / registration ---------------------------------------------------

async def test_primitive_failure_becomes_error(fakes, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("event tap failed")

    monkeypatch.setattr(ev, "click", boom)
    res = await c.computer_click.handler({"x": 0, "y": 0})
    assert res.get("is_error") and "event tap failed" in text(res)


def test_server_and_tool_names():
    assert c.COMPUTER_TOOL_NAMES == [
        "computer_move", "computer_scroll", "computer_find", "computer_click",
        "computer_click_text", "computer_drag", "computer_type", "computer_key",
    ]
    assert c.computer_server is not None
    assert "positive dy scrolls down" in c.computer_scroll.description.lower()
