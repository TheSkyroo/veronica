"""`screenshot` tool and the capture pipeline with every Windows seam faked:
`screen._win32` (monitors, windows, clipboard) and `screen._grab` (the mss
pixel grab, here a plain Pillow image of the asked size). Pillow itself is
real — encoding, downscaling and the JPEG fallback run for real."""
import base64
import json
import os
import stat
import struct
import sys

import pytest

from veronica.tools import computer_events, screen

# Two monitors the way Windows reports them: a 1920×1080 primary at the
# origin and a 2560×1440 secondary to its right.
PRIMARY = {"handle": 101, "rect": (0, 0, 1920, 1080), "primary": True, "name": r"\\.\DISPLAY1"}
RIGHT = {"handle": 202, "rect": (1920, 0, 4480, 1440), "primary": False, "name": r"\\.\DISPLAY2"}
LEFT = {"handle": 303, "rect": (-2560, 0, 0, 1440), "primary": False, "name": r"\\.\DISPLAY3"}


class FakeWin32:
    def __init__(self, monitors=(PRIMARY,), windows=None, fg=0):
        self._monitors = list(monitors)
        # hwnd → dict(rect=(l, t, r, b), pid, title, cls, visible)
        self.windows = dict(windows or {})
        self.fg = fg
        self.z = list(self.windows)            # EnumWindows order, front to back
        self.paths = {}
        self.names = {}
        self.clip_seq = 1

    def monitors(self):
        return list(self._monitors)

    def virtual_screen(self):
        ls = [m["rect"] for m in self._monitors]
        x, y = min(r[0] for r in ls), min(r[1] for r in ls)
        return (x, y, max(r[2] for r in ls) - x, max(r[3] for r in ls) - y)

    def foreground_window(self):
        return self.fg

    def _w(self, hwnd):
        return self.windows.get(hwnd, {})

    def window_class(self, hwnd):
        return self._w(hwnd).get("cls", "AppWindow")

    def window_title(self, hwnd):
        return self._w(hwnd).get("title", "")

    def window_pid(self, hwnd):
        return self._w(hwnd).get("pid", 0)

    def window_visible(self, hwnd):
        return self._w(hwnd).get("visible", True)

    def window_rect(self, hwnd):
        return self._w(hwnd).get("rect")

    def top_level_windows(self):
        return list(self.z)

    def process_path(self, pid):
        return self.paths.get(pid, "")

    def app_name(self, path):
        return self.names.get(path, "")

    def clipboard_sequence(self):
        return self.clip_seq


def _window(rect, pid=100, title="Doc", cls="AppWindow", visible=True):
    return {"rect": rect, "pid": pid, "title": title, "cls": cls, "visible": visible}


class Grabs:
    """Fake `_grab`: records each rectangle and returns an image of that
    size (noise when `noisy`, so PNGs are big and JPEG has work to do)."""

    def __init__(self, noisy=False, fail=0):
        self.rects: list[tuple] = []
        self.noisy, self.fail = noisy, fail

    def __call__(self, left, top, width, height):
        self.rects.append((left, top, width, height))
        if self.fail:
            self.fail -= 1
            raise OSError("BitBlt failed")
        from PIL import Image
        if self.noisy:
            return Image.frombytes("RGB", (width, height), os.urandom(width * height * 3))
        return Image.new("RGB", (width, height), (40, 90, 160))


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch, request):
    """No test reaches real Win32 or mss; tests that need monitors/windows
    install a FakeWin32, and the pixel grab is always a fake. The `live`
    tests are the exception — the real screens are the point of them."""
    if request.node.get_closest_marker("live") is not None:
        return

    def no_win32():
        raise OSError("Win32 disabled in tests")

    def no_mss():
        raise ImportError("mss disabled in tests")

    monkeypatch.setattr(screen, "_win32", no_win32)
    monkeypatch.setattr(screen, "_mss", no_mss)
    monkeypatch.setattr(screen, "_grab", Grabs())
    monkeypatch.setattr(computer_events, "ensure_dpi_awareness", lambda: "per-monitor-v2")


@pytest.fixture
def screens(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "_screens_dir", lambda: tmp_path)
    monkeypatch.setattr(screen, "_now", lambda: 1000.0)
    return tmp_path


def use(monkeypatch, win=None, grab=None):
    win = win if win is not None else FakeWin32()
    monkeypatch.setattr(screen, "_win32", lambda: win)
    if grab is not None:
        monkeypatch.setattr(screen, "_grab", grab)
    return win


def text(res):
    return res["content"][-1]["text"]


def two_displays(front_on_right=True):
    """Primary + right monitor, with a foreground window on the right one."""
    windows = {42: _window((2000, 200, 2900, 900), title="Apple")} if front_on_right else {}
    return FakeWin32([PRIMARY, RIGHT], windows=windows, fg=42 if front_on_right else 0)


# ---- basic capture ----------------------------------------------------------

async def test_screenshot_screen_region(screens, monkeypatch):
    grab = Grabs()
    use(monkeypatch, grab=grab)
    res = await screen.screenshot.handler({"region": "screen"})
    assert not res.get("is_error"), text(res)
    assert [b["type"] for b in res["content"]] == ["image", "text"]
    assert res["content"][0]["mimeType"] == "image/png"
    data = base64.b64decode(res["content"][0]["data"])
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data == (screens / "latest.png").read_bytes()
    assert grab.rects == [(0, 0, 1920, 1080)]
    # no on-disk path is advertised to the model
    assert "saved to" not in text(res).lower() and str(screens) not in text(res)
    assert text(res) == (
        "Screenshot of the screen: 1568×882 px (screen 1920×1080 px). "
        "Coordinates you pass to computer_* tools are in these image pixels."
    )


async def test_screenshot_keeps_only_latest_png_with_0600(screens, monkeypatch):
    """Nothing accumulates — every capture overwrites the single
    latest.png (and its sidecar), never a timestamped file."""
    use(monkeypatch)
    await screen.screenshot.handler({"region": "screen"})
    await screen.screenshot.handler({"region": "screen"})
    assert sorted(p.name for p in screens.iterdir()) == ["latest.json", "latest.png"]
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(screens / "latest.png").st_mode) == 0o600
        assert stat.S_IMODE(os.stat(screens / "latest.json").st_mode) == 0o600


async def test_failed_capture_removes_stale_latest_and_sidecar(screens, monkeypatch):
    (screens / "latest.png").write_bytes(b"old")
    (screens / "latest.json").write_text("{}")
    use(monkeypatch, grab=Grabs(fail=2))
    res = await screen.screenshot.handler({})
    assert res["is_error"] and screen.DISPLAY_CHANGE_ERROR in text(res)
    assert not (screens / "latest.png").exists()
    assert not (screens / "latest.json").exists()


def test_grab_failure_is_retried_once(screens, monkeypatch):
    """Right after a monitor is (un)plugged the first grab can fail."""
    grab = Grabs(fail=1)
    use(monkeypatch, grab=grab)
    result = screen.capture_screenshot("screen")
    assert not isinstance(result, str)
    assert grab.rects == [(0, 0, 1920, 1080)] * 2


def test_capture_makes_the_process_dpi_aware_first(screens, monkeypatch):
    calls = []
    monkeypatch.setattr(computer_events, "ensure_dpi_awareness", lambda: calls.append(1))
    use(monkeypatch)
    screen.capture_screenshot("screen")
    screen.capture_all_displays()
    assert calls == [1, 1]


def test_capture_screenshot_returns_mime_and_path(screens, monkeypatch):
    use(monkeypatch)
    _data, path, mime = screen.capture_screenshot("screen")
    assert mime == "image/png" and path == screens / "latest.png"


def test_latest_screenshot_path():
    assert screen.latest_screenshot_path().name == "latest.png"


async def test_bad_region_defaults_to_screen(screens, monkeypatch):
    grab = Grabs()
    use(monkeypatch, grab=grab)
    res = await screen.screenshot.handler({"region": "everything"})
    assert not res.get("is_error")
    assert grab.rects == [(0, 0, 1920, 1080)]


# ---- downscale / encode -------------------------------------------------------

def test_downscale_keeps_aspect_and_geometry_maps_back(screens, monkeypatch):
    use(monkeypatch, FakeWin32([{**PRIMARY, "rect": (0, 0, 3000, 2000)}]))
    screen.capture_screenshot("screen")
    g = screen.load_geometry()
    assert (g.image_w, g.image_h) == (1568, 1045)
    assert screen._png_size(screens / "latest.png") == (1568, 1045)
    assert g.scale == pytest.approx(1568 / 3000)
    assert g.to_screen(1568, 1045) == (pytest.approx(3000, abs=2), pytest.approx(2000, abs=2))


def test_small_captures_are_never_enlarged(screens, monkeypatch):
    use(monkeypatch, FakeWin32([{**PRIMARY, "rect": (0, 0, 800, 600)}]))
    screen.capture_screenshot("screen")
    g = screen.load_geometry()
    assert (g.image_w, g.image_h, g.scale) == (800, 600, 1.0)
    assert g.to_screen(10, 20) == (10.0, 20.0)


def test_oversized_png_is_sent_as_jpeg_but_latest_stays_png(screens, monkeypatch):
    use(monkeypatch, FakeWin32([{**PRIMARY, "rect": (0, 0, 600, 400)}]), grab=Grabs(noisy=True))
    monkeypatch.setattr(screen, "MAX_PNG_BYTES", 100 * 1024)
    data, path, mime = screen.capture_screenshot("screen")
    assert mime == "image/jpeg" and data[:2] == b"\xff\xd8"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert sorted(p.name for p in screens.iterdir()) == ["latest.json", "latest.png"]   # no temp JPEG
    assert screen.load_geometry() is not None


def test_jpeg_still_too_big_retries_at_lower_quality(screens, monkeypatch):
    use(monkeypatch, FakeWin32([{**PRIMARY, "rect": (0, 0, 300, 200)}]), grab=Grabs(noisy=True))
    monkeypatch.setattr(screen, "MAX_PNG_BYTES", 10)
    qualities = []

    def reencode(img, quality=screen.JPEG_QUALITY):
        qualities.append(quality)
        return b"\xff\xd8" + b"q" * quality
    monkeypatch.setattr(screen, "_reencode_jpeg", reencode)
    data, _path, mime = screen.capture_screenshot("screen")
    assert qualities == [screen.JPEG_QUALITY, screen.JPEG_QUALITY_LOW]
    assert mime == "image/jpeg" and len(data) == 2 + screen.JPEG_QUALITY_LOW


def test_jpeg_failure_keeps_the_png(screens, monkeypatch):
    use(monkeypatch, grab=Grabs(noisy=True))
    monkeypatch.setattr(screen, "MAX_PNG_BYTES", 10)
    monkeypatch.setattr(screen, "_reencode_jpeg", lambda img, quality=80: None)
    data, _path, mime = screen.capture_screenshot("screen")
    assert mime == "image/png" and data[:4] == b"\x89PNG"


def test_png_size_from_ihdr(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 640, 480))
    assert screen._png_size(p) == (640, 480)
    p.write_bytes(b"not a png")
    assert screen._png_size(p) is None
    assert screen._png_size(tmp_path / "missing.png") is None


# ---- geometry sidecar -------------------------------------------------------------

def test_capture_writes_geometry_sidecar_for_screen(screens, monkeypatch):
    use(monkeypatch)
    screen.capture_screenshot("screen")
    raw = json.loads((screens / "latest.json").read_text())
    assert raw["region"] == "screen"
    assert (raw["image_w"], raw["image_h"]) == (1568, 882)
    assert (raw["origin_x"], raw["origin_y"], raw["screen_w"], raw["screen_h"]) == (0.0, 0.0, 1920.0, 1080.0)
    assert raw["scale"] == pytest.approx(1568 / 1920)
    assert raw["captured_at"] == 1000.0
    assert raw["window"] is None
    assert raw["display"] == {"id": 101, "index": 1, "main": True}


def test_load_geometry_roundtrip_to_screen_and_age(screens, monkeypatch):
    use(monkeypatch)
    screen.capture_screenshot("screen")
    g = screen.load_geometry()
    assert isinstance(g, screen.Geometry)
    assert g.to_screen(784, 441) == (pytest.approx(960, abs=1), pytest.approx(540, abs=1))
    monkeypatch.setattr(screen, "_now", lambda: 1030.0)
    assert g.age_s == pytest.approx(30.0)
    assert screen.GEOMETRY_MAX_AGE_S == 120


def test_to_screen_uses_origin_offset():
    g = screen.Geometry(region="window", image_w=1600, image_h=1200, origin_x=-300, origin_y=50,
                        screen_w=800, screen_h=600, scale=2.0, captured_at=0.0, window=None)
    assert g.to_screen(200, 100) == (-200.0, 100.0)


def test_load_geometry_missing_or_corrupt_is_none(screens):
    assert screen.load_geometry() is None
    (screens / "latest.json").write_text("{not json")
    assert screen.load_geometry() is None
    (screens / "latest.json").write_text('{"region": "screen"}')
    assert screen.load_geometry() is None


def test_build_geometry_needs_size_and_rect():
    assert screen._build_geometry("screen", None, (0, 0, 10, 10)) is None
    assert screen._build_geometry("screen", (10, 10), None) is None
    assert screen._build_geometry("screen", (10, 10), (0, 0, 0, 10)) is None
    assert screen._build_geometry("selection", (10, 10), (0, 0, 10, 10)) is None


def test_capture_without_a_display_list_uses_mss_primary(screens, monkeypatch):
    """No Win32 display list (odd failure): mss' primary monitor is
    captured, and the sidecar still describes it."""
    grab = Grabs()
    monkeypatch.setattr(screen, "_grab", grab)
    monkeypatch.setattr(screen, "_primary_rect", lambda: (0, 0, 1280, 720))
    screen.capture_screenshot("screen")
    assert grab.rects == [(0, 0, 1280, 720)]
    g = screen.load_geometry()
    assert (g.screen_w, g.screen_h, g.display) == (1280.0, 720.0, None)


async def test_screenshot_result_text_without_geometry_is_plain(screens, monkeypatch):
    use(monkeypatch)
    monkeypatch.setattr(screen, "load_geometry", lambda path=None: None)
    res = await screen.screenshot.handler({"region": "screen"})
    assert text(res) == "Screenshot of the screen."


# ---- window region ------------------------------------------------------------

def test_window_capture_grabs_the_window_and_records_its_bounds(screens, monkeypatch):
    win = FakeWin32(windows={7: _window((200, 100, 984, 610), pid=55, title="Groceries - Notepad")}, fg=7)
    win.paths[55] = r"C:\Windows\notepad.exe"
    win.names[win.paths[55]] = "Notepad"
    grab = Grabs()
    use(monkeypatch, win, grab)
    screen.capture_screenshot("window")
    assert grab.rects == [(200, 100, 784, 510)]
    g = screen.load_geometry()
    assert g.region == "window"
    assert (g.origin_x, g.origin_y, g.screen_w, g.screen_h) == (200.0, 100.0, 784.0, 510.0)
    assert g.scale == 1.0
    assert g.window == {"id": 7, "app": "Notepad", "title": "Groceries - Notepad",
                        "x": 200.0, "y": 100.0, "w": 784.0, "h": 510.0}
    assert g.display == {"id": 101, "index": 1, "main": True}


def test_window_capture_is_clipped_to_the_screen(screens, monkeypatch):
    """A window hanging off the edge: only its on-screen part is captured
    and the sidecar describes exactly that part."""
    win = FakeWin32(windows={7: _window((-100, -50, 700, 550))}, fg=7)
    grab = Grabs()
    use(monkeypatch, win, grab)
    screen.capture_screenshot("window")
    assert grab.rects == [(0, 0, 700, 550)]
    g = screen.load_geometry()
    assert (g.origin_x, g.origin_y, g.screen_w, g.screen_h) == (0.0, 0.0, 700.0, 550.0)


async def test_window_region_without_a_front_window_is_an_error(screens, monkeypatch):
    use(monkeypatch, FakeWin32(fg=0))
    res = await screen.screenshot.handler({"region": "window"})
    assert res["is_error"] and "front window" in text(res)


def test_window_capture_records_the_display_it_is_on(screens, monkeypatch):
    use(monkeypatch, two_displays())
    screen.capture_screenshot("window")
    g = screen.load_geometry()
    assert (g.origin_x, g.origin_y, g.screen_w, g.screen_h) == (2000.0, 200.0, 900.0, 700.0)
    assert g.display == {"id": 202, "index": 2, "main": False}


async def test_window_text_names_the_display(screens, monkeypatch):
    use(monkeypatch, two_displays())
    res = await screen.screenshot.handler({"region": "window"})
    assert text(res).startswith("Screenshot of the window on display 2 of 2 (secondary): 900×700 px")


# ---- front window -------------------------------------------------------------------

def test_front_window_id_is_the_foreground_window(monkeypatch):
    use(monkeypatch, FakeWin32(windows={5: _window((0, 0, 800, 600))}, fg=5))
    assert screen.front_window_id() == 5


@pytest.mark.parametrize("cls", ["Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"])
def test_front_window_id_none_for_the_desktop_and_taskbar(monkeypatch, cls):
    use(monkeypatch, FakeWin32(windows={5: _window((0, 0, 1920, 1080), cls=cls)}, fg=5))
    assert screen.front_window_id() is None


def test_front_window_id_skips_a_helper_sliver_for_the_apps_real_window(monkeypatch):
    win = FakeWin32(windows={
        9: _window((0, 0, 1920, 1080), pid=1),             # someone else's, in front
        5: _window((10, 10, 30, 30), pid=100),             # the foreground: a tiny helper
        6: _window((0, 0, 800, 600), pid=100, visible=False),
        8: _window((100, 100, 900, 700), pid=100),         # the app's real window
    }, fg=5)
    win.z = [9, 5, 6, 8]
    use(monkeypatch, win)
    assert screen.front_window_id() == 8


def test_front_window_id_none_when_nothing_qualifies(monkeypatch):
    use(monkeypatch, FakeWin32(windows={5: _window((0, 0, 20, 20))}, fg=5))
    assert screen.front_window_id() is None
    use(monkeypatch, FakeWin32(fg=0))
    assert screen.front_window_id() is None


def test_front_window_id_none_without_win32():
    assert screen.front_window_id() is None


# ---- displays -----------------------------------------------------------------------

def test_active_displays_primary_first_then_left_to_right():
    ds = screen._active_displays(FakeWin32([RIGHT, LEFT, PRIMARY]))
    assert [(d.id, d.index, d.main) for d in ds] == [(101, 1, True), (303, 2, False), (202, 3, False)]
    left = ds[1]
    assert (left.x, left.y, left.w, left.h) == (-2560.0, 0.0, 2560.0, 1440.0)
    assert left.name == r"\\.\DISPLAY3"
    assert ds[0].label == "primary" and left.label == "secondary"
    assert left.contains(-100, 700) and not left.contains(100, 700)


def test_displays_empty_without_win32():
    assert screen.displays() == []


def test_pick_display_auto_follows_the_front_window(monkeypatch):
    use(monkeypatch, two_displays())
    assert screen.pick_display("auto").index == 2


def test_pick_display_auto_falls_back_to_the_primary(monkeypatch):
    use(monkeypatch, two_displays(front_on_right=False))
    assert screen.pick_display("auto").index == 1
    assert screen.pick_display("main").index == 1
    assert screen.pick_display("primary").index == 1


def test_pick_display_by_number_and_bad_spec(monkeypatch):
    use(monkeypatch, two_displays(front_on_right=False))
    assert screen.pick_display("2").id == 202
    with pytest.raises(ValueError, match="no display 5"):
        screen.pick_display("5")
    with pytest.raises(ValueError, match="must be"):
        screen.pick_display("left")


def test_screen_capture_grabs_the_front_windows_display(screens, monkeypatch):
    grab = Grabs()
    use(monkeypatch, two_displays(), grab)
    screen.capture_screenshot("screen")
    assert grab.rects == [(1920, 0, 2560, 1440)]
    g = screen.load_geometry()
    assert (g.origin_x, g.origin_y, g.screen_w, g.screen_h) == (1920.0, 0.0, 2560.0, 1440.0)
    assert g.display == {"id": 202, "index": 2, "main": False}
    x, _y = g.to_screen(g.image_w / 2, 0)
    assert 1920.0 < x < 1920.0 + 2560.0


def test_geometry_origin_is_negative_left_of_the_primary(screens, monkeypatch):
    win = FakeWin32([PRIMARY, LEFT], windows={42: _window((-2000, 100, -1000, 900))}, fg=42)
    grab = Grabs()
    use(monkeypatch, win, grab)
    screen.capture_screenshot("screen")
    assert grab.rects == [(-2560, 0, 2560, 1440)]
    g = screen.load_geometry()
    assert (g.origin_x, g.origin_y) == (-2560.0, 0.0)
    assert g.to_screen(0, 0) == (-2560.0, 0.0)
    assert g.to_screen(g.image_w, 0)[0] == pytest.approx(0.0, abs=2)


async def test_screenshot_text_names_the_display_when_there_are_two(screens, monkeypatch):
    use(monkeypatch, two_displays())
    res = await screen.screenshot.handler({"region": "screen"})
    assert text(res) == (
        "Screenshot of display 2 of 2 (secondary, 2560×1440 px): 1568×882 px. "
        "Coordinates you pass to computer_* tools are in these image pixels."
    )


async def test_screenshot_display_number_picks_that_screen(screens, monkeypatch):
    grab = Grabs()
    use(monkeypatch, two_displays(front_on_right=False), grab)
    res = await screen.screenshot.handler({"region": "screen", "display": "2"})
    assert not res.get("is_error")
    assert grab.rects == [(1920, 0, 2560, 1440)]
    assert screen.load_geometry().origin_x == 1920.0


async def test_screenshot_display_out_of_range_is_an_error(screens, monkeypatch):
    use(monkeypatch, two_displays())
    res = await screen.screenshot.handler({"region": "screen", "display": "7"})
    assert res["is_error"] and "no display 7" in text(res)


async def test_screenshot_all_returns_one_image_per_display(screens, monkeypatch):
    grab = Grabs()
    use(monkeypatch, two_displays(), grab)
    res = await screen.screenshot.handler({"region": "screen", "display": "all"})
    assert [b["type"] for b in res["content"]] == ["image", "image", "text"]
    # the front window's display goes last, so latest.png/the sidecar are its
    assert grab.rects == [(0, 0, 1920, 1080), (1920, 0, 2560, 1440)]
    g = screen.load_geometry()
    assert g.display == {"id": 202, "index": 2, "main": False}
    # each capture is downscaled harder to share the one-message budget
    assert max(g.image_w, g.image_h) == screen.DOWNSCALE_ALL_MAX_PX
    body = text(res)
    assert "display 1 (primary, 1920×1080 px)" in body
    assert "display 2 (secondary, 2560×1440 px)" in body
    assert "pixels of the display 2 (secondary) image" in body


async def test_screenshot_all_names_what_did_not_fit(screens, monkeypatch):
    """Over the budget the oversized display is named in the text, never
    dropped in silence."""
    use(monkeypatch, two_displays(front_on_right=False), Grabs(noisy=True))
    monkeypatch.setattr(screen, "MAX_ALL_BYTES", 1)
    res = await screen.screenshot.handler({"region": "screen", "display": "all"})
    assert [b["type"] for b in res["content"]] == ["image", "text"]
    assert "display 2 didn't fit" in text(res)


async def test_screenshot_all_with_one_display_reads_like_a_plain_screen_shot(screens, monkeypatch):
    use(monkeypatch)
    res = await screen.screenshot.handler({"region": "screen", "display": "all"})
    assert [b["type"] for b in res["content"]] == ["image", "text"]
    assert text(res).startswith("Screenshot of the screen:")


# ---- selection ------------------------------------------------------------------------

def _snip(monkeypatch, win, image_after=2, size=(300, 200)):
    """The snipping overlay: the clipboard changes after `image_after` polls."""
    from PIL import Image
    polls = []
    started = []
    monkeypatch.setattr(screen, "_start_snip", lambda: started.append(1))

    def sleep(s):
        polls.append(s)
        if image_after is not None and len(polls) >= image_after:
            win.clip_seq = 2
    monkeypatch.setattr(screen, "_poll_sleep", sleep)
    t = [1000.0]

    def now():
        t[0] += screen.SELECTION_POLL_S
        return t[0]
    monkeypatch.setattr(screen, "_now", now)
    monkeypatch.setattr(screen, "_clipboard_image",
                        lambda: Image.new("RGBA", size) if win.clip_seq == 2 else None)
    return started, polls


def test_selection_waits_for_the_snip_and_writes_no_sidecar(screens, monkeypatch):
    """A snipped area has no known screen origin: no geometry at all, and
    the previous capture's sidecar must not survive to describe it."""
    (screens / "latest.json").write_text('{"region": "screen"}')
    win = use(monkeypatch)
    started, polls = _snip(monkeypatch, win)
    _data, path, mime = screen.capture_screenshot("selection")
    assert started == [1] and len(polls) == 2
    assert mime == "image/png" and screen._png_size(path) == (300, 200)
    assert not (screens / "latest.json").exists()
    assert screen.load_geometry() is None


def test_selection_cancelled_or_timed_out_is_an_error(screens, monkeypatch):
    win = use(monkeypatch)
    _snip(monkeypatch, win, image_after=None)
    result = screen.capture_screenshot("selection")
    assert isinstance(result, str) and "no area was selected" in result
    assert not (screens / "latest.png").exists()


async def test_selection_screenshot_text_has_no_geometry(screens, monkeypatch):
    win = use(monkeypatch)
    _snip(monkeypatch, win)
    res = await screen.screenshot.handler({"region": "selection"})
    assert text(res) == "Screenshot of the selection."


def test_selection_ignores_a_clipboard_change_without_an_image(screens, monkeypatch):
    win = use(monkeypatch)
    _started, polls = _snip(monkeypatch, win, image_after=1)
    images = iter([None, None])
    from PIL import Image
    monkeypatch.setattr(screen, "_clipboard_image", lambda: next(images, Image.new("RGB", (10, 10))))
    assert not isinstance(screen.capture_screenshot("selection"), str)
    assert len(polls) == 3


# ---- registration / live ---------------------------------------------------------------

def test_server_and_names():
    assert screen.screen_server["name"] == "screen"
    assert screen.SCREEN_TOOL_NAMES == ["screenshot"]


live_windows = pytest.mark.skipif(sys.platform != "win32", reason="needs Windows")


@pytest.mark.live
@live_windows
async def test_live_screenshot_under_2mb():
    res = await screen.screenshot.handler({"region": "screen"})
    assert not res.get("is_error"), text(res)
    data = base64.b64decode(res["content"][0]["data"])
    assert len(data) < 2 * 1024 * 1024


@pytest.mark.live
@live_windows
def test_live_displays_have_one_primary_and_real_sizes():
    found = screen.displays()
    assert found and sum(1 for d in found if d.main) == 1
    assert found[0].main and found[0].index == 1
    assert all(d.w > 0 and d.h > 0 for d in found)


@pytest.mark.live
@live_windows
async def test_live_each_display_number_captures_that_display():
    for d in screen.displays():
        res = await screen.screenshot.handler({"region": "screen", "display": str(d.index)})
        assert not res.get("is_error"), text(res)
        g = screen.load_geometry()
        assert (g.origin_x, g.origin_y, g.screen_w, g.screen_h) == (d.x, d.y, d.w, d.h)
        assert g.image_w / g.image_h == pytest.approx(d.w / d.h, rel=0.01)
