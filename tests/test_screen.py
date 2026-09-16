import subprocess

import pytest

from veronica.tools import screen


class Done:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


@pytest.fixture
def fake_run(monkeypatch):
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        return Done(out="")

    monkeypatch.setattr(screen.subprocess, "run", run)
    return calls


@pytest.fixture
def fake_screens_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "_screens_dir", lambda: tmp_path)
    return tmp_path


def _write_png_after_capture(monkeypatch, tmp_path, data=b"\x89PNG-fake"):
    """Make the fake screencapture/sips calls actually drop a file at the
    out_path argv entry, since capture_screenshot() checks out_path.exists()
    and later reads it."""
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        if argv[0] == "screencapture":
            out_path = argv[-1]
            with open(out_path, "wb") as f:
                f.write(data)
        return Done(out="")

    monkeypatch.setattr(screen.subprocess, "run", run)
    return calls


def text(res):
    return res["content"][-1]["text"]


async def test_screenshot_screen_region(fake_screens_dir, monkeypatch):
    calls = _write_png_after_capture(monkeypatch, fake_screens_dir)
    res = await screen.screenshot.handler({"region": "screen"})
    assert not res.get("is_error")
    argv = calls[0][0]
    assert argv[:4] == ["screencapture", "-x", "-t", "png"]
    assert "-l" not in argv and "-i" not in argv
    assert argv[-1].endswith(".png")
    # image content block present
    kinds = [b["type"] for b in res["content"]]
    assert kinds == ["image", "text"]
    assert res["content"][0]["mimeType"] == "image/png"
    import base64
    assert base64.b64decode(res["content"][0]["data"]) == b"\x89PNG-fake"
    assert "Screenshot saved to" in text(res)
    # sips was called to downscale
    sips_calls = [c for c in calls if c[0][0] == "sips"]
    assert sips_calls and "--resampleHeightWidthMax" in sips_calls[0][0]
    assert "1568" in sips_calls[0][0]


async def test_screenshot_selection_region(fake_screens_dir, monkeypatch):
    calls = _write_png_after_capture(monkeypatch, fake_screens_dir)
    await screen.screenshot.handler({"region": "selection"})
    argv = calls[0][0]
    assert "-i" in argv


async def test_screenshot_window_region_uses_front_window_id(fake_screens_dir, monkeypatch):
    calls = _write_png_after_capture(monkeypatch, fake_screens_dir)
    monkeypatch.setattr(screen, "front_window_id", lambda: 4242)
    await screen.screenshot.handler({"region": "window"})
    argv = calls[0][0]
    assert "-l" in argv
    assert argv[argv.index("-l") + 1] == "4242"


async def test_screenshot_window_region_no_front_window_is_error(fake_screens_dir, monkeypatch):
    monkeypatch.setattr(screen, "front_window_id", lambda: None)
    res = await screen.screenshot.handler({"region": "window"})
    assert res["is_error"]


async def test_screenshot_bad_region_defaults_to_screen(fake_screens_dir, monkeypatch):
    calls = _write_png_after_capture(monkeypatch, fake_screens_dir)
    await screen.screenshot.handler({"region": "bogus"})
    argv = calls[0][0]
    assert "-l" not in argv and "-i" not in argv


async def test_screenshot_no_file_produced_is_error(fake_run, fake_screens_dir):
    res = await screen.screenshot.handler({})
    assert res["is_error"]
    assert "no file" in res["content"][0]["text"] or "error" in res["content"][0]["text"]


async def test_screenshot_nonzero_exit_is_error(monkeypatch, fake_screens_dir):
    monkeypatch.setattr(screen.subprocess, "run", lambda *a, **k: Done(rc=1, err="denied"))
    res = await screen.screenshot.handler({})
    assert res["is_error"]
    assert "denied" in res["content"][0]["text"]


async def test_screenshot_timeout_is_error(monkeypatch, fake_screens_dir):
    def run(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=15)

    monkeypatch.setattr(screen.subprocess, "run", run)
    res = await screen.screenshot.handler({})
    assert res["is_error"]


def test_front_window_id_returns_none_without_quartz(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "Quartz":
            raise ImportError("no Quartz")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert screen.front_window_id() is None


def test_server_and_names():
    assert screen.screen_server["name"] == "screen"
    assert screen.SCREEN_TOOL_NAMES == ["screenshot"]


@pytest.mark.live
async def test_live_screenshot_under_2mb():
    res = await screen.screenshot.handler({"region": "screen"})
    assert not res.get("is_error")
    import base64
    data = base64.b64decode(res["content"][0]["data"])
    assert len(data) < 2 * 1024 * 1024
