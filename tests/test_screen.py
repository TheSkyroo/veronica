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


def _write_png_after_capture(monkeypatch, tmp_path, data=b"\x89PNG-fake", jpeg=b"\xff\xd8JPEG-fake"):
    """Make the fake screencapture/sips calls actually drop a file at the
    out_path argv entry, since capture_screenshot() checks out_path.exists()
    and later reads it. A fake `sips ... --out X.jpg` writes `jpeg` to X."""
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        if argv[0] == "screencapture":
            out_path = argv[-1]
            with open(out_path, "wb") as f:
                f.write(data)
        elif argv[0] == "sips" and "--out" in argv:
            with open(argv[argv.index("--out") + 1], "wb") as f:
                f.write(jpeg)
        elif argv[0] == "sips":
            # the real sips rewrites the file in place as a *new* file
            # (fresh inode, umask mode) — mirror that so mode handling that
            # only works before the downscale is caught here.
            import os
            target = argv[-1]
            with open(target, "rb") as f:
                existing = f.read()
            os.unlink(target)
            with open(target, "wb") as f:
                f.write(existing)
            os.chmod(target, 0o644)
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
    # T1: no on-disk path is advertised to the model
    assert "saved to" not in text(res).lower()
    assert str(fake_screens_dir) not in text(res)
    # sips was called to downscale
    sips_calls = [c for c in calls if c[0][0] == "sips"]
    assert sips_calls and "--resampleHeightWidthMax" in sips_calls[0][0]
    assert "1568" in sips_calls[0][0]
    # default timeout for a non-interactive capture
    assert calls[0][1]["timeout"] == screen.TIMEOUT_S


async def test_screenshot_keeps_only_latest_png_with_0600(fake_screens_dir, monkeypatch):
    """T1: nothing accumulates — every capture overwrites the single
    latest.png (mode 0600), never a timestamped file."""
    import os
    import stat
    _write_png_after_capture(monkeypatch, fake_screens_dir)
    await screen.screenshot.handler({"region": "screen"})
    await screen.screenshot.handler({"region": "screen"})
    files = sorted(p.name for p in fake_screens_dir.iterdir())
    assert files == ["latest.png"]
    mode = stat.S_IMODE(os.stat(fake_screens_dir / "latest.png").st_mode)
    assert mode == 0o600


async def test_screenshot_stale_latest_removed_before_capture(fake_screens_dir, monkeypatch):
    """A failed capture must not leave (or serve) the previous capture."""
    (fake_screens_dir / "latest.png").write_bytes(b"old")
    monkeypatch.setattr(screen.subprocess, "run", lambda *a, **k: Done(rc=1, err="denied"))
    res = await screen.screenshot.handler({})
    assert res["is_error"]
    assert not (fake_screens_dir / "latest.png").exists()


async def test_screenshot_over_3mb_reencoded_as_jpeg(fake_screens_dir, monkeypatch):
    """T7: a PNG over MAX_PNG_BYTES is re-encoded via sips as JPEG q80,
    returned with mimeType image/jpeg, and the transient .jpg is deleted."""
    big = b"\x89PNG" + b"\0" * (screen.MAX_PNG_BYTES + 1)
    calls = _write_png_after_capture(monkeypatch, fake_screens_dir, data=big)
    res = await screen.screenshot.handler({"region": "screen"})
    assert not res.get("is_error")
    assert res["content"][0]["mimeType"] == "image/jpeg"
    import base64
    assert base64.b64decode(res["content"][0]["data"]) == b"\xff\xd8JPEG-fake"
    jpeg_calls = [c[0] for c in calls if c[0][0] == "sips" and "--out" in c[0]]
    assert len(jpeg_calls) == 1
    argv = jpeg_calls[0]
    assert argv[1:5] == ["-s", "format", "jpeg", "-s"] and argv[5:7] == ["formatOptions", "80"]
    assert argv[-1].endswith(".jpg")
    assert sorted(p.name for p in fake_screens_dir.iterdir()) == ["latest.png"]


async def test_screenshot_over_3mb_keeps_png_if_reencode_fails(fake_screens_dir, monkeypatch):
    big = b"\x89PNG" + b"\0" * (screen.MAX_PNG_BYTES + 1)
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        if argv[0] == "screencapture":
            with open(argv[-1], "wb") as f:
                f.write(big)
        if argv[0] == "sips" and "--out" in argv:
            return Done(rc=1, err="nope")
        return Done(out="")

    monkeypatch.setattr(screen.subprocess, "run", run)
    res = await screen.screenshot.handler({"region": "screen"})
    assert not res.get("is_error")
    assert res["content"][0]["mimeType"] == "image/png"


def test_capture_screenshot_returns_mime(fake_screens_dir, monkeypatch):
    _write_png_after_capture(monkeypatch, fake_screens_dir)
    data, path, mime = screen.capture_screenshot("screen")
    assert data == b"\x89PNG-fake" and path == fake_screens_dir / "latest.png" and mime == "image/png"


def test_latest_screenshot_path():
    assert screen.latest_screenshot_path().name == "latest.png"


async def test_screenshot_selection_region(fake_screens_dir, monkeypatch):
    calls = _write_png_after_capture(monkeypatch, fake_screens_dir)
    await screen.screenshot.handler({"region": "selection"})
    argv, kw = calls[0]
    assert "-i" in argv
    # T8: the user has to drag out a region first
    assert kw["timeout"] == screen.SELECTION_TIMEOUT_S == 60


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


async def test_screenshot_retries_on_main_display_after_display_change(monkeypatch, fake_screens_dir):
    """Right after a monitor change, screencapture can fail with "could not
    create image from display"; a second attempt pinned to display 1
    usually works."""
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        if argv[0] == "screencapture" and "-D" not in argv:
            return Done(rc=1, err="screencapture: could not create image from display")
        if argv[0] == "screencapture":
            with open(argv[-1], "wb") as f:
                f.write(b"\x89PNG-fake")
        return Done(out="")

    monkeypatch.setattr(screen.subprocess, "run", run)
    res = await screen.screenshot.handler({})
    assert not res.get("is_error")
    captures = [a for a in calls if a[0] == "screencapture"]
    assert len(captures) == 2
    assert captures[1][:6] == ["screencapture", "-x", "-t", "png", "-D", "1"]
    assert captures[1][-1] == captures[0][-1]


async def test_screenshot_display_change_error_copy_when_retry_also_fails(monkeypatch, fake_screens_dir):
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return Done(rc=1, err="screencapture: could not create image from display")

    monkeypatch.setattr(screen.subprocess, "run", run)
    res = await screen.screenshot.handler({})
    assert res["is_error"]
    assert len([a for a in calls if a[0] == "screencapture"]) == 2
    text = res["content"][0]["text"]
    assert "display setup just changed" in text
    assert "Screen Recording" in text
    assert "Try again in a moment" in text


async def test_screenshot_other_failures_are_not_retried(monkeypatch, fake_screens_dir):
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return Done(rc=1, err="denied")

    monkeypatch.setattr(screen.subprocess, "run", run)
    res = await screen.screenshot.handler({})
    assert res["is_error"]
    assert len(calls) == 1


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
    monkeypatch.setattr(screen, "_frontmost_pid", lambda: None)
    assert screen.front_window_id() is None


def _win(number, *, pid=100, layer=0, alpha=1.0, w=800, h=600):
    return {
        "kCGWindowNumber": number, "kCGWindowOwnerPID": pid, "kCGWindowLayer": layer,
        "kCGWindowAlpha": alpha, "kCGWindowBounds": {"X": 0, "Y": 0, "Width": w, "Height": h},
    }


def test_front_window_id_skips_alpha0_other_pid_and_tiny_windows(monkeypatch):
    """T4: the first layer-0 entry is often an invisible helper window;
    pick the frontmost app's first visible, real-sized window instead."""
    monkeypatch.setattr(screen, "_frontmost_pid", lambda: 100)
    monkeypatch.setattr(screen, "_window_list", lambda: [
        _win(1, alpha=0.0),                 # invisible helper window
        _win(2, layer=25),                  # menu bar / overlay layer
        _win(3, pid=200),                   # some other app's window
        _win(4, w=20, h=20),                # sliver
        _win(5),                            # the real one
        _win(6),
    ])
    assert screen.front_window_id() == 5


def test_front_window_id_without_pid_still_filters_alpha_and_size(monkeypatch):
    monkeypatch.setattr(screen, "_frontmost_pid", lambda: None)
    monkeypatch.setattr(screen, "_window_list", lambda: [_win(1, alpha=0.0), _win(2, w=10), _win(3, pid=999)])
    assert screen.front_window_id() == 3


def test_front_window_id_none_when_nothing_qualifies(monkeypatch):
    monkeypatch.setattr(screen, "_frontmost_pid", lambda: 100)
    monkeypatch.setattr(screen, "_window_list", lambda: [_win(1, alpha=0.0), _win(2, pid=5)])
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


async def test_screenshot_jpeg_still_big_retries_at_lower_quality(fake_screens_dir, monkeypatch):
    """If the q80 JPEG is still over MAX_PNG_BYTES, re-encode at q60 so the
    base64 stays under the Agent SDK's 1 MiB JSON line limit."""
    big = b"\x89PNG" + b"\0" * (screen.MAX_PNG_BYTES + 1)
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        if argv[0] == "screencapture":
            with open(argv[-1], "wb") as f:
                f.write(big)
        if argv[0] == "sips" and "--out" in argv:
            q = argv[argv.index("formatOptions") + 1]
            payload = b"\xff\xd8" + (b"\0" * (screen.MAX_PNG_BYTES + 5) if q == "80" else b"small")
            with open(argv[argv.index("--out") + 1], "wb") as f:
                f.write(payload)
        return Done(out="")

    monkeypatch.setattr(screen.subprocess, "run", run)
    res = await screen.screenshot.handler({"region": "screen"})
    assert res["content"][0]["mimeType"] == "image/jpeg"
    import base64
    assert base64.b64decode(res["content"][0]["data"]) == b"\xff\xd8small"
    qualities = [c[c.index("formatOptions") + 1] for c in calls if c[0] == "sips" and "--out" in c]
    assert qualities == ["80", "60"]
