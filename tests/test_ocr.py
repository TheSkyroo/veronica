"""OCR over a fake WinRT namespace: the Windows.Media.Ocr / imaging /
streams classes `ocr._winrt()` would hand over, with the async methods as
coroutine functions — nothing here needs Windows."""
import asyncio
from types import SimpleNamespace

import pytest

from veronica.tools import ocr
from veronica.tools.ocr import Word


def _rect(x, y, w, h):
    return SimpleNamespace(x=x, y=y, width=w, height=h)


def _line(*words, text=None):
    """An OcrLine from (text, (x, y, w, h)) words; its text joins them."""
    ws = [SimpleNamespace(text=t, bounding_rect=_rect(*r)) for t, r in words]
    return SimpleNamespace(text=text if text is not None else " ".join(t for t, _ in words), words=ws)


class FakeBitmap:
    def __init__(self, w, h, fmt="BGRA8", alpha="PREMULTIPLIED"):
        self.pixel_width, self.pixel_height = w, h
        self.bitmap_pixel_format, self.bitmap_alpha_mode = fmt, alpha


class FakeRt:
    """Records the decode → convert → recognize pipeline."""

    def __init__(self, lines=(), size=(100, 100), *, fmt="BGRA8", alpha="PREMULTIPLIED",
                 profile_engine=True, supported=("en-US",), fail=None, max_dim=10000):
        rt = self
        self.calls: list = []
        self.written = b""
        self.lines = list(lines)

        class Engine:
            def __init__(self, lang):
                self.lang = lang

            async def recognize_async(self, bitmap):
                rt.calls.append(("recognize", self.lang, bitmap))
                if fail == "recognize":
                    raise OSError("ocr crashed")
                await asyncio.sleep(0)
                return SimpleNamespace(lines=list(rt.lines))

        class OcrEngine:
            max_image_dimension = max_dim

            @staticmethod
            def try_create_from_user_profile_languages():
                return Engine("profile") if profile_engine else None

            @staticmethod
            def is_language_supported(lang):
                return lang.tag in supported

            @staticmethod
            def try_create_from_language(lang):
                return Engine(lang.tag)

        class Language:
            def __init__(self, tag):
                self.tag = tag

        class Stream:
            def get_output_stream_at(self, pos):
                rt.calls.append(("output_at", pos))
                return self

            def seek(self, pos):
                rt.calls.append(("seek", pos))

        class DataWriter:
            def __init__(self, out):
                self.out = out

            def write_bytes(self, data):
                rt.written += bytes(data)

            async def store_async(self):
                rt.calls.append("store")

            async def flush_async(self):
                rt.calls.append("flush")

            def detach_stream(self):
                rt.calls.append("detach")

        class Decoder:
            async def get_software_bitmap_async(self):
                if fail == "decode":
                    raise OSError("not an image")
                return FakeBitmap(*size, fmt=fmt, alpha=alpha)

        class BitmapDecoder:
            @staticmethod
            async def create_async(stream):
                rt.calls.append("decode")
                return Decoder()

        class SoftwareBitmap:
            @staticmethod
            def convert(bitmap, fmt_, alpha_):
                rt.calls.append(("convert", fmt_, alpha_))
                return FakeBitmap(bitmap.pixel_width, bitmap.pixel_height, fmt_, alpha_)

        self.OcrEngine, self.Language, self.DataWriter = OcrEngine, Language, DataWriter
        self.InMemoryRandomAccessStream = Stream
        self.BitmapDecoder, self.SoftwareBitmap = BitmapDecoder, SoftwareBitmap
        self.BitmapPixelFormat = SimpleNamespace(BGRA8="BGRA8", RGBA8="RGBA8")
        self.BitmapAlphaMode = SimpleNamespace(PREMULTIPLIED="PREMULTIPLIED", STRAIGHT="STRAIGHT")


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "latest.png"
    p.write_bytes(b"\x89PNG-bytes")
    return p


def test_recognize_text_reads_top_left_pixel_boxes_as_is(png):
    rt = FakeRt([_line(("Save", (156.8, 50.95, 313.6, 50.95)))], size=(1568, 1019))
    (w,) = ocr.recognize_text(png, winrt=rt, image_size=(1568, 1019))
    assert w.text == "Save" and w.confidence == 1.0 and not w.line
    assert (w.x, w.y, w.w, w.h) == (156.8, 50.95, 313.6, 50.95)   # no y flip
    assert w.center == (pytest.approx(156.8 + 313.6 / 2), pytest.approx(50.95 + 50.95 / 2))


def test_recognize_text_feeds_the_file_through_a_bitmap(png):
    rt = FakeRt([])
    ocr.recognize_text(png, winrt=rt)
    assert rt.written == png.read_bytes()
    assert rt.calls[:5] == [("output_at", 0), "store", "flush", "detach", ("seek", 0)]
    assert "decode" in rt.calls
    assert not any(c[0] == "convert" for c in rt.calls if isinstance(c, tuple))   # already Bgra8
    assert rt.calls[-1][:2] == ("recognize", "profile")


def test_recognize_text_converts_other_pixel_formats_to_bgra8_premultiplied(png):
    rt = FakeRt([], fmt="RGBA8", alpha="STRAIGHT")
    ocr.recognize_text(png, winrt=rt)
    assert ("convert", "BGRA8", "PREMULTIPLIED") in rt.calls
    assert rt.calls[-1][2].bitmap_pixel_format == "BGRA8"


def test_recognize_text_falls_back_to_an_installed_language(png):
    rt = FakeRt([], profile_engine=False, supported=("hi-IN",))
    ocr.recognize_text(png, winrt=rt)
    assert rt.calls[-1][:2] == ("recognize", "hi-IN")


def test_recognize_text_without_any_ocr_language_is_an_error(png):
    rt = FakeRt([], profile_engine=False, supported=())
    with pytest.raises(RuntimeError, match="no OCR language"):
        ocr.recognize_text(png, winrt=rt)


@pytest.mark.parametrize("fail", ["decode", "recognize"])
def test_recognize_text_winrt_failures_raise_runtime_error(png, fail):
    with pytest.raises(RuntimeError, match="text recognition failed"):
        ocr.recognize_text(png, winrt=FakeRt([], fail=fail))


def test_recognize_text_refuses_images_over_the_engine_limit(png):
    with pytest.raises(RuntimeError, match="over 1000 px"):
        ocr.recognize_text(png, winrt=FakeRt([], size=(1200, 800), max_dim=1000))


def test_recognize_text_scales_boxes_to_the_callers_image_size(png):
    rt = FakeRt([_line(("x", (10, 20, 30, 40)))], size=(100, 100))
    (w,) = ocr.recognize_text(png, winrt=rt, image_size=(200, 50))
    assert (w.x, w.y, w.w, w.h) == (20.0, 10.0, 60.0, 20.0)


async def test_recognize_text_works_from_inside_a_running_loop(png):
    """Sync API, async WinRT: called on a thread that already runs a loop
    it must not try to nest one."""
    rt = FakeRt([_line(("ok", (0, 0, 1, 1)))])
    assert [w.text for w in ocr.recognize_text(png, winrt=rt)] == ["ok"]


# --- per-word boxes ----------------------------------------------------------

def test_recognize_text_emits_a_word_per_token_plus_the_line(png):
    rt = FakeRt([_line(("Cancel", (10, 80, 30, 10)), ("Save", (50, 80, 20, 10)))])
    words = ocr.recognize_text(png, winrt=rt)
    assert [(w.text, w.line) for w in words] == [("Cancel Save", True), ("Cancel", False), ("Save", False)]
    line, cancel, save = words
    assert (line.x, line.y, line.w, line.h) == (10.0, 80.0, 60.0, 10.0)   # union of the words
    assert (cancel.x, cancel.w) == (10.0, 30.0)
    assert (save.x, save.w) == (50.0, 20.0)


def test_recognize_text_line_box_is_the_union_of_uneven_words(png):
    rt = FakeRt([_line(("Big", (10, 5, 20, 30)), ("small", (40, 15, 25, 10)))])
    line = ocr.recognize_text(png, winrt=rt)[0]
    assert (line.x, line.y, line.w, line.h) == (10.0, 5.0, 55.0, 30.0)


def test_recognize_text_single_token_line_is_not_duplicated(png):
    rt = FakeRt([_line(("Save", (10, 90, 20, 5)))])
    words = ocr.recognize_text(png, winrt=rt)
    assert [(w.text, w.line) for w in words] == [("Save", False)]   # a lone label is a token, not a line


def test_recognize_text_skips_lines_without_words(png):
    rt = FakeRt([_line(text=""), _line(("  ", (0, 0, 1, 1))), _line(("a", (0, 0, 1, 1)))])
    assert [w.text for w in ocr.recognize_text(png, winrt=rt)] == ["a"]


def test_find_text_save_on_cancel_save_line_returns_the_token_centre(png):
    rt = FakeRt([_line(("Cancel", (10, 10, 30, 10)), ("Save", (50, 10, 20, 10)))])
    words = ocr.recognize_text(png, winrt=rt)
    (m,) = ocr.find_text(words, "Save")
    assert m.text == "Save" and not m.line
    assert m.center == (pytest.approx(60.0), pytest.approx(15.0))    # not the line's (40, 15)
    (m,) = ocr.find_text(words, "Cancel Save")
    assert m.line and m.center == (pytest.approx(40.0), pytest.approx(15.0))


def _w(text, x=0.0, y=0.0, line=False):
    return Word(text=text, x=x, y=y, w=10.0, h=10.0, confidence=1.0, line=line)


def test_find_text_contains_and_fuzzy_prefer_tokens_over_lines():
    words = [_w("Open Preferences now", y=0, line=True), _w("Open", x=0), _w("Preferences", x=30), _w("now", x=60),
             _w("Prefernces", y=20, line=True)]
    # contains: the token wins over the line that also contains it
    assert [(w.text, w.line) for w in ocr.find_text(words, "Prefer")] == [("Preferences", False)]
    # fuzzy: the token beats the line; a near-miss single-token line is a token-less fallback
    assert [(w.text, w.line) for w in ocr.find_text(words, "Preferenses")] == [("Preferences", False)]


def test_find_text_falls_back_to_lines_when_no_token_matches():
    words = [_w("Save As Template", line=True), _w("Save", x=0), _w("As", x=20), _w("Template", x=40)]
    assert [(w.text, w.line) for w in ocr.find_text(words, "Save As")] == [("Save As Template", True)]
    assert [(w.text, w.line) for w in ocr.find_text(words, "Save As Templete")] == [("Save As Template", True)]


def test_word_line_defaults_false():
    assert Word("x", 0, 0, 1, 1, 1.0).line is False


def test_find_text_exact_before_contains_before_fuzzy():
    words = [_w("Save As", y=10), _w("save", y=30), _w("Sav", y=20), _w("Cancel", y=0)]
    assert [w.text for w in ocr.find_text(words, "  SAVE ")] == ["save", "Save As", "Sav"]


def test_find_text_sorts_top_to_bottom_left_to_right_within_tier():
    words = [_w("OK", x=300, y=50), _w("ok", x=10, y=50), _w("Ok", x=100, y=5)]
    assert [(w.x, w.y) for w in ocr.find_text(words, "ok")] == [(100, 5), (10, 50), (300, 50)]


def test_find_text_collapses_whitespace_and_case():
    words = [_w("Open   Recent"), _w("Open Recent Files")]
    assert [w.text for w in ocr.find_text(words, "open  recent")] == ["Open   Recent", "Open Recent Files"]


def test_find_text_fuzzy_threshold():
    words = [_w("Preferences"), _w("Prefernces"), _w("Something else")]
    res = ocr.find_text(words, "Preferences")
    assert [w.text for w in res] == ["Preferences", "Prefernces"]
    assert ocr.find_text(words, "zzzz") == []
    assert ocr.find_text(words, "   ") == []


def test_single_token_line_counts_as_token_and_is_not_suppressed():
    from veronica.tools.ocr import Word, find_text
    words = [
        Word("Save changes before closing?", 0, 10, 200, 12, 0.9, line=True),
        Word("Save", 20, 10, 20, 12, 0.9),
        Word("changes", 45, 10, 40, 12, 0.9),
        Word("Save", 310, 100, 30, 20, 0.95),          # a standalone button: line=False
    ]
    hits = find_text(words, "Sav")
    centres = [(round(w.x + w.w / 2), round(w.y + w.h / 2)) for w in hits]
    assert (325, 110) in centres
