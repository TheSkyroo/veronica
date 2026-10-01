"""On-device OCR over the latest screenshot via Windows' built-in OCR
engine (Windows.Media.Ocr, through the `winrt` projection).

`recognize_text` decodes the image into a SoftwareBitmap, runs
`OcrEngine.RecognizeAsync` and returns `Word`s in **image pixels with a
top-left origin** — the same coordinate system Claude reads off the
screenshot and passes to the computer_* tools. Windows reports word
rectangles that way already, so nothing is flipped.

The engine reports *lines* made of *words*. For each line the whole-line
`Word` (`line=True`, boxed by the union of its words) is emitted plus one
`Word` per word, so that clicking "Save" lands on the Save button and not
the middle of the line "Cancel Save". `find_text` prefers word matches.
Windows' OCR reports no confidence, so every `Word` carries 1.0.

All WinRT access goes through `_winrt()` so tests can hand in a fake
namespace (the WinRT classes used here, with async methods as coroutine
functions). The WinRT calls are async; `recognize_text` stays synchronous
and runs them on an event loop of its own.
"""
import asyncio
import concurrent.futures
import difflib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

log = logging.getLogger(__name__)

# Tried in order when no OCR language from the user's profile is
# installed (Windows recognizes one language per engine).
DEFAULT_LANGUAGES = ("en-US", "hi-IN")
FUZZY_RATIO = 0.8
_TOKEN = re.compile(r"\S+")


@dataclass
class Word:
    """One recognized token or whole line (`line=True`): bounding box in
    image pixels, top-left origin."""
    text: str
    x: float
    y: float
    w: float
    h: float
    confidence: float
    line: bool = False

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)


def _winrt():
    """The WinRT classes OCR needs, in one namespace."""
    from winrt.windows.globalization import Language
    from winrt.windows.graphics.imaging import (
        BitmapAlphaMode,
        BitmapDecoder,
        BitmapPixelFormat,
        SoftwareBitmap,
    )
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream
    return SimpleNamespace(
        Language=Language, BitmapAlphaMode=BitmapAlphaMode, BitmapDecoder=BitmapDecoder,
        BitmapPixelFormat=BitmapPixelFormat, SoftwareBitmap=SoftwareBitmap, OcrEngine=OcrEngine,
        DataWriter=DataWriter, InMemoryRandomAccessStream=InMemoryRandomAccessStream,
    )


def _engine(rt, languages):
    """An OcrEngine for the user's profile languages, else the first of
    `languages` that has an OCR language pack installed. Raises
    RuntimeError when there is none."""
    engine = rt.OcrEngine.try_create_from_user_profile_languages()
    if engine is not None:
        return engine
    for tag in languages:
        try:
            lang = rt.Language(tag)
            if rt.OcrEngine.is_language_supported(lang):
                engine = rt.OcrEngine.try_create_from_language(lang)
                if engine is not None:
                    return engine
        except Exception as e:  # noqa: BLE001 — try the next language
            log.debug("no OCR engine for %s: %s", tag, e)
    raise RuntimeError(
        "text recognition failed: no OCR language is installed "
        "(Settings > Time & language > Language & region)"
    )


async def _load_bitmap(rt, data: bytes):
    """Decode `data` (PNG/JPEG bytes) into a Bgra8 premultiplied
    SoftwareBitmap, the format the OCR engine takes."""
    stream = rt.InMemoryRandomAccessStream()
    writer = rt.DataWriter(stream.get_output_stream_at(0))
    writer.write_bytes(data)
    await writer.store_async()
    await writer.flush_async()
    writer.detach_stream()
    stream.seek(0)
    decoder = await rt.BitmapDecoder.create_async(stream)
    bitmap = await decoder.get_software_bitmap_async()
    if (bitmap.bitmap_pixel_format != rt.BitmapPixelFormat.BGRA8
            or bitmap.bitmap_alpha_mode != rt.BitmapAlphaMode.PREMULTIPLIED):
        bitmap = rt.SoftwareBitmap.convert(bitmap, rt.BitmapPixelFormat.BGRA8, rt.BitmapAlphaMode.PREMULTIPLIED)
    return bitmap


async def _recognize(rt, data: bytes, languages):
    """(OcrResult, (bitmap width, bitmap height))."""
    engine = _engine(rt, languages)
    bitmap = await _load_bitmap(rt, data)
    width, height = int(bitmap.pixel_width), int(bitmap.pixel_height)
    limit = int(getattr(rt.OcrEngine, "max_image_dimension", 0) or 0)
    if limit and max(width, height) > limit:
        raise RuntimeError(f"text recognition failed: image {width}×{height} is over {limit} px")
    return await engine.recognize_async(bitmap), (width, height)


def _run_sync(make_coro):
    """Run `make_coro()` to completion from synchronous code: on a fresh
    event loop here, or — if this thread already runs a loop — on one in
    a short-lived helper thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make_coro())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(make_coro())).result()


def recognize_text(
    png_path: Path,
    *,
    winrt=None,
    languages=DEFAULT_LANGUAGES,
    image_size: tuple[int, int] | None = None,
) -> list[Word]:
    """OCR `png_path`. `image_size` (w, h) is the pixel size the caller's
    coordinates use; boxes are scaled to it if it differs from the decoded
    image's. Raises RuntimeError if recognition fails."""
    rt = winrt if winrt is not None else _winrt()
    data = Path(png_path).read_bytes()
    try:
        result, (width, height) = _run_sync(lambda: _recognize(rt, data, languages))
    except RuntimeError:
        raise
    except Exception as e:  # noqa: BLE001 — one error type for callers
        raise RuntimeError(f"text recognition failed: {e}") from e
    sx = sy = 1.0
    if image_size is not None and width and height:
        sx, sy = image_size[0] / width, image_size[1] / height
    words: list[Word] = []
    for line in result.lines or ():
        boxes = [(str(w.text), w.bounding_rect) for w in (line.words or ()) if str(w.text).strip()]
        if not boxes:
            continue
        text = str(line.text or "") or " ".join(t for t, _ in boxes)
        # A single-word line IS a token (a lone button label like "Save"):
        # flag it as such so find_text's token-preference never hides it.
        single = len(boxes) < 2
        words.append(_word(text, _union(r for _, r in boxes), sx, sy, line=not single))
        if not single:
            words.extend(_word(t, r, sx, sy, line=False) for t, r in boxes)
    return words


def _union(rects) -> tuple[float, float, float, float]:
    x1 = y1 = float("inf")
    x2 = y2 = float("-inf")
    for r in rects:
        x1, y1 = min(x1, float(r.x)), min(y1, float(r.y))
        x2, y2 = max(x2, float(r.x) + float(r.width)), max(y2, float(r.y) + float(r.height))
    return SimpleNamespace(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


def _word(text: str, rect, sx: float, sy: float, *, line: bool) -> Word:
    """A `Word` from a WinRT Rect (image pixels, top-left origin)."""
    return Word(
        text=text, x=float(rect.x) * sx, y=float(rect.y) * sy,
        w=float(rect.width) * sx, h=float(rect.height) * sy, confidence=1.0, line=line,
    )


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def find_text(words: list[Word], query: str) -> list[Word]:
    """Words matching `query`, case/whitespace-insensitive: exact matches
    first, then words containing it, then fuzzy (difflib ratio >= 0.8);
    each tier ordered top-to-bottom, left-to-right.

    Whole-line words only appear in the contains/fuzzy tiers when no token
    matched at all: the line "Cancel Save" also contains "Save", but its
    centre is between the two buttons, so once the Save token matched the
    line is not offered as a second, worse target."""
    q = _norm(query)
    if not q:
        return []
    exact: list[Word] = []
    contains: list[Word] = []
    fuzzy: list[Word] = []
    for w in words:
        t = _norm(w.text)
        if t == q:
            exact.append(w)
        elif q in t:
            contains.append(w)
        elif difflib.SequenceMatcher(None, t, q).ratio() >= FUZZY_RATIO:
            fuzzy.append(w)

    if any(not w.line for w in exact + contains + fuzzy):
        contains = [w for w in contains if not w.line]
        fuzzy = [w for w in fuzzy if not w.line]

    def key(w: Word) -> tuple[float, float]:
        return (w.y, w.x)

    return sorted(exact, key=key) + sorted(contains, key=key) + sorted(fuzzy, key=key)
