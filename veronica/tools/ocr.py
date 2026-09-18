"""On-device OCR over the latest screenshot via Apple's Vision framework.

`recognize_text` runs VNRecognizeTextRequest (accurate, with language
correction) on a PNG and returns `Word`s in **image pixels with a
top-left origin** — the same coordinate system Claude reads off the
screenshot and passes to the computer_* tools. Vision reports boxes
normalized with a bottom-left origin, so y is flipped here.

All framework access goes through `_vision()` so tests can hand in a fake.
"""
import difflib
import re
from dataclasses import dataclass
from pathlib import Path

from veronica.tools.screen import _png_size, load_geometry

DEFAULT_LANGUAGES = ("en-US", "hi-IN")
FUZZY_RATIO = 0.8


@dataclass
class Word:
    """One recognized line/word: bounding box in image pixels, top-left origin."""
    text: str
    x: float
    y: float
    w: float
    h: float
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)


def _vision():
    import Vision
    return Vision


def _image_size(png_path: Path, image_size: tuple[int, int] | None) -> tuple[int, int]:
    if image_size is not None:
        return int(image_size[0]), int(image_size[1])
    size = _png_size(png_path)
    if size is not None:
        return size
    geometry = load_geometry()
    if geometry is not None:
        return geometry.image_w, geometry.image_h
    raise ValueError(f"cannot determine the pixel size of {png_path}")


def _set_languages(req, vision, languages) -> None:
    """Ask for `languages` (only those Vision says it supports); any
    failure leaves the request on Vision's default."""
    try:
        supported, err = req.supportedRecognitionLanguagesAndReturnError_(None)
        if err is not None or supported is None:
            return
        supported = {str(s) for s in supported}
        wanted = [lang for lang in languages if lang in supported]
        if wanted:
            req.setRecognitionLanguages_(wanted)
    except Exception:
        pass


def recognize_text(
    png_path: Path,
    *,
    vision=None,
    languages=DEFAULT_LANGUAGES,
    image_size: tuple[int, int] | None = None,
) -> list[Word]:
    """OCR `png_path`. `image_size` (w, h) overrides reading it from the
    file / the geometry sidecar. Raises RuntimeError if Vision fails."""
    vision = vision if vision is not None else _vision()
    width, height = _image_size(png_path, image_size)
    url = vision.NSURL.fileURLWithPath_(str(png_path))
    handler = vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    req = vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(True)
    _set_languages(req, vision, languages)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        raise RuntimeError(f"text recognition failed: {err}")
    words: list[Word] = []
    for obs in req.results() or []:
        candidates = obs.topCandidates_(1)
        if not candidates:
            continue
        cand = candidates[0]
        bb = obs.boundingBox()
        bx, by = float(bb.origin.x), float(bb.origin.y)
        bw, bh = float(bb.size.width), float(bb.size.height)
        words.append(Word(
            text=str(cand.string()),
            x=bx * width,
            y=(1.0 - by - bh) * height,
            w=bw * width,
            h=bh * height,
            confidence=float(cand.confidence()),
        ))
    return words


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def find_text(words: list[Word], query: str) -> list[Word]:
    """Words matching `query`, case/whitespace-insensitive: exact matches
    first, then lines containing it, then fuzzy (difflib ratio >= 0.8);
    each tier ordered top-to-bottom, left-to-right."""
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
    def key(w: Word) -> tuple[float, float]:
        return (w.y, w.x)

    return sorted(exact, key=key) + sorted(contains, key=key) + sorted(fuzzy, key=key)
