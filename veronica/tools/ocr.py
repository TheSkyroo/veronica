"""On-device OCR over the latest screenshot via Apple's Vision framework.

`recognize_text` runs VNRecognizeTextRequest (accurate, with language
correction) on a PNG and returns `Word`s in **image pixels with a
top-left origin** — the same coordinate system Claude reads off the
screenshot and passes to the computer_* tools. Vision reports boxes
normalized with a bottom-left origin, so y is flipped here.

Vision's observations are *lines* ("Cancel Save" is one box). For each
line the whole-line `Word` (`line=True`) is emitted plus one `Word` per
whitespace token, boxed via `VNRecognizedText.boundingBoxForRange_error_`,
so that clicking "Save" lands on the Save button and not the middle of
the line. `find_text` prefers token matches.

All framework access goes through `_vision()` so tests can hand in a fake
(it needs the Vision names used here plus `NSMakeRange`).
"""
import difflib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from veronica.tools.screen import _png_size, load_geometry

log = logging.getLogger(__name__)

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


class _Frameworks:
    """Vision plus the one Foundation helper OCR needs (`NSMakeRange`),
    behind a single object so tests can hand in one fake."""

    def __init__(self, vision, make_range):
        self._vision = vision
        self.NSMakeRange = make_range

    def __getattr__(self, name):
        return getattr(self._vision, name)


def _vision():
    import Vision
    from Foundation import NSMakeRange
    return _Frameworks(Vision, NSMakeRange)


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
        text = str(cand.string())
        confidence = float(cand.confidence())
        words.append(_word(text, obs.boundingBox(), confidence, width, height, line=True))
        words.extend(_token_words(vision, cand, text, confidence, width, height))
    return words


def _word(text: str, bb, confidence: float, width: int, height: int, *, line: bool) -> Word:
    """A `Word` from a Vision bounding box (normalized, bottom-left origin)."""
    bx, by = float(bb.origin.x), float(bb.origin.y)
    bw, bh = float(bb.size.width), float(bb.size.height)
    return Word(
        text=text, x=bx * width, y=(1.0 - by - bh) * height, w=bw * width, h=bh * height,
        confidence=confidence, line=line,
    )


def _utf16_len(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def _token_words(vision, cand, text: str, confidence: float, width: int, height: int) -> list[Word]:
    """One `Word` per whitespace token of `text`, boxed by asking the
    candidate for the sub-range (NSRange is in UTF-16 units). A single-
    token line is already represented by its line `Word`. A token Vision
    can't box (error, or no such method) is skipped, never fatal."""
    tokens = list(_TOKEN.finditer(text))
    if len(tokens) < 2:
        return []
    out: list[Word] = []
    for m in tokens:
        loc, length = _utf16_len(text[:m.start()]), _utf16_len(m.group())
        try:
            box, err = cand.boundingBoxForRange_error_(vision.NSMakeRange(loc, length), None)
        except Exception as e:  # noqa: BLE001 — a missing box is not a failed OCR
            log.debug("no box for %r: %s", m.group(), e)
            continue
        if err is not None or box is None:
            log.debug("no box for %r: %s", m.group(), err)
            continue
        out.append(_word(m.group(), box.boundingBox(), confidence, width, height, line=False))
    return out


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
