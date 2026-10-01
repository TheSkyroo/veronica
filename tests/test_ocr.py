from pathlib import Path

import pytest

from veronica.tools import ocr
from veronica.tools.ocr import Word


class _Pt:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _Sz:
    def __init__(self, w, h):
        self.width, self.height = w, h


class _Box:
    def __init__(self, x, y, w, h):
        self.origin, self.size = _Pt(x, y), _Sz(w, h)


class _RangeBox:
    """What VNRecognizedText.boundingBoxForRange_error_ returns: a
    VNRectangleObservation with .boundingBox()."""
    def __init__(self, box):
        self._box = _Box(*box)

    def boundingBox(self):
        return self._box


class FakeCandidate:
    """`token_boxes` maps a token's text to its normalized box; a token
    without one gets an error back, like Vision does for a bad range.
    The range is in UTF-16 units (NSRange), as PyObjC hands it over."""
    def __init__(self, s, c, token_boxes=None):
        self._s, self._c = s, c
        self._boxes = dict(token_boxes or {})
        self.ranges: list[tuple[int, int]] = []

    def string(self):
        return self._s

    def confidence(self):
        return self._c

    def boundingBoxForRange_error_(self, rng, err):
        loc, length = rng
        self.ranges.append((loc, length))
        units = self._s.encode("utf-16-le")
        token = units[loc * 2:(loc + length) * 2].decode("utf-16-le")
        if token not in self._boxes:
            return None, "no box for range"
        return _RangeBox(self._boxes[token]), None


class FakeObservation:
    def __init__(self, text, box, conf=0.9, token_boxes=None):
        self._c, self._box = FakeCandidate(text, conf, token_boxes), _Box(*box)

    def topCandidates_(self, n):
        return [self._c]

    def boundingBox(self):
        return self._box


class FakeRequest:
    def __init__(self, vision):
        self.vision = vision
        self.level = None
        self.correction = None
        self.languages = None
        self._results = []

    def setRecognitionLevel_(self, level):
        self.level = level

    def setUsesLanguageCorrection_(self, flag):
        self.correction = flag

    def supportedRecognitionLanguagesAndReturnError_(self, err):
        if self.vision.supported_error:
            return None, "boom"
        return list(self.vision.supported), None

    def setRecognitionLanguages_(self, langs):
        if self.vision.set_languages_raises:
            raise ValueError("bad languages")
        self.languages = list(langs)

    def results(self):
        return self._results


class FakeHandler:
    def __init__(self, vision, url):
        self.vision, self.url = vision, url

    def performRequests_error_(self, reqs, err):
        self.vision.performed.append((self.url, reqs))
        if self.vision.perform_fails:
            return False, "vision failed"
        for r in reqs:
            r._results = list(self.vision.observations)
        return True, None


class _Alloc:
    def __init__(self, factory):
        self._factory = factory

    def alloc(self):
        return self

    def init(self):
        return self._factory()

    def initWithURL_options_(self, url, opts):
        return self._factory(url)


class FakeNSURL:
    @staticmethod
    def fileURLWithPath_(p):
        return f"file://{p}"


class FakeVision:
    VNRequestTextRecognitionLevelAccurate = "accurate"
    NSURL = FakeNSURL
    NSMakeRange = staticmethod(lambda loc, length: (loc, length))

    def __init__(self, observations=(), supported=("en-US", "hi-IN"), *, supported_error=False,
                 set_languages_raises=False, perform_fails=False):
        self.observations = list(observations)
        self.supported = supported
        self.supported_error = supported_error
        self.set_languages_raises = set_languages_raises
        self.perform_fails = perform_fails
        self.performed = []
        self.requests = []
        self.VNRecognizeTextRequest = _Alloc(self._new_request)
        self.VNImageRequestHandler = _Alloc(lambda url: FakeHandler(self, url))

    def _new_request(self):
        r = FakeRequest(self)
        self.requests.append(r)
        return r


PNG = Path("/tmp/does-not-matter.png")


def test_recognize_text_flips_y_and_scales_to_pixels():
    # 1568x1019 image; a box at normalized (0.1, 0.9) size (0.2, 0.05), bottom-left origin
    v = FakeVision([FakeObservation("Save", (0.1, 0.9, 0.2, 0.05), conf=0.75)])
    words = ocr.recognize_text(PNG, vision=v, image_size=(1568, 1019))
    assert len(words) == 1
    w = words[0]
    assert w.text == "Save" and w.confidence == pytest.approx(0.75)
    assert w.x == pytest.approx(156.8)
    assert w.w == pytest.approx(313.6)
    assert w.h == pytest.approx(50.95)
    # top edge in top-left pixels: (1 - 0.9 - 0.05) * 1019
    assert w.y == pytest.approx(0.05 * 1019)
    assert w.center == (pytest.approx(156.8 + 313.6 / 2), pytest.approx(0.05 * 1019 + 50.95 / 2))


def test_recognize_text_configures_request_and_handler():
    v = FakeVision([])
    ocr.recognize_text(PNG, vision=v, image_size=(100, 100))
    (req,) = v.requests
    assert req.level == "accurate" and req.correction is True
    assert req.languages == ["en-US", "hi-IN"]
    assert v.performed[0][0] == f"file://{PNG}"
    assert v.performed[0][1] == [req]


def test_recognize_text_filters_unsupported_languages():
    v = FakeVision([], supported=("en-US", "fr-FR"))
    ocr.recognize_text(PNG, vision=v, image_size=(100, 100))
    assert v.requests[0].languages == ["en-US"]


def test_recognize_text_language_failures_are_ignored():
    v = FakeVision([FakeObservation("ok", (0, 0, 1, 1))], supported_error=True)
    assert [w.text for w in ocr.recognize_text(PNG, vision=v, image_size=(10, 10))] == ["ok"]
    assert v.requests[0].languages is None
    v = FakeVision([FakeObservation("ok", (0, 0, 1, 1))], set_languages_raises=True)
    assert [w.text for w in ocr.recognize_text(PNG, vision=v, image_size=(10, 10))] == ["ok"]


def test_recognize_text_perform_failure_raises():
    v = FakeVision([], perform_fails=True)
    with pytest.raises(RuntimeError, match="vision failed"):
        ocr.recognize_text(PNG, vision=v, image_size=(10, 10))


def test_recognize_text_image_size_from_geometry(monkeypatch):
    from veronica.tools import screen
    g = screen.Geometry(region="screen", image_w=200, image_h=100, origin_x=0, origin_y=0,
                        width_pt=200, height_pt=100, scale=1.0, captured_at=0.0, window=None)
    monkeypatch.setattr(ocr, "load_geometry", lambda: g)
    v = FakeVision([FakeObservation("x", (0.5, 0.5, 0.5, 0.5))])
    (w,) = ocr.recognize_text(PNG, vision=v)
    assert (w.x, w.y, w.w, w.h) == (100.0, 0.0, 100.0, 50.0)


def test_recognize_text_image_size_from_png_header(tmp_path, monkeypatch):
    import struct
    p = tmp_path / "a.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 400, 300) + b"\x08\x06\x00\x00\x00")
    monkeypatch.setattr(ocr, "load_geometry", lambda: None)
    v = FakeVision([FakeObservation("x", (0, 0, 1, 1))])
    (w,) = ocr.recognize_text(p, vision=v)
    assert (w.w, w.h) == (400.0, 300.0)


def test_recognize_text_without_any_size_is_error(monkeypatch):
    monkeypatch.setattr(ocr, "load_geometry", lambda: None)
    with pytest.raises(ValueError):
        ocr.recognize_text(PNG, vision=FakeVision([]))


# --- per-token boxes ---------------------------------------------------------

def test_recognize_text_emits_a_word_per_token_plus_the_line():
    # 100x100 image; the line spans x 0.1..0.7, its tokens sit inside it
    v = FakeVision([FakeObservation(
        "Cancel Save", (0.1, 0.8, 0.6, 0.1), conf=0.9,
        token_boxes={"Cancel": (0.1, 0.8, 0.3, 0.1), "Save": (0.5, 0.8, 0.2, 0.1)},
    )])
    words = ocr.recognize_text(PNG, vision=v, image_size=(100, 100))
    assert [(w.text, w.line) for w in words] == [("Cancel Save", True), ("Cancel", False), ("Save", False)]
    line, cancel, save = words
    assert (line.x, line.w) == (pytest.approx(10.0), pytest.approx(60.0))
    assert (cancel.x, cancel.w) == (pytest.approx(10.0), pytest.approx(30.0))
    assert (save.x, save.w) == (pytest.approx(50.0), pytest.approx(20.0))
    assert save.y == pytest.approx((1 - 0.8 - 0.1) * 100) and save.h == pytest.approx(10.0)
    assert save.confidence == pytest.approx(0.9)
    assert v.observations[0]._c.ranges == [(0, 6), (7, 4)]


def test_recognize_text_token_ranges_are_utf16_units():
    # an astral character counts as two UTF-16 units before the next token
    v = FakeVision([FakeObservation(
        "\U0001F600 Save", (0, 0, 1, 1),
        token_boxes={"\U0001F600": (0, 0, 0.5, 1), "Save": (0.5, 0, 0.5, 1)},
    )])
    words = ocr.recognize_text(PNG, vision=v, image_size=(100, 100))
    assert [w.text for w in words] == ["\U0001F600 Save", "\U0001F600", "Save"]
    assert v.observations[0]._c.ranges == [(0, 2), (3, 4)]


def test_recognize_text_single_token_line_is_not_duplicated():
    v = FakeVision([FakeObservation("Save", (0.1, 0.9, 0.2, 0.05), token_boxes={"Save": (0.1, 0.9, 0.2, 0.05)})])
    words = ocr.recognize_text(PNG, vision=v, image_size=(100, 100))
    assert [(w.text, w.line) for w in words] == [("Save", False)]   # a lone label is a token, not a line
    assert v.observations[0]._c.ranges == []


def test_recognize_text_token_without_a_box_is_skipped():
    v = FakeVision([FakeObservation("Cancel Save", (0, 0, 1, 1), token_boxes={"Save": (0.5, 0, 0.5, 1)})])
    words = ocr.recognize_text(PNG, vision=v, image_size=(100, 100))
    assert [(w.text, w.line) for w in words] == [("Cancel Save", True), ("Save", False)]


def test_recognize_text_survives_a_candidate_without_range_boxes():
    class Old(FakeCandidate):
        boundingBoxForRange_error_ = None
    obs = FakeObservation("Cancel Save", (0, 0, 1, 1))
    obs._c = Old("Cancel Save", 0.9)
    words = ocr.recognize_text(PNG, vision=FakeVision([obs]), image_size=(100, 100))
    assert [(w.text, w.line) for w in words] == [("Cancel Save", True)]


def test_find_text_save_on_cancel_save_line_returns_the_token_centre():
    v = FakeVision([FakeObservation(
        "Cancel Save", (0.1, 0.8, 0.6, 0.1),
        token_boxes={"Cancel": (0.1, 0.8, 0.3, 0.1), "Save": (0.5, 0.8, 0.2, 0.1)},
    )])
    words = ocr.recognize_text(PNG, vision=v, image_size=(100, 100))
    (m,) = ocr.find_text(words, "Save")
    assert m.text == "Save" and not m.line
    assert m.center == (pytest.approx(60.0), pytest.approx(15.0))    # not the line's (40, 15)
    (m,) = ocr.find_text(words, "Cancel Save")
    assert m.line and m.center == (pytest.approx(40.0), pytest.approx(15.0))


def test_recognize_text_skips_observations_without_candidates():
    class Empty(FakeObservation):
        def topCandidates_(self, n):
            return []

    v = FakeVision([Empty("", (0, 0, 1, 1)), FakeObservation("a", (0, 0, 1, 1))])
    assert [w.text for w in ocr.recognize_text(PNG, vision=v, image_size=(10, 10))] == ["a"]


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
