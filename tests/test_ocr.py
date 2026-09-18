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


class FakeCandidate:
    def __init__(self, s, c):
        self._s, self._c = s, c

    def string(self):
        return self._s

    def confidence(self):
        return self._c


class FakeObservation:
    def __init__(self, text, box, conf=0.9):
        self._c, self._box = FakeCandidate(text, conf), _Box(*box)

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


def test_recognize_text_skips_observations_without_candidates():
    class Empty(FakeObservation):
        def topCandidates_(self, n):
            return []

    v = FakeVision([Empty("", (0, 0, 1, 1)), FakeObservation("a", (0, 0, 1, 1))])
    assert [w.text for w in ocr.recognize_text(PNG, vision=v, image_size=(10, 10))] == ["a"]


def _w(text, x=0.0, y=0.0):
    return Word(text=text, x=x, y=y, w=10.0, h=10.0, confidence=1.0)


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
