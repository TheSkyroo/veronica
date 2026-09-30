"""Small ONNX models fetched on demand into Settings.models_dir.

Each is pinned by SHA256: a download that doesn't match is deleted, never
loaded. scripts/download_models.py fetches them up front; the app fetches a
missing one the first time it is needed (the voice-isolation spec,
docs/superpowers/specs/2026-10-01-veronica-voice-isolation-design.md)."""
import hashlib
import logging
import shutil
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("veronica.audio")

_SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
# Seconds a stalled connection may sit before the download gives up (per
# socket operation, so a slow but moving download still finishes).
TIMEOUT_S = 30
# One download at a time: startup, a Settings toggle and an enrolment may all
# ask for a model at once, and must not write the same .part file.
_lock = threading.Lock()


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    sha256: str


# GTCRN (MIT, github.com/Xiaobin-Rong/gtcrn): 48k-parameter streaming speech
# enhancement at 16 kHz, the sherpa-onnx export. ~0.5 MB.
GTCRN = ModelFile(
    "gtcrn_simple.onnx",
    f"{_SHERPA}/speech-enhancement-models/gtcrn_simple.onnx",
    "e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534",
)
# CAM++ speaker embeddings (Apache-2.0, 3D-Speaker, trained on VoxCeleb),
# the sherpa-onnx export. ~29 MB. ("recongition" is sherpa's own spelling.)
CAMPPLUS = ModelFile(
    "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
    f"{_SHERPA}/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
    "357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b",
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, with TIMEOUT_S on the connection and on each
    read (urlretrieve has neither, and could hang a thread forever)."""
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)


def ensure(model: ModelFile, models_dir: Path, *, fetch=None) -> Path:
    """The model's path, downloading and verifying it first if it isn't
    there. Raises on a network failure or a checksum mismatch (the partial
    file is removed either way). Blocks for as long as a download takes:
    call it from startup, enrolment or a background thread — never from a
    turn's capture path."""
    dest = models_dir / model.name
    if dest.exists():
        return dest
    with _lock:
        if dest.exists():   # another thread fetched it while we waited
            return dest
        return _fetch(model, dest, fetch or download)


def _fetch(model: ModelFile, dest: Path, fetch) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    log.info("fetching %s", model.url)
    try:
        fetch(model.url, part)
        got = sha256_of(part)
        if got != model.sha256:
            raise ValueError(f"{model.name}: checksum {got} does not match {model.sha256}")
        part.rename(dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return dest
