"""Small ONNX models fetched on demand into Settings.models_dir.

Each is pinned by SHA256: a download that doesn't match is deleted, never
loaded. scripts/download_models.py fetches them up front; the app fetches a
missing one the first time it is needed (the voice-isolation spec,
docs/superpowers/specs/2026-10-01-veronica-voice-isolation-design.md)."""
import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("veronica.audio")

_SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


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


def ensure(model: ModelFile, models_dir: Path, *, fetch=urllib.request.urlretrieve) -> Path:
    """The model's path, downloading and verifying it first if it isn't
    there. Raises on a network failure or a checksum mismatch (the partial
    file is removed either way)."""
    dest = models_dir / model.name
    if dest.exists():
        return dest
    models_dir.mkdir(parents=True, exist_ok=True)
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
