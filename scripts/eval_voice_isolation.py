"""Measure noise suppression and the speaker check on synthetic audio.

    uv run python scripts/eval_voice_isolation.py [--out DIR] [--quick]

Synthesises commands with the Kokoro voices (so no microphone is touched),
reverberates them, mixes in pink noise / babble / a TV-like bed at several
SNRs, and reports — with the code Veronica actually runs (Denoiser,
SpeakerGate): how often noise alone opens a capture before (raw audio, VAD
only) and now (suppressed audio + the speech level floor), whether real
speech near and far still gets in, faster-whisper small.en word error on raw
vs suppressed audio (the reason whisper gets the raw audio), and speaker-
check acceptance for the enrolled voice, other voices and short answers. The clips land in
DIR (default ~/.veronica/eval, outside git; ~40 MB). TTS voices are a sanity
check, not a speaker-verification benchmark: real people differ more than
Kokoro's blended voices, and real rooms are messier than this reverb.
Numbers from the run behind the defaults are in
docs/superpowers/specs/2026-10-01-veronica-voice-isolation-design.md."""
import argparse
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import webrtcvad
from scipy.signal import fftconvolve, resample_poly

from veronica.audio import denoise, models
from veronica.audio.speaker import SpeakerGate, SpeakerModel
from veronica.config import Settings, settings
from veronica.orchestrator import Orchestrator

COMMANDS = [
    "What time is it in London right now", "Set a timer for ten minutes",
    "Open the calendar and show me tomorrow", "Play some jazz music please",
    "Remind me to call my mother at six", "What is the weather like today",
    "Send a message to Rahul saying I am late", "Turn the volume down a little",
    "Search the web for flights to Delhi", "Read my latest email",
]
BABBLE = [
    "I told him the meeting was moved to Thursday but nobody listened",
    "The prices at the market have gone up again this month",
    "Did you see the match last night it was unbelievable",
    "We should repaint the kitchen before the guests arrive",
]
SHORT = ["Yes.", "No.", "Go ahead.", "Yes, do it."]
TARGETS = ["af_sarah", "am_adam"]
OTHERS = ["af_bella", "am_michael", "bf_emma", "bm_george"]
SPEECH_RMS = 0.01
rng = np.random.default_rng(7)


def rmsv(x) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))


def at(x, level):
    return (x / (rmsv(x) + 1e-12) * level).astype(np.float32)


def i16(x):
    return np.clip(np.asarray(x) * 32767, -32768, 32767).astype(np.int16)


class Clips:
    def __init__(self, out: Path):
        from kokoro_onnx import Kokoro

        self.out = out
        self.k = Kokoro(str(settings.models_dir / "kokoro-v1.0.onnx"), str(settings.models_dir / "voices-v1.0.bin"))

    def say(self, text: str, voice: str) -> np.ndarray:
        p = self.out / "clips" / f"{voice}_{re.sub(r'[^a-z]+', '_', text.lower()).strip('_')}.wav"
        if p.exists():
            return sf.read(p, dtype="float32")[0]
        s, _sr = self.k.create(text, voice=voice, lang="en-gb" if voice.startswith("b") else "en-us")
        x = resample_poly(np.asarray(s, np.float32), 2, 3)
        n = int(0.35 * 16000)   # a small room: exponential tail, strong direct path
        ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
        ir[0] = 3.0
        x = fftconvolve(x, ir)[: x.size]
        x = (x / np.max(np.abs(x)) * 0.5).astype(np.float32)
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, i16(x), 16000)
        return x

    def noise(self, name: str, seconds: int = 40) -> np.ndarray:
        p = self.out / "clips" / f"noise_{name}.wav"
        if p.exists():
            return sf.read(p, dtype="float32")[0]
        n = seconds * 16000
        t = np.arange(n) / 16000
        if name in ("pink", "clatter"):
            f = np.fft.rfftfreq(n)
            spec = rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)
            spec[1:] /= np.sqrt(f[1:])
            spec[0] = 0
            x = np.fft.irfft(spec, n)
            if name == "clatter":   # keyboard / dishes: short decaying bursts over a faint floor
                x *= 0.02
                pos = 0
                while pos < n - 2000:
                    pos += int(rng.integers(800, 8000))
                    ln = min(int(rng.integers(200, 1600)), n - pos)
                    x[pos:pos + ln] += rng.standard_normal(ln) * np.exp(-np.arange(ln) / (ln / 5))
        elif name == "music":   # plucked chords, two a second
            x = np.zeros(n)
            for j in range(seconds * 2):
                seg = (t >= j / 2) & (t < j / 2 + 0.5)
                for mult in (1, 1.26, 1.5, 2, 3):
                    x[seg] += np.sin(2 * np.pi * (220, 262, 196, 175)[j % 4] * mult * t[seg]) \
                        * np.exp(-4 * (t[seg] - j / 2)) / mult
        else:
            voices = OTHERS if name == "babble" else ["am_onyx"]
            x = np.zeros(n)
            for i, v in enumerate(voices):
                pos = int(rng.integers(0, 16000))
                while pos < n:
                    s = self.say(BABBLE[i % len(BABBLE)], v)
                    end = min(n, pos + s.size)
                    x[pos:end] += s[: end - pos] / rmsv(s)
                    pos = end + int(rng.integers(1000, 6000))
            if name == "tv":   # a presenter over a music bed
                x += 0.3 * np.sin(2 * np.pi * 220 * t) * np.exp(-3 * (t % 1.0))
        x = at(x, 0.1)
        sf.write(p, i16(x), 16000)
        return x


def opens(x16: np.ndarray, s: Settings, aggr: int) -> bool:
    """Recorder._capture's decision, reduced: would this open a capture?"""
    v = webrtcvad.Vad(aggr)
    started, speech, sil = False, 0, 0
    for i in range(0, x16.size - 479, 480):
        f = x16[i:i + 480]
        sp = v.is_speech(f.tobytes(), 16000) and rmsv(f / 32768) >= s.vad_min_rms
        if not started:
            if sp:
                started, speech, sil = True, 1, 0
            continue
        speech, sil = (speech + 1, 0) if sp else (speech, sil + 1)
        if sil >= s.vad_silence_ms // 30:
            if speech >= s.min_speech_ms // 30:
                return True
            started = False
    return started and speech >= s.min_speech_ms // 30


def opens_now(raw16: np.ndarray, s: Settings) -> bool:
    """What ships: the VAD and the level floor on the suppressed audio."""
    return opens(suppress(raw16), s, s.vad_aggressiveness)


def opens_before(raw16: np.ndarray, s: Settings) -> bool:
    """Before voice isolation: the VAD alone on the raw audio."""
    return opens(raw16, s.model_copy(update={"vad_min_rms": 0.0}), s.vad_aggressiveness)


def suppress(x16: np.ndarray) -> np.ndarray:
    d = denoise.Denoiser(denoise._session(str(models.ensure(models.GTCRN, settings.models_dir))))
    y = np.concatenate([d.process(x16[i:i + 480]) for i in range(0, x16.size, 480)])
    return np.concatenate([y[denoise.LAG:], np.zeros(denoise.LAG, np.int16)])


def words(t: str) -> list[str]:
    nums = {"6": "six", "10": "ten"}
    return [nums.get(w, w) for w in re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()]


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[:], i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)


def eval_noise(c: Clips, quick: bool) -> dict:
    from faster_whisper import WhisperModel

    stt = WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")

    def text(x16):
        segs, _ = stt.transcribe(x16.astype(np.float32) / 32768, beam_size=1, language="en", vad_filter=False)
        return " ".join(s.text.strip() for s in segs)

    noises = {n: c.noise(n) for n in ("pink", "clatter", "music", "babble", "tv")}
    out: dict = {}
    s = Settings()
    for n, x in noises.items():
        for level in (0.002, 0.005, 0.01):
            for _ in range(4 if quick else 12):
                off = int(rng.integers(0, x.size - 64000))
                raw = i16(at(x[off:off + 64000], level))
                for tag, fn in (("before", opens_before), ("now", opens_now)):
                    acc = out.setdefault(f"noise opens a capture: {n} @{level} {tag}", [0, 0])
                    acc[0] += fn(raw, s)
                    acc[1] += 1
    cmds = COMMANDS[:4] if quick else COMMANDS
    for v in TARGETS:
        for cmd in cmds:
            for level in (0.01, 0.003, 0.0015):   # near, across the room, far
                sp = np.concatenate([np.zeros(8000, np.float32), at(c.say(cmd, v), level), np.zeros(24000, np.float32)])
                raw = i16(sp + at(noises["clatter"][: sp.size], level / 10 ** 0.75))
                for tag, fn in (("before", opens_before), ("now", opens_now)):
                    acc = out.setdefault(f"speech missed: @{level} over clatter {tag}", [0, 0])
                    acc[0] += not fn(raw, s)
                    acc[1] += 1
            sp = at(c.say(cmd, v), SPEECH_RMS)
            sp = np.concatenate([np.zeros(8000, np.float32), sp, np.zeros(8000, np.float32)])
            for n, snr in [("clean", None)] + [(n, snr) for n in ("pink", "babble", "tv") for snr in (10, 5, 0)]:
                x = sp
                if n != "clean":
                    off = int(rng.integers(0, noises[n].size - sp.size))
                    x = sp + at(noises[n][off:off + sp.size], SPEECH_RMS / 10 ** (snr / 20))
                raw = i16(x)
                for tag, a in (("raw (what whisper gets)", raw), ("suppressed", suppress(raw))):
                    e, r = wer(cmd, text(a))
                    acc = out.setdefault(f"word error: {n}{'' if snr is None else f' {snr}dB'} {tag}", [0, 0])
                    acc[0] += e
                    acc[1] += r
    return {k: round(a / b, 3) for k, (a, b) in out.items()}


def eval_speaker(c: Clips) -> dict:
    # a throwaway home for the profiles; the model from the real models dir
    gate = SpeakerGate(Settings(home=Path(tempfile.mkdtemp())))
    gate._model = SpeakerModel(models.ensure(models.CAMPPLUS, settings.models_dir))
    babble = c.noise("babble")

    def capture(x):   # as the recorder hands it over: lead-in, speech, trailing silence
        return i16(np.concatenate([np.zeros(4800, np.float32), at(x, SPEECH_RMS), np.zeros(19200, np.float32)]))

    out: dict = {}
    voices = TARGETS + OTHERS
    profiles = {}
    for v in voices:
        profiles[v], _agreement = gate.enrol([capture(c.say(line, v)) for line in Orchestrator.ENROL_LINES])
        assert profiles[v] is not None, f"{v}: enrolment clips disagreed"
    for v in voices:
        for kind, texts in (("command", COMMANDS[:3]), ("short", SHORT)):
            for t in texts:
                pcm = capture(c.say(t, v))
                noisy = pcm + i16(at(babble[: pcm.size], SPEECH_RMS / 10 ** 0.5))
                for u in voices:
                    gate.profile = profiles[u]
                    who = "me" if u == v else "other"
                    out.setdefault(f"{kind} {who}", []).append(gate.check(pcm, kind))
                    if who == "me":
                        out.setdefault(f"{kind} me babble10", []).append(gate.check(noisy, kind))
    return {k: {"accepted": round(float(np.mean([ok for ok, _ in v])), 3),
                "score mean": round(float(np.mean([sc for _, sc in v])), 3)} for k, v in out.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=settings.home / "eval")
    p.add_argument("--quick", action="store_true", help="fewer clips (a few minutes instead of ~half an hour)")
    args = p.parse_args()
    c = Clips(args.out)
    report = {"speaker": eval_speaker(c), "noise": eval_noise(c, args.quick)}
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    for section, rows in report.items():
        print(f"\n{section}")
        for k, v in rows.items():
            print(f"  {k:40s} {v}")


if __name__ == "__main__":
    main()
