"""Benchmark a TTS candidate the way Veronica uses TTS, and render listening samples.

    python scripts/tts_bench.py <candidate> [--out samples] [--runs 5]
    python scripts/tts_bench.py --list

Veronica synthesises one whole sentence at a time and starts playing it while
the next one is synthesised (orchestrator.handle_text), so the latency that
matters is the time to synthesise the first sentence of a reply, not a
chunk-streaming first-packet time. This script measures, for one candidate:

  load_s       model load (cold, includes graph build / weight mmap)
  first_call_s first synth after load (warm-up cost a fresh process pays once)
  ttfa_s       median of --runs syntheses of a typical first sentence
               (English; and a Hindi one when the candidate speaks Hindi)
  rtf          synth seconds / audio seconds over all sample lines (<1 = faster
               than real time)
  cpu_cores    process CPU seconds / wall seconds while synthesising (GPU work
               under MLX / MPS is not in this number)
  footprint_mb peak physical footprint (what Activity Monitor shows; includes
               Metal buffers on Apple Silicon unlike ru_maxrss)

and writes one WAV per sample line plus metrics.json to <out>/<candidate>/.
It never plays audio. Each candidate may need its own venv (the heavy ones pull
torch or MLX); the engine is imported only when that candidate runs. Models
are fetched into .spike/ (gitignored) or the HF cache that HF_HOME points to.
Numbers from the run behind the recommendation are in
docs/superpowers/specs/2026-10-01-veronica-voice-model-options.md."""
import argparse
import ctypes
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SPIKE = ROOT / ".spike" / "models"

# The fixed listening set. Hindi lines are rendered only by candidates that
# speak Hindi.
LINES: list[tuple[str, str, str]] = [
    ("01_greeting", "en", "Good morning! What can I do for you today?"),
    ("02_answer", "en",
     "The capital of Australia is Canberra, not Sydney as many people assume. "
     "It was chosen in 1908 as a compromise between Sydney and Melbourne."),
    ("03_confirm", "en", "Copy to clipboard? Say yes, no, or always."),
    ("04_numbers", "en",
     "Your flight leaves at 6:40 AM on Tuesday, October 14th, from gate 23B. "
     "Two tickets came to $1,249.50, and the drive takes about 45 minutes."),
    ("05_hindi_greeting", "hi", "नमस्ते! आज मौसम बहुत सुहावना है, क्या आप बाहर घूमने चलना चाहेंगे?"),
    ("06_hindi_answer", "hi",
     "आपकी मीटिंग शाम चार बजे है। क्या मैं आपको पंद्रह मिनट पहले याद दिला दूँ?"),
]
TTFA_EN = "The capital of Australia is Canberra, not Sydney as many people assume."
TTFA_HI = "आपकी मीटिंग शाम चार बजे है।"
# Same boundary rule as veronica.brain.sentences (copied so the script runs in
# venvs that don't have Veronica installed).
_END = re.compile(r"(?<=[.!?।])\s+(?=[A-Z0-9ऀ-ॿ])")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _END.split(text) if s.strip()]


# --- process metrics ------------------------------------------------------

class _RUsage(ctypes.Structure):  # rusage_info_v4 (sys/resource.h)
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("f", ctypes.c_uint64 * 40)]


_libc = ctypes.CDLL("libc.dylib") if sys.platform == "darwin" else None


def footprint_mb() -> tuple[float, float]:
    """(current, lifetime peak) physical footprint in MB, macOS only."""
    if _libc is None:
        return float("nan"), float("nan")
    ru = _RUsage()
    if _libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) != 0:
        return float("nan"), float("nan")
    return ru.f[7] / 2**20, ru.f[28] / 2**20


def cpu_s() -> float:
    t = os.times()
    return t.user + t.system


# --- engines --------------------------------------------------------------

class Engine:
    hindi = False
    sr = 24000
    files: list[Path] = []  # weights on disk, for the size column
    licence = "?"

    def synth(self, text: str, lang: str) -> np.ndarray:
        raise NotImplementedError


def _voice_mix(get, spec: str):
    """'af_bella' or 'af_bella*0.6+af_heart*0.4' -> style vector."""
    parts = []
    for term in spec.split("+"):
        name, _, w = term.partition("*")
        parts.append((name.strip(), float(w) if w else 1.0))
    total = sum(w for _, w in parts)
    return sum(get(n) * (w / total) for n, w in parts)


class KokoroOnnx(Engine):
    licence = "Apache-2.0 (weights + kokoro-onnx MIT)"

    def __init__(self, voice: str, hindi_voice: str | None, model: str = "kokoro-v1.0.onnx",
                 voices: str = "voices-v1.0.bin", speed: float = 1.0):
        from kokoro_onnx import Kokoro
        d = SPIKE / "kokoro"
        self.k = Kokoro(str(d / model), str(d / voices))
        self.files = [d / model, d / voices]
        self.speed = speed
        self.v_en = self._v(voice)
        self.v_hi = self._v(hindi_voice) if hindi_voice else None
        self.hindi = hindi_voice is not None

    def _v(self, spec):
        return spec if "+" not in spec and "*" not in spec else _voice_mix(self.k.get_voice_style, spec)

    def synth(self, text, lang):
        v, kl = (self.v_hi, "hi") if lang == "hi" else (self.v_en, "en-us")
        s, self.sr = self.k.create(text, voice=v, speed=self.speed, lang=kl)
        return np.asarray(s, np.float32)


class Piper(Engine):
    licence = "piper1-gpl GPL-3.0 (engine); voice: see its MODEL_CARD"

    def __init__(self, voice: str, hindi_voice: str | None = None):
        from piper import PiperVoice
        d = SPIKE / "piper"
        self.v = {"en": PiperVoice.load(str(d / f"{voice}.onnx"))}
        self.files = [d / f"{voice}.onnx"]
        if hindi_voice:
            self.v["hi"] = PiperVoice.load(str(d / f"{hindi_voice}.onnx"))
            self.files.append(d / f"{hindi_voice}.onnx")
            self.hindi = True

    def synth(self, text, lang):
        chunks = list(self.v[lang].synthesize(text))
        self.sr = chunks[0].sample_rate
        return np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)


class Supertonic(Engine):
    licence = "OpenRAIL-M (weights), MIT (code)"
    hindi = True

    def __init__(self, voice: str, hindi_voice: str, steps: int = 8):
        from supertonic import TTS
        d = SPIKE / "supertonic3"
        self.t = TTS(model="supertonic-3", model_dir=d)
        self.files = sorted(d.rglob("*.onnx")) + sorted(d.rglob("*.json"))
        self.sr = self.t.sample_rate
        self.st = {"en": self.t.get_voice_style(voice), "hi": self.t.get_voice_style(hindi_voice)}
        self.steps = steps

    def synth(self, text, lang):
        wav, _ = self.t.synthesize(text, voice_style=self.st[lang], lang=lang, total_steps=self.steps)
        return np.asarray(wav, np.float32).reshape(-1)


class Mlx(Engine):
    """Any mlx-audio TTS model. `gen` holds per-language generate() kwargs;
    a language missing from it is not spoken by this candidate."""

    def __init__(self, repo: str, gen: dict, licence: str = "?"):
        from huggingface_hub import snapshot_download
        from mlx_audio.tts.utils import load_model
        self.path = Path(snapshot_download(repo))
        self.m = load_model(repo)  # model type is inferred from the repo name
        self.files = [p for p in self.path.rglob("*") if p.is_file()]
        self.gen, self.licence = gen, licence
        self.hindi = "hi" in gen
        self.sr = self.m.sample_rate

    def synth(self, text, lang):
        import mlx.core as mx
        out = [np.asarray(r.audio, np.float32).reshape(-1) for r in self.m.generate(text=text, **self.gen[lang])]
        mx.clear_cache()
        return np.concatenate(out)


class Mms(Engine):
    """Meta MMS-TTS (VITS, one small model per language) via transformers, CPU."""
    licence = "CC-BY-NC-4.0"
    hindi = True

    def __init__(self):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer, VitsModel
        torch.manual_seed(0)
        self.torch, self.m, self.tok, self.files = torch, {}, {}, []
        for lang, repo in (("en", "facebook/mms-tts-eng"), ("hi", "facebook/mms-tts-hin")):
            self.m[lang] = VitsModel.from_pretrained(repo).eval()
            self.tok[lang] = AutoTokenizer.from_pretrained(repo)
            self.files += list(Path(snapshot_download(repo)).glob("*.safetensors"))
        self.sr = self.m["en"].config.sampling_rate

    def synth(self, text, lang):
        with self.torch.no_grad():
            out = self.m[lang](**self.tok[lang](text, return_tensors="pt")).waveform
        return out[0].numpy().astype(np.float32)


# name -> (factory, note). Voice names follow each engine's own catalogue.
CANDIDATES: dict[str, tuple] = {
    # Baseline: exactly what Veronica ships (kokoro-onnx, fp32, espeak G2P).
    "kokoro-onnx_af_bella": (lambda: KokoroOnnx("af_bella", "hf_alpha"), "current production voice pair"),
    "kokoro-onnx_af_sarah": (lambda: KokoroOnnx("af_sarah", "hf_alpha"), "config.py default voice"),
    "kokoro-onnx_af_heart": (lambda: KokoroOnnx("af_heart", "hf_beta"), "Kokoro's top-graded voice (A)"),
    "kokoro-onnx_blend_heart-bella": (lambda: KokoroOnnx("af_heart*0.6+af_bella*0.4", "hf_beta*0.7+hf_alpha*0.3"),
                                      "style-vector blend"),
    "kokoro-onnx_blend_bella-nicole": (lambda: KokoroOnnx("af_bella*0.7+af_nicole*0.3", None), "calmer blend"),
    "kokoro-onnx_int8_af_bella": (lambda: KokoroOnnx("af_bella", "hf_alpha", model="kokoro-v1.0.int8.onnx"),
                                  "int8 quantised weights"),
    "kokoro-onnx_fp16_af_bella": (lambda: KokoroOnnx("af_bella", "hf_alpha", model="kokoro-v1.0.fp16.onnx"),
                                  "fp16 weights"),
    "kokoro-onnx_hindi_voices": (lambda: KokoroOnnx("af_bella", "hf_beta"), "Hindi voice hf_beta"),
    "kokoro-onnx_hm_omega": (lambda: KokoroOnnx("af_bella", "hm_omega"), "Hindi male"),
    "kokoro-onnx_hm_psi": (lambda: KokoroOnnx("af_bella", "hm_psi"), "Hindi male"),
    "kokoro-onnx_hf_blend_beta-alpha": (lambda: KokoroOnnx("af_bella", "hf_beta*0.6+hf_alpha*0.4"), "Hindi blend"),
    # Official Kokoro pipeline on MLX (misaki G2P for English, GPU).
    "kokoro-mlx_af_heart": (lambda: Mlx("mlx-community/Kokoro-82M-bf16", {
        "en": dict(voice="af_heart", lang_code="a"), "hi": dict(voice="hf_beta", lang_code="h")},
        "Apache-2.0"), "official misaki G2P, MLX bf16"),
    "kokoro-mlx_af_bella": (lambda: Mlx("mlx-community/Kokoro-82M-bf16", {
        "en": dict(voice="af_bella", lang_code="a"), "hi": dict(voice="hf_alpha", lang_code="h")},
        "Apache-2.0"), "official misaki G2P, MLX bf16"),
    "mms-tts": (lambda: Mms(), "Meta MMS-TTS VITS eng + hin"),
    "piper_lessac-high": (lambda: Piper("en_US-lessac-high", "hi_IN-priyamvada-medium"), "Piper high + Hindi F"),
    "piper_hi-pratham": (lambda: Piper("en_US-lessac-medium", "hi_IN-pratham-medium"), "Piper medium + Hindi M"),
    "supertonic3_F1": (lambda: Supertonic("F1", "F1"), "flow-matching ONNX, 8 steps"),
    "supertonic3_F2": (lambda: Supertonic("F2", "F2"), "flow-matching ONNX, 8 steps"),
    "supertonic3_M1": (lambda: Supertonic("M1", "M1"), "flow-matching ONNX, 8 steps"),
    "kitten-mini-mlx": (lambda: Mlx("mlx-community/kitten-tts-mini-0.8", {"en": dict(voice="expr-voice-2-f")},
                                     "Apache-2.0"), "KittenTTS mini 0.8 (80M)"),
    "soprano-mlx": (lambda: Mlx("mlx-community/Soprano-1.1-80M-bf16", {"en": dict()}, "Apache-2.0"), "80M English"),
    "pocket-tts-mlx": (lambda: Mlx("mlx-community/pocket-tts", {"en": dict(voice="alba")}, "CC-BY-4.0"),
                       "Kyutai Pocket TTS 100M"),
    "qwen3-tts-0.6b-mlx": (lambda: Mlx("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit", {
        "en": dict(voice="serena", lang_code="english")}, "Apache-2.0"), "Qwen3-TTS 0.6B CustomVoice 8-bit"),
    "chatterbox-mlx": (lambda: Mlx("mlx-community/chatterbox-fp16", {
        "en": dict(), "hi": dict(lang_code="hi")}, "MIT"), "Resemble Chatterbox multilingual"),
    "orpheus-3b-4bit-mlx": (lambda: Mlx("mlx-community/orpheus-3b-0.1-ft-4bit", {"en": dict(voice="tara")},
                                        "Apache-2.0 (Llama-3.2 base: Llama licence)"), "Orpheus 3B 4-bit"),
    "omnivoice-mlx": (lambda: Mlx("mlx-community/OmniVoice-bf16", {
        "en": dict(instruct="female, young adult, moderate pitch, american accent", lang_code="en"),
        "hi": dict(instruct="female, middle-aged, moderate pitch", lang_code="hi")},
        "CC-BY-NC (weights), Apache-2.0 (code)"), "OmniVoice 0.6B bf16, voice design (no reference audio)"),
    "chatterbox-4bit-mlx": (lambda: Mlx("mlx-community/chatterbox-4bit", {
        "en": dict(), "hi": dict(lang_code="hi")}, "MIT"), "Chatterbox multilingual, 4-bit"),
    "vibevoice-rt-mlx": (lambda: Mlx("mlx-community/VibeVoice-Realtime-0.5B-fp16", {"en": dict(voice="en-Emma_woman")},
                                      "MIT"), "VibeVoice Realtime 0.5B"),
}


# --- run ------------------------------------------------------------------

def run(name: str, out: Path, runs: int) -> dict:
    factory, note = CANDIDATES[name]
    dest = out / name
    dest.mkdir(parents=True, exist_ok=True)
    f0, _ = footprint_mb()
    t = time.perf_counter()
    eng = factory()
    load_s = time.perf_counter() - t
    f_load, _ = footprint_mb()

    langs = ["en"] + (["hi"] if eng.hindi else [])
    first = {}
    for lang in langs:  # first call per language pays lazy init (G2P, graph warm-up)
        t = time.perf_counter()
        eng.synth(TTFA_EN if lang == "en" else TTFA_HI, lang)
        first[lang] = time.perf_counter() - t

    ttfa = {}
    for lang in langs:
        text = TTFA_EN if lang == "en" else TTFA_HI
        ts = []
        for _ in range(runs):
            t = time.perf_counter()
            a = eng.synth(text, lang)
            ts.append(time.perf_counter() - t)
        ttfa[lang] = {"median_s": statistics.median(ts), "min_s": min(ts), "max_s": max(ts),
                      "audio_s": len(a) / eng.sr}

    per_line, synth_s, audio_s = {}, {"en": 0.0, "hi": 0.0}, {"en": 0.0, "hi": 0.0}
    c0, w0 = cpu_s(), time.perf_counter()
    for key, lang, text in LINES:
        if lang not in langs:
            continue
        pieces, first_s = [], None
        t = time.perf_counter()
        for s in sentences(text):  # sentence by sentence, as Veronica feeds it
            ts = time.perf_counter()
            pieces.append(eng.synth(s, lang))
            first_s = first_s if first_s is not None else time.perf_counter() - ts
        el = time.perf_counter() - t
        audio = np.concatenate(pieces)
        sf.write(dest / f"{key}.wav", audio, eng.sr, subtype="PCM_16")
        dur = len(audio) / eng.sr
        synth_s[lang] += el
        audio_s[lang] += dur
        per_line[key] = {"lang": lang, "text": text, "synth_s": round(el, 3), "audio_s": round(dur, 3),
                         "first_sentence_s": round(first_s, 3)}
    cpu_cores = (cpu_s() - c0) / (time.perf_counter() - w0)
    _, peak = footprint_mb()

    res = {
        "candidate": name, "note": note, "licence": eng.licence, "hindi": eng.hindi, "sample_rate": eng.sr,
        "load_s": round(load_s, 3), "first_call_s": {k: round(v, 3) for k, v in first.items()},
        "ttfa": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in ttfa.items()},
        "rtf": {k: round(synth_s[k] / audio_s[k], 4) for k in langs},
        "cpu_cores": round(cpu_cores, 2),
        "footprint_mb": {"before_load": round(f0), "after_load": round(f_load), "peak": round(peak)},
        "disk_mb": round(sum(p.stat().st_size for p in set(eng.files)) / 1e6, 1),
        "lines": per_line,
    }
    try:
        import mlx.core as mx
        res["mlx_peak_mb"] = round(mx.get_peak_memory() / 2**20)
    except Exception:
        pass
    (dest / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("candidate", nargs="?")
    p.add_argument("--out", type=Path, default=ROOT / "samples")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list or not a.candidate:
        for k, (_, note) in CANDIDATES.items():
            print(f"{k:34} {note}")
        return
    r = run(a.candidate, a.out, a.runs)
    print(json.dumps({k: v for k, v in r.items() if k != "lines"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
