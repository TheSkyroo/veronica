"""Objective quality proxies for the WAVs scripts/tts_bench.py rendered.

    python scripts/tts_score.py asr   [samples/<candidate> ...]   # project venv (faster-whisper)
    python scripts/tts_score.py mos   [samples/<candidate> ...]   # a venv with torch + librosa

Nobody listened to these samples when this was written, so "quality" rests on
proxies that can be computed:

  asr  intelligibility: faster-whisper large-v3-turbo transcribes each line;
       word error rate for English, character error rate for Hindi. Low is
       necessary, not sufficient: a flat robotic voice can score 0.
  mos  UTMOS22-strong (SpeechMOS) predicted naturalness, 1-5, trained on
       English; on Hindi it is a weak signal. Also the median / 10th-90th
       percentile F0 of the voiced frames (pyin) and speaking rate - the
       objective side of "sounds like a child" (a high, narrow pitch) and of
       "too fast / too slow".

Results merge into <candidate>/score.json; with no folders given, every
folder under samples/ that has a metrics.json is scored."""
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def folders(args: list[str]) -> list[Path]:
    if args:
        return [Path(a) for a in args]
    return sorted(p.parent for p in (ROOT / "samples").glob("*/metrics.json"))


def merge(folder: Path, part: dict) -> None:
    f = folder / "score.json"
    cur = json.loads(f.read_text()) if f.exists() else {}
    for k, v in part.items():
        cur.setdefault(k, {}).update(v) if isinstance(v, dict) else cur.__setitem__(k, v)
    f.write_text(json.dumps(cur, indent=2, ensure_ascii=False))


def norm(text: str, lang: str) -> str:
    t = unicodedata.normalize("NFC", text.lower())
    t = t.replace("$", " dollars ").replace("%", " percent ")
    t = re.sub(r"(?<=\d)[,:](?=\d)", "", t)  # 1,249 / 6:40 -> 1249 / 640
    t = re.sub(r"[^\w\sऀ-ॿ]", " ", t)
    t = t.replace("।", " ")
    return " ".join(t.split())


def edit_distance(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def asr(dirs: list[Path]) -> None:
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel
    from scipy.signal import resample_poly
    m = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    for d in dirs:
        meta = json.loads((d / "metrics.json").read_text())
        out = {}
        for key, line in meta["lines"].items():
            y, sr = sf.read(d / f"{key}.wav", dtype="float32")
            y = resample_poly(y, 16000, sr).astype(np.float32)  # decoded here: faster-whisper's own
            segs, _ = m.transcribe(y, language=line["lang"], beam_size=5,  # decoder trips on some av builds
                                   condition_on_previous_text=False)
            hyp = " ".join(s.text.strip() for s in segs)
            ref_n, hyp_n = norm(line["text"], line["lang"]), norm(hyp, line["lang"])
            if line["lang"] == "hi":  # character error rate, spaces ignored
                r, h = list(ref_n.replace(" ", "")), list(hyp_n.replace(" ", ""))
            else:
                r, h = ref_n.split(), hyp_n.split()
            out[key] = {"heard": hyp, "err": round(edit_distance(r, h) / max(1, len(r)), 3),
                        "metric": "cer" if line["lang"] == "hi" else "wer"}
        merge(d, {"asr": out})
        print(d.name, {k: v["err"] for k, v in out.items()})


def mos(dirs: list[Path]) -> None:
    import librosa
    import numpy as np
    import torch
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    for d in dirs:
        meta = json.loads((d / "metrics.json").read_text())
        out = {}
        for key, line in meta["lines"].items():
            y, _ = librosa.load(d / f"{key}.wav", sr=16000)
            with torch.no_grad():
                score = float(predictor(torch.from_numpy(y).unsqueeze(0), 16000))
            y22, _ = librosa.load(d / f"{key}.wav", sr=22050)
            f0, voiced, _ = librosa.pyin(y22, fmin=60, fmax=500, sr=22050)
            f0 = f0[voiced & ~np.isnan(f0)]
            dur = len(y) / 16000
            words = len(line["text"].split())
            out[key] = {"utmos": round(score, 2),
                        "f0_median": round(float(np.median(f0)), 1) if len(f0) else None,
                        "f0_p10_p90": [round(float(np.percentile(f0, q)), 1) for q in (10, 90)] if len(f0) else None,
                        "words_per_s": round(words / dur, 2)}
        merge(d, {"prosody": out})
        print(d.name, {k: (v["utmos"], v["f0_median"]) for k, v in out.items()})


if __name__ == "__main__":
    {"asr": asr, "mos": mos}[sys.argv[1]](folders(sys.argv[2:]))
