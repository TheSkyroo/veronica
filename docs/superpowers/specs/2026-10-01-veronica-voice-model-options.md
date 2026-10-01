# Veronica — voice model options (spike)

Date: 2026-10-01. Branch `voice-model-spike` (worktree `veronica-l4`), off `master` at `0aabc90`. An evaluation
and a recommendation only: no runtime code changed. Not merged.

## The ask

> "now we improve the voice model" — the user, early in the project, never scoped further.

Feedback from the logs: the Hindi voice sounded like "some innocent voice" (wrong character or age), and a Hindi
voice was sometimes used on English sentences (that routing bug is already fixed).

Today: Kokoro-82M v1.0 through `kokoro-onnx` 0.6.1 on CPU (`veronica/speech/tts.py` `Synthesizer`). The English
voice is `af_bella` and the Hindi voice is `hf_alpha` (`veronica/speech/voices.py`). Synthesis is one whole
sentence at a time, and sentence N+1 is synthesised while N plays (`orchestrator.handle_text`). The latency that
matters is therefore **the time to synthesise the first sentence**. A model's own chunk streaming would only help
if the orchestrator changed.

## Method

- Machine: Apple M5 MacBook Air (fanless), 10 cores, 24 GB. Veronica.app was running in the background.
- `scripts/tts_bench.py <candidate>` (one venv per engine family) does the following:
  - loads the model;
  - warms it up once per language;
  - takes the median of 5 syntheses of a typical first sentence: "The capital of Australia is Canberra, not Sydney
    as many people assume." (≈4.4 s of audio), plus "आपकी मीटिंग शाम चार बजे है।" for Hindi;
  - renders the six fixed lines sentence by sentence, the way Veronica does;
  - records the real-time factor, CPU cores busy, peak physical footprint (`proc_pid_rusage`, which includes Metal
    buffers) and weights on disk.
- `scripts/tts_score.py` holds the objective quality proxies, because **nobody has listened to these samples**:
  - `asr`: faster-whisper large-v3-turbo transcribes each line. Output is WER for English and CER for Hindi.
  - `mos`: UTMOS22-strong predicted naturalness (1–5). It was trained on English, so it is a weak signal on Hindi.
  - The same `mos` step records pyin F0 (median pitch) and speaking rate.
  - "Quality" below rests on these proxies plus published evaluations. It is not my ear. Please listen.
- **Thermal and contention caveat.** On a fanless Air the numbers drift. The same baseline went from 0.92 s to
  1.45 s TTFA within ~10 minutes of back-to-back runs. A later "clean" pass (60 s cooldowns) ran into a load
  average of ~21 from other processes and read 3.9 s for the baseline, so it was aborted.
  - The numbers below are from the first, quieter run of each candidate.
  - Compare candidates within a row group, and read anything under ~1.5× apart as a tie.
  - The `metrics.json` in some sample folders is from the aborted pass. Prefer this table.

## Samples (listen to these)

Each candidate folder in `samples/<candidate>/` (gitignored, in the `veronica-l4` worktree) holds six WAVs plus
`metrics.json` and, where scored, `score.json`. The six lines are:

| File | Text |
|---|---|
| `01_greeting.wav` | "Good morning! What can I do for you today?" |
| `02_answer.wav` | Two sentences: Canberra is the capital of Australia (it is not Sydney), and why it was chosen in 1908 |
| `03_confirm.wav` | "Copy to clipboard? Say yes, no, or always." |
| `04_numbers.wav` | A flight at 6:40 AM on October 14th from gate 23B, two tickets for $1,249.50, a 45-minute drive |
| `05_hindi_greeting.wav` | "नमस्ते! आज मौसम बहुत सुहावना है, क्या आप बाहर घूमने चलना चाहेंगे?" |
| `06_hindi_answer.wav` | "आपकी मीटिंग शाम चार बजे है। क्या मैं आपको पंद्रह मिनट पहले याद दिला दूँ?" |

English-only candidates have no `05`/`06` files.

Start with these:

- `samples/kokoro-onnx_af_bella/`: the baseline (today's voice pair).
- `samples/kokoro-onnx_af_heart/`: `af_heart` for English, `hf_beta` for Hindi. This is the recommended zero-cost
  swap.
- `samples/kokoro-mlx_af_heart/`: the same voices through the official Kokoro pipeline on MLX (misaki G2P).
- `samples/kokoro-onnx_hindi_voices/`, `kokoro-onnx_hm_omega/`, `kokoro-onnx_hm_psi/`,
  `kokoro-onnx_hf_blend_beta-alpha/`: the other Kokoro Hindi voices and a Hindi blend.
- `samples/supertonic3_F1/`, `samples/chatterbox-4bit-mlx/`, `samples/piper_hi-pratham/`,
  `samples/piper_lessac-high/`, `samples/mms-tts/`, `samples/omnivoice-mlx/`: alternatives that speak Hindi.

## Comparison (measured)

Column key:

- **TTFA**: median seconds to synthesise the first sentence, English / Hindi. Lower is better. Veronica starts
  speaking after this.
- **RTF**: synthesis time ÷ audio time, English / Hindi. Lower is better.
- **CPU**: average cores busy while synthesising.
- **Peak MB**: physical footprint. The bracketed number is MLX/Metal peak.
- **WER / CER**: from `tts_score.py asr`. The numbers line is listed separately. "–" means not scored.
- **UTMOS**: English mean / Hindi mean. **F0 hi**: median pitch of the Hindi lines.

### Kokoro-82M (the current model): voices, quantisation, runtime

| Candidate | TTFA en / hi | RTF en / hi | CPU | Peak MB | Disk MB | WER en (num) | CER hi | UTMOS en / hi | F0 hi |
|---|---|---|---|---|---|---|---|---|---|
| **baseline** kokoro-onnx fp32, af_bella + hf_alpha | 0.92 / 0.56 | 0.23 / 0.25 | 4.8 | 688 | 354 | 0.00 (0.15) | 0.04 | 4.17 / 4.15 | 214 Hz |
| kokoro-onnx af_sarah (config.py default) | 1.02 / 0.59 | 0.26 / 0.26 | 4.8 | 678 | 354 | – | – | – | – |
| kokoro-onnx **af_heart** + **hf_beta** | 0.98 / 0.58 | 0.25 / 0.28 | 4.8 | 674 | 354 | 0.00 (0.15) | 0.02 | **4.51** / 4.17 | **186 Hz** |
| kokoro-onnx blend heart 0.6 + bella 0.4 | 1.04 / 0.61 | 0.26 / 0.28 | 4.8 | 672 | 354 | – | – | – | – |
| kokoro-onnx blend bella 0.7 + nicole 0.3 | 1.22 / – | 0.26 | 4.8 | 685 | 354 | – | – | – | – |
| kokoro-onnx **fp16** weights | 1.10 / 0.63 | 0.26 / 0.27 | 4.8 | 542 | 206 | – | – | – | – |
| kokoro-onnx **int8** weights | 2.25 / 1.37 | 0.54 / 0.62 | 2.5 | 422 | 121 | – | – | – | – |
| kokoro-onnx Hindi hf_beta (\*) | 1.45 / 0.77 | 0.34 / 0.38 | 4.8 | 682 | 354 | – | 0.02 | 4.17† / 4.17 | 186 Hz |
| kokoro-onnx Hindi hm_omega, male (\*) | 1.45 / 0.86 | 0.34 / 0.36 | 4.8 | 683 | 354 | – | – | 4.17† / 4.00 | 123 Hz |
| kokoro-onnx Hindi hm_psi, male (\*) | 1.49 / 1.00 | 0.35 / 0.37 | 4.8 | 681 | 354 | – | – | – | – |
| kokoro-onnx Hindi blend beta 0.6 + alpha 0.4 (\*) | 1.51 / 0.84 | 0.35 / 0.38 | 4.8 | 682 | 354 | – | – | – | – |
| **kokoro-mlx** (mlx-audio, bf16, misaki G2P), af_heart + hf_beta | **0.26 / 0.16** | **0.056 / 0.055** | **0.5** | 3696 [1807] | 389 | 0.00 (0.19‡) | 0.07‡ | **4.52** / 4.25 | 188 Hz |
| onnxruntime **CoreML** execution provider | 0.99 (NeuralNetwork); MLProgram fails | – | – | – | – | – | – | – | – |

(\*) These ran last in a back-to-back batch, and the baseline re-run at the same point was also ~1.5× slower.
Their real cost equals the baseline's: it is the same model, and the voice is only a style vector.
† The English half of these rows is the same `af_bella` voice as the baseline.
‡ This difference is in how the ASR writes the output, not in the speech. Whisper transcribed MLX Kokoro's "14th" as
"14" (probably said "fourteen"), and Whisper wrote "15" for "पंद्रह", which the simple CER counts as wrong.

### Other engines

| Candidate | TTFA en / hi | RTF en / hi | CPU | Peak MB | Disk MB | Licence | Hindi | Proxies / note |
|---|---|---|---|---|---|---|---|---|
| Piper lessac-high + hi priyamvada | 0.68 / 0.05 | 0.17 / 0.02 | 4.9 | 546 | 177 | GPL-3.0 engine (piper1-gpl). lessac: Blizzard research licence. Hindi voices: CC-BY-NC-SA-4.0 | yes | VITS-era. Published listening tests place Piper well below Kokoro |
| Piper lessac-medium + hi pratham (male) | **0.09 / 0.05** | 0.02 / 0.02 | 4.7 | 450 | 127 | same | yes | UTMOS en 4.35. UTMOS hi 3.54, the lowest Hindi score measured. F0 hi 101 Hz |
| Supertonic-3 F1 (ONNX flow matching, 8 steps) | 1.06 / 0.64 | 0.22 / 0.25 | 5.1 | 830 | 401 | OpenRAIL-M weights, MIT code | yes (31 langs) | WER 0. Misread "$1,249.50" as "1.49,950". UTMOS 4.39 / 4.09. F0 hi 186 Hz |
| Meta MMS-TTS eng + hin (VITS, torch CPU) | 0.45 / 0.25 | 0.09 / 0.09 | 2.1 | 1235 | 291 | CC-BY-NC-4.0 | yes | Single-speaker research voices. Unscored |
| KittenTTS mini 0.8 (MLX) | 0.38 / – | 0.06 | 0.7 | 2969 [1599] | 292 | Apache-2.0 | no | Tiny. Unscored |
| Soprano 1.1 80M (MLX) | 0.16 / – | 0.05 | 0.4 | 1433 [697] | 284 | Apache-2.0 | no | Fastest English measured. Unscored |
| Kyutai Pocket TTS 100M (MLX) | 0.53 / – | 0.12 | 0.4 | 2023 [788] | 240 | CC-BY-4.0 | no | Unscored |
| Chatterbox multilingual 4-bit (MLX) | 1.59 / 1.01 | 0.44 / 0.47 | 0.2 | 3812 [1984] | 612 | MIT | yes (23 langs) | Garbled the money and a Hindi word ("सांग"). UTMOS 3.95 / 3.35. F0 hi 158 Hz (lower, adult) |
| Chatterbox multilingual fp16 (MLX) | 3.12 / 1.81 | 0.82 / 0.85 | 0.2 | 6036 [4089] | 2577 | MIT | yes | Too slow for turn-taking. Unscored |
| OmniVoice 0.6B bf16, voice design (MLX) | 1.13 / 0.78 | 0.28 / 0.34 | 0.2 | 4146 [2707] | 1640 | **CC-BY-NC** weights | yes (600+) | Prompted "female, middle-aged" for Hindi. Its README says voice design is trained on zh/en only. Unscored |
| Qwen3-TTS 0.6B CustomVoice 8-bit (MLX) | 1.87 / – | 0.32 | 0.5 | 9152 [5285] | 1974 | Apache-2.0 | no (10 langs, no hi) | Memory-hungry. Unscored |
| VibeVoice-Realtime 0.5B fp16 (MLX) | 4.56 / – | 0.89 | 0.2 | 8295 [4177] | 2135 | MIT | no | Too slow. Unscored |
| Orpheus 3B 4-bit (MLX) | 8.70 / – | 1.41 | 0.2 | 6021 [4640] | 1885 | Apache-2.0 plus Llama 3.2 licence | no | Slower than real time here. Unscored |

The heavy MLX rows (Chatterbox, OmniVoice, Qwen3, VibeVoice, Orpheus) were measured while the ASR scorer was also
running. Their absolute latency is pessimistic by perhaps 1.2–1.5×, but none of them would come under the 0.92 s
baseline even so. Load time is not in the table. It was 0.3–0.8 s from disk for Kokoro, Piper and Supertonic,
which matters only at app start. The MLX Kokoro first call took 9.8 s (spaCy and misaki initialise). That cost
must be paid at startup, not on the first reply.

## What the numbers say

1. **Latency.** No higher-quality model beats today's first-sentence latency on this machine.
   - Every LLM- or diffusion-style model (Chatterbox, OmniVoice, Qwen3-TTS, VibeVoice, Orpheus) is 1.2–9× slower
     than the baseline, and each uses several GB of memory.
   - The only things meaningfully faster are the same Kokoro model on the GPU via MLX (3.5× faster) and older or
     tiny models (Piper medium, Soprano, Kitten), which give up quality.
2. **English quality.** `af_heart` is Kokoro's own top-graded voice: grade **A** in hexgrad's `VOICES.md`, against
   **A-** for `af_bella` and **C+** for `af_sarah`.
   - It also scores highest on UTMOS here (4.51 against 4.17), at identical cost and with zero ASR errors.
   - This is the cheapest real win.
3. **Pronunciation.** The official Kokoro pipeline (misaki G2P, used by the MLX port) and kokoro-onnx (espeak G2P)
   produced the same English transcripts. The UTMOS difference between them is negligible (4.52 against 4.51).
   - The MLX gain is speed and CPU (0.5 cores against 4.8, which leaves the CPU to Whisper and the wake word), not
     voice quality.
4. **Hindi character.** `hf_alpha` has the highest and widest pitch of the Kokoro Hindi voices: median 214 Hz,
   10th–90th percentile 156–283 Hz. That is consistent with the "innocent" (young-sounding) complaint.
   - `hf_beta` is lower: 186 Hz, which is typical of an adult female.
   - `hf_beta` also has the lowest Hindi CER here (0.02 against 0.05 for `hf_alpha`).
   - Kokoro grades all four of its Hindi voices **C**, because they were trained on minutes of data, not hours. No
     Kokoro Hindi voice will sound as natural as the English ones.
5. **Quantisation.** int8 Kokoro is 2.4× *slower* on Apple Silicon (onnxruntime has no fast int8 path here). fp16
   saves 150 MB of disk and ~150 MB of RAM at about the same speed. The onnxruntime CoreML EP gives nothing:
   MLProgram crashes on the iSTFT, and NeuralNetwork falls back to the CPU at 0.99 s.

## Recommendation (ranked)

1. **Ship now: switch the English voice to `af_heart` and the Hindi voice to `hf_beta`. Stay on kokoro-onnx.**
   - Quality: Kokoro's best-graded English voice, plus a lower, more adult Hindi voice that is the more
     intelligible of the two.
   - Latency, memory and size are unchanged.
   - Risk is near zero.
   - Smallest change:
     - `veronica/speech/voices.py`: add `"heart": "af_heart"` to `VOICES` and set
       `DEFAULT_HINDI_VOICE = "hf_beta"`.
     - `veronica/config.py`: set `kokoro_voice = "af_heart"`.
     - `veronica/speech/tts.py`: change the `hindi_voice` default to `"hf_beta"`.
     - The user's saved voice in settings overrides the default. They would switch with "switch to Heart" (once
       the voice is in the table) or by clearing `tts_voice`. Hindi likewise: "switch to Beta".
   - Streaming is unchanged. No new dependencies. The voices are already in `voices-v1.0.bin`.
   - If they prefer a blend, `kokoro-onnx` accepts a style vector in place of a voice name
     (`0.6*af_heart + 0.4*af_bella` is in the samples). That would need a voice-spec parser in `Synthesizer`
     (~10 lines, as in `scripts/tts_bench.py` `_voice_mix`).
2. **Next, if the first-sentence delay bothers them: run the same Kokoro model on MLX.**
   - TTFA falls from 0.92 s to 0.26 s (English) and from 0.56 s to 0.16 s (Hindi).
   - CPU falls from ~4.8 cores to ~0.5 during speech. That matters, because Whisper, the wake word and the speaker
     check share the CPU.
   - Same voices, same 24 kHz output, Apache-2.0.
   - Costs:
     - Peak footprint is ~3.7 GB against 0.7 GB (MLX buffer cache plus spaCy and torch, which misaki imports).
       `mx.set_cache_limit` should cut this but was not tested.
     - Heavy new dependencies: `mlx`, `mlx-audio`, `misaki[en]`, spaCy plus `en_core_web_sm`, `transformers`.
     - A ~10 s first-call warm-up that must happen at app start.
     - More bundling work in `scripts/build_app.py`: Metal libraries, the spaCy model.
   - Change: a second `Synthesizer` implementation (`veronica/speech/tts.py`) that calls
     `mlx_audio.tts.utils.load_model("mlx-community/Kokoro-82M-bf16")` and `model.generate(text, voice=…,
     lang_code="a"|"h")`, warms both languages up in `__init__`, and is chosen by a setting.
     - The `synth()` / `asynth()` contract and the sentence pipeline stay as they are, so streaming still works.
     - Model download goes in `scripts/download_models.py`.
   - Keep kokoro-onnx as the fallback.
3. **Not now: a "bigger" neural voice (Chatterbox, OmniVoice, Qwen3-TTS, VibeVoice, Orpheus).**
   - Published blind tests rate them as more expressive. Resemble reports Chatterbox preferred over ElevenLabs; the
     Orpheus and Sesame-class models are known for prosody.
   - On this Air, though, each costs 1.1–8.7 s before the first word, 2–9 GB of memory, and 0.6–2.6 GB of disk.
   - Where measured, they were also *less* accurate on numbers and Hindi than Kokoro.
   - OmniVoice's weights are non-commercial.
   - They would fit only with chunk-level streaming inside a sentence (an orchestrator change), and still at a large
     memory cost. Revisit on a machine with more headroom, or if the Kokoro voices are judged too flat after
     listening.
4. **Not recommended: Piper, MMS-TTS, Kitten, Soprano, Pocket TTS.**
   - They are fast, but they are older or tiny architectures.
   - Piper's engine is now GPL-3.0, and its Hindi voices are CC-BY-NC-SA. MMS is CC-BY-NC.
   - Soprano at 0.16 s is worth a listen as an English speed outlier, but it has no Hindi and was not quality-scored.
   - Supertonic-3 is the most interesting of the rest: one ONNX model for both languages, UTMOS close to Kokoro,
     OpenRAIL-M licence. But it was no faster on CPU and misread the money line.

## Best Hindi voice

- **Recommended: Kokoro `hf_beta`.** Treat this as provisional until the user listens. It costs nothing, and it is
  lower-pitched than `hf_alpha` (186 Hz against 214 Hz median, a narrower top range), so it reads as adult, not
  girlish. It also has the best Hindi intelligibility measured (CER 0.02) and UTMOS 4.17.
- **If a male voice is acceptable: `hm_omega`.** Median 123 Hz, UTMOS 4.00.
- **If no Kokoro Hindi voice satisfies after listening** (all are grade C, trained on minutes of data), the next step
  up that runs here is one of:
  - **Supertonic-3 F1**: adult pitch (186 Hz), UTMOS 4.09 on Hindi, 0.64 s TTFA, one 400 MB ONNX model, OpenRAIL-M.
  - **Chatterbox multilingual 4-bit**: an adult, lower voice at 158 Hz, MIT, but ~1 s TTFA and it garbled one Hindi
    word.
  - Either would be a Hindi-only second engine behind the existing `lang == "hi"` routing in `Synthesizer.synth`.
- The purpose-built Indic models could not be run (see below). They are the ones to try with a Hugging Face login.

## What could not be evaluated, and why

- **AI4Bharat IndicF5** (MIT, 1.4 GB, Hindi): the repo is gated. It needs a Hugging Face login and acceptance of
  the access terms. No token was available, and unofficial re-uploads were deliberately not used.
- **AI4Bharat Indic Parler-TTS** (Apache-2.0): its single weights file is 3.75 GB, which is over the ~3 GB per-model
  limit for this spike.
- **VoxCPM2** (5 GB), **Dia 1.6B** (3.2 GB even at 4-bit, and a dialogue model), **Spark-TTS** (2.9 GB,
  CC-BY-NC-SA): skipped on size or licence.
- **F5-TTS / StyleTTS2 / XTTS-v2**:
  - F5-class quality is represented by OmniVoice and Chatterbox, which ran.
  - StyleTTS2 is Kokoro's own architecture.
  - XTTS-v2 is under the Coqui Public Model License and is effectively unmaintained.
  - None was installed separately.
- **Sesame CSM-1B**: not run, because of time (it is English-only and conversational; expect Orpheus-like cost).
- **Hindi Orpheus fine-tunes**: community uploads only, 3B, with no clear licence. Not run.
- **OmniVoice 8-bit**: the MLX checkpoint failed to load (quantised-weight mismatch), so bf16 was used.
- **onnxruntime CoreML for Kokoro**: MLProgram fails on a zero-length iSTFT tensor. NeuralNetwork runs, with no
  gain.
- **Newer Kokoro**: none exists. The latest is v1.0 (Jan 2025). v1.1-zh (Mar 2025) is a Chinese-focused variant
  that adds no Hindi or English voices that matter here.
- **Partial scoring**: the ASR and UTMOS proxies cover only the rows marked above. Every "Unscored" row has samples
  but no proxy numbers. Run `scripts/tts_score.py asr|mos samples/<candidate>` to fill them in.
- **A controlled timing pass**: aborted because of outside CPU load (see Method). The latency ratios are
  consistent across runs; the absolute numbers drift.

## Reproduce

```sh
# kokoro-onnx candidates use the project venv; others need their own (see tts_bench.py docstring)
uv run python scripts/tts_bench.py --list
uv run python scripts/tts_bench.py kokoro-onnx_af_heart            # -> samples/kokoro-onnx_af_heart/
HF_HOME=.spike/hf uv run python scripts/tts_score.py asr samples/kokoro-onnx_af_heart
```

Model files go in `.spike/models/kokoro/` (the `model-files-v1.0` release of thewh1teagle/kokoro-onnx) and
`.spike/models/piper/` (rhasspy/piper-voices). The MLX, Supertonic and MMS models come from the Hugging Face cache
at `HF_HOME=.spike/hf`. The scratch area is ~20 GB and safe to delete: `rm -rf .spike samples`.
