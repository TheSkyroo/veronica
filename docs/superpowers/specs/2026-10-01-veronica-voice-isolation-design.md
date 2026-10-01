# Veronica — voice isolation: less background noise, only my voice

Date: 2026-10-01. Branch `voice-isolation` (worktree `veronica-l3`), off `master` at `fdbdd41`. Not merged.

## The ask

> "I want a feature to reduce the background noise and also it should only listen to my voice not others."
> — the user, voice turn, 26 Sep

Evidence from the logs: room noise and other people's voices get transcribed and acted on. During a confirm,
noise came through as "Ich küsse, küsse, küsse." and was classified as the answer ("other" = a redirect), which
ended the task.

Two independent parts:

- **A. Noise** — noise should not open or stretch a capture, and should not reach the brain as words.
- **B. Only my voice** — once the user has enrolled, an utterance in another voice is ignored: never a request,
  never a follow-up, never a confirm answer.

## Summary of decisions

| | Default | Why |
|---|---|---|
| `noise_suppression` (GTCRN + AGC + 60 Hz high-pass) on what the **VAD and level floor** see | **on** | Clatter and music stopped opening captures (100% → 0%) with no loss of near or across-the-room speech; ~2.7% of one core, only while a capture is open |
| STT input | **raw audio** | Every suppressor we tried made whisper worse (pink 5 dB: 3.0% → 16.7% WER; TV 5 dB: 10.6% → 40.2%) |
| Wake path (`WhisperWake`) | **untouched** (raw) | Suppressed audio cost wake hits (75% → 33% at pink 5 dB); a level gate on suppressed audio missed quiet "Veronica"s (12/12 → 9/12) |
| `vad_min_rms` (level floor on the suppressed frame; off whenever suppression isn't running) | **0.001** | Keeps a quiet background voice (0.002 rms) from opening a capture (100% → 0%) and still misses ≤5% of far speech (0.0015 rms); 0.002 missed 88% of it |
| `vad_aggressiveness` | **2 (unchanged)** | 3 missed 40% of across-the-room speech over clatter even with suppression |
| `speaker_verification` | **on, inert until enrolled** | Nothing happens without a profile; enrolling is the opt-in |
| `speaker_threshold` | **0.35**, scaled down for short audio (never under 80% for a confirm answer) | Enrolled voice accepted on 100% of commands (clean and noisy), other voices 4% (synthetic voices, which are closer than real people) |
| `speaker_verification_wake` | **off** | The wake word is as short as a "yes", where 8% (quiet) to ~55% (noisy) of the user's own takes scored under the bar; a missed wake is worse than a stray one |
| Speaker model | 3D-Speaker **CAM++** (VoxCeleb), ONNX, Apache-2.0, 29 MB | Best separation in stationary noise of the three tried, 11 ms per 3 s utterance, clean licence |

## Part A — noise

### Candidates

All offline on Apple Silicon, measured on the same clips (below).

| | GTCRN | RNNoise (`pyrnnoise`) | WebRTC NS (`webrtc-noise-gain`) | Spectral gate (`noisereduce`) |
|---|---|---|---|---|
| Licence | MIT (model), sherpa-onnx export | BSD-3 | MIT (wraps BSD WebRTC) | MIT |
| Dependency weight | a 0.5 MB `.onnx` on the `onnxruntime` we already ship (Kokoro) | ctypes lib **plus PyAV/FFmpeg** for its resampler | C extension | pure Python + scipy |
| Python 3.13 wheel (our venv) | n/a | yes | **no — sdist fails to build on 3.13** | yes |
| Streaming at 16 kHz | yes, native, 16 ms hop, recurrent state | 48 kHz inside, resampled | 10 ms | no — whole clip, needs a noise sample |
| CPU | 2.7% of one M5 core (streamed, 30 ms chunks) | ~2× GTCRN in the harness | — | ~1% |
| Clean speech word error (whisper small.en) | 4.5% → 6.1% | 4.5% → 30%* | — | 4.5% → 48% |

\* Probably partly our harness around its resampling graph; not chased, because (a) no suppressor helped whisper
(below) and (b) it drags FFmpeg in. WebRTC NS could not be installed at all.

**GTCRN** won on weight and streaming. Two fixes were needed before it behaved at real mic levels:

- **Level.** GTCRN's suppression depends on input level: pink noise at a real room level (0.001–0.01 rms) lost only
  ~10 dB, the same noise at 0.03–0.1 lost ~30 dB. A per-frame AGC (target 0.05 rms, up to ×32, fast attack /
  ~1.6 s release) scales each STFT frame in and back out: now 27–31 dB of suppression at every level from 0.001 to
  0.1, speech correlation 0.997.
- **Rumble.** Pink noise's sub-audio content went straight through (desk thumps, DC drift in real mics) and was most
  of the residual at room level. A 2nd-order 60 Hz Butterworth high-pass in front, as speech front ends do.

### Where it goes: detection yes, recognition no

Eval set (`scripts/eval_voice_isolation.py` reproduces it): 20 commands in two Kokoro voices, 0.35 s synthetic
room reverb, speech at 0.01 rms (a near-field voice), mixed with pink noise, 6-voice babble or a TV-like bed (one
presenter + a music bed) at 10 / 5 / 0 dB; faster-whisper `small.en`, beam 1, as `Transcriber` runs it.

Word error, raw audio vs the shipped suppressor:

| | clean | pink 10 | pink 5 | pink 0 | babble 10 | babble 5 | TV 10 | TV 5 |
|---|---|---|---|---|---|---|---|---|
| raw | 4.5% | 4.5% | 3.0% | 7.6% | 6.1% | 21.2% | 5.3% | 10.6% |
| suppressed | 6.1% | 4.5% | 16.7% | 32.6% | 12.9% | 62.9% | 28.0% | 40.2% |

Whisper is trained on noisy audio and is thrown by enhancement artefacts; this matches the literature. So the
recorder runs suppression to decide **whether** someone is speaking, and hands whisper (and the speaker check) the
**raw** capture.

Two details of that split (both from review): the suppressed frame lags the raw one by 32 ms (`denoise.LAG`), so
the onset it reveals is 1–2 raw frames late — the recorder keeps the last `ceil(LAG / frame) + 1` raw frames and
puts them in front of the capture (without that, every capture lost its first ~30 ms). And the floor only applies
to suppressed frames: with suppression off (or failed) the VAD decides alone, exactly as before. The HUD's mic
meter shows the raw level. A suppressor that raises mid-capture is logged once and switched off until restart;
the capture carries on raw.

What opens a capture (the recorder's decision, 10 × 4 s clips each; "before" = raw audio, VAD only; "now" = the
VAD and the 0.001 floor on suppressed audio):

| noise, level (rms) | before | now |
|---|---|---|
| fan / motor hum, 0.002–0.01 | 0% | 0% |
| keyboard / dish clatter, 0.002 / 0.005 / 0.01 | 100% / 100% / 100% | 0% / 0% / 0% |
| music, 0.005 / 0.01 | 100% / 100% | 0% / 0% |
| background talk (babble), 0.002 / 0.005 | 100% / 100% | 0% / 100% |
| TV, 0.002 / 0.005 | 100% / 100% | 100% / 100% |

Voices are voices: suppression keeps them, and the floor only stops faint ones. Background speech is Part B's job.

Speech that must still get in (40 commands each, over a fan or clatter 15 dB below the voice):

| voice level | missed before | missed now (floor 0) | now (**0.001**) | now (0.002) | now, VAD 3 |
|---|---|---|---|---|---|
| 0.01 (near) | 0% | 0% | 0% | 0% | 0% |
| 0.003 (across the room) | 0% | 0% | 0% | 0% | 3–40% |
| 0.0015 (far) | 0–3% | 0–3% | 0–5% | 88% | 100% |

Hence `vad_min_rms = 0.001` and `vad_aggressiveness` stays 2. The "Ich küsse" capture was room noise opening a
capture during a confirm; that class of noise no longer opens one, and with a voice profile a capture that does
open in some other voice can't be an answer (Part B).

### The wake path is left alone

`WhisperWake` transcribes a 2 s window with `tiny.en` every hop. Suppressing it hurt (12 "Veronica"/"Hey
Veronica" takes per cell, 6 voices):

- transcribing suppressed audio: hits at pink 5 dB 75% → 33%, babble 10 dB 75% → 17%;
- gating on the suppressed window's level, transcribing raw: a quiet "Veronica" (0.005 rms voice) 12/12 → 9/12.

It would also have cost ~3% of a core all day. `mic.py`'s shared reader stays a raw frame source; the only change in
the wake path is an optional speaker-check hook (Part B, off by default).

## Part B — only my voice

### Model

Three permissively licensed ONNX speaker-embedding models from the sherpa-onnx release, same features, 12 Kokoro
voices × 6 phrases (enrol on 3, test on 3):

| model | size | EER clean | babble 10 dB | pink 5 dB | 3 s, one call |
|---|---|---|---|---|---|
| WeSpeaker ResNet34 (VoxCeleb; weights CC-BY-4.0) | 25 MB | 0% | 2.5% | 0.5% | ~2× CAM++ |
| WeSpeaker CAM++ (VoxCeleb; weights CC-BY-4.0) | 29 MB | 0% | 3.0% | 2.8% | ≈ |
| **3D-Speaker CAM++ (VoxCeleb; Apache-2.0)** | 29 MB | 0% | 2.8% | 0.4% | **11 ms** idle (5 ms / 1 s, 19 ms / 6 s) |

CAM++ from 3D-Speaker: best in stationary noise, Apache-2.0 end to end. It is fetched like the other models, into
`~/.veronica/models`, pinned by SHA256 in `veronica/audio/models.py` (`357a834f…129b`; GTCRN `e77603ac…b534`), on
first enrolment or by `scripts/download_models.py`.

Features are 80-bin Kaldi fbanks (Povey window, [-1, 1] input, utterance mean subtracted), computed in numpy
(`veronica/audio/fbank.py`) rather than adding torchaudio or kaldi-native-fbank; checked against
kaldi-native-fbank to 1e-4 (hamming) / 3e-3 (povey), pinned by `tests/test_fbank.py`.

Embeddings are taken on the **raw** audio: with suppressed audio EER at pink 5 dB went 0.4% → 7.8% (3.2% even
with suppressed enrolment; measured before the suppressor got its AGC, and the recorder hands over raw audio
anyway). Only 30 ms frames within 20 dB of the utterance's loud part are embedded, so the VAD's
1.2 s of trailing silence doesn't dilute the voice.

### Threshold

Production path (`SpeakerGate`: trimming, length scaling), 12 voices; each capture padded as the recorder hands it
over (0.3 s lead-in, 1.2 s tail, a quiet room floor). Share of checks that go the wrong way at each base threshold:

| base threshold | 0.30 | **0.35** | 0.40 | 0.45 |
|---|---|---|---|---|
| my command ignored — quiet / babble 10 dB / pink 5 dB | 0 / 0 / 0% | **0 / 0 / 0%** | 0 / 5.6 / 2.8% | 0 / 14 / 5.6% |
| someone else's command accepted | 6.3% | **4.0%** | 2.5% | 2.0% |
| my short answer ("yes", "go ahead") ignored — quiet / babble / pink | 4 / 50 / 29% | **8 / 60 / 49%** | 15 / 75 / 63% | 19 / 82 / 79% |
| someone else's short answer accepted | 3.0% | **1.9%** | 1.3% | 0.8% |
| noise alone accepted — pink / TV | 0 / 0% | **0 / 0%** | 0 / 0% | 0 / 0% |

Short utterances embed less reliably (the user's own "yes" scored 0.43 on average against 0.73 for a sentence),
so the bar scales linearly from 60% of the threshold with no voiced audio to 100% at 2 s. "Voiced" counts only
30 ms frames within 20 dB of the loud part and above 0.0005 rms, and fewer than ten count as none, so more speech
can never lower the bar. A confirm answer's bar never drops under 80% of the threshold: at 0.35 that costs the
user's quiet short answers 8% → 15% ignored, and cuts other voices' short answers accepted 1.9% → 1.4% (noisy
takes are voiced long enough to be unaffected). Babble noise "accepted" (8%) is babble made of the same Kokoro
voices as the profiles.

**0.35** keeps every one of the user's commands, even in noise, and lets ~4% of other (synthetic) voices through;
real people are further apart than Kokoro's blended voices, so that is pessimistic. The cost is short answers in
a loud room, where the gate errs safe (see below); the log and the Settings window show every score for tuning.

**Honesty:** TTS voices are a sanity check, not a speaker-verification benchmark. Real voices, real mics and real
rooms will move these numbers; that's why the default leans safe, the threshold is a live setting, and every
score is logged.

### Behaviour

- **Checked:** the turn request, every follow-up, every confirm answer, each dictated utterance, the unmute
  utterance while muted. `Orchestrator._speaker_ok` → `SpeakerGate.check` on a worker thread.
- **Rejected = silence.** Request / follow-up: the turn ends (no "didn't catch that", no brain turn) and the HUD
  shows an "Ignored another voice" card. It is transcribed only so that "forget my voice" can still get through —
  a profile that stopped matching the user (new mic, a cold) must not lock them out; that phrase is the one
  exemption. Dictation: not typed, keep listening (same card). **Confirm:** never transcribed, so never classified — it
  can't approve and it can't redirect. The confirm listens once more (as it already does for its own echo); a second
  rejection, like silence, is a deny ("Okay, skipping that."). The confirm gate only ever gets stricter.
- **Push-to-talk captures are not checked:** the key is the proof. (PTT during a confirm is a normal confirm
  capture, so it is checked.)
- **Wake:** speaker-agnostic unless `speaker_verification_wake` is on (whisper engine only; openwakeword has no
  window to embed). On, a rejected wake match is dropped and the engine keeps listening; barge-in uses the same wait.
- **Never waits, fails open.** `check()` never downloads or loads: until the model is loaded (at startup when a
  profile exists, or by enrolment) it accepts and starts a background load. No profile, verification off, or a
  model that won't load all mean "accept" — a broken model must not make her deaf, and it's exactly the
  behaviour before this feature. A load failure is visible: a one-time "Voice check unavailable, hearing
  everyone" card, and Settings says so; it is retried in the background at most every 10 minutes. A single
  scoring error accepts that one capture and is logged, without latching. Downloads have a 30 s timeout per
  socket operation and run one at a time.
- **Logged:** `speaker confirm: score=0.123 threshold=0.30 -> ignored (0.7 s voiced of 2.2 s, 9 ms)`. The last 8
  (where, score, verdict, time) are shown in Settings → Listening under the buttons.

### Enrolment and storage

- "learn my voice" / "only listen to me" / "meri awaaz yaad rakho" / "sirf meri awaaz suno", or **Learn my voice**
  in Settings. The Settings button queues the turn on the announcement queue so it runs when she's idle, never
  beside a capture that already has the mic.
- She reads three fixed English lines (no wake word in them); the user repeats each after the chime; each take gets
  one retry. A take is refused unless it has ≥1 s of voiced audio, whisper hears at least half the line's words
  in it, and its voice is not hers (similarity to her own synthesised line under 0.5 — `_is_own_speech` can't
  tell, since a correct take *is* her words). Without these, three takes of the same wrong source (her tail, a
  TV, a fan) would agree with each other and make a profile that ignores the user. The profile is the normalised mean of the three embeddings; if any clip agrees with the mean
  of the others under 0.3 (someone else answered a line) nothing is saved: "Those didn't sound like one voice."
- Stored as `~/.veronica/voice_profile.json` (0600, written atomically): version, model name, date, clip count,
  the 512 floats. A profile made with another model is ignored with a warning.
- "forget my voice" / "meri awaaz bhool jao" / **Forget my voice** deletes it. The phrase is matched before the
  memory intents, or it would be taken as a fact to forget.

## Settings

Hand-listed like every other setting (`EDITABLE_SETTINGS`, `SETTING_SECTIONS["listening"]`, rows in
`settings.js`), all live (no restart): Reduce background noise, Speech level floor, Only listen to my voice, Voice
match strictness (0.2–0.7), Only wake for my voice. Plus a read-only block: profile status, Learn / Forget buttons,
recent scores.

## Tests

- Hermetic: fake ONNX sessions and fake models throughout; a conftest fixture stubs the suppressor (and its
  startup fetch) out of every non-`live` test so a model on disk can't change results. New: `test_denoise.py`
  (STFT round trip, AGC, high-pass, state, loading, checksum pinning), `test_fbank.py`, `test_speaker.py`,
  `test_voice_isolation.py` (orchestrator: requests, follow-ups, confirm never approved/redirected, dictation,
  unmute, PTT exemption, enrol/forget turns), plus additions to the recorder, wake, intents, bridge and Settings
  page tests.
- `live`: `test_voice_isolation_live.py` runs the real models on `tests/fixtures/voice/*.flac` (280 KB of Kokoro
  clips) through the recorder's `frames` hook — no microphone is opened.
