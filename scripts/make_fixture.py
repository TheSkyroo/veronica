import numpy as np
import soundfile as sf

from veronica.config import settings
from veronica.speech.tts import Synthesizer

s = Synthesizer(voice=settings.kokoro_voice, models_dir=settings.models_dir)
samples, sr = s.synth("What time is it")
# naive resample 24k -> 16k
idx = np.arange(0, len(samples), sr / 16000)
pcm = np.interp(idx, np.arange(len(samples)), samples)
sf.write("tests/fixtures/speech_1s.wav", (pcm * 32767).astype(np.int16), 16000)
print("wrote tests/fixtures/speech_1s.wav")
