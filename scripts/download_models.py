"""Fetch Kokoro TTS model/voices and openwakeword base models into ~/.veronica/models.

Whisper models are fetched on first use by faster-whisper (into its own
Hugging Face cache); pass --whisper to prefetch the English pair now and
--hindi to also prefetch the multilingual pair the Hindi/auto language
modes use (~500 MB), so the first "speak hindi" doesn't stall on a download."""
import argparse
import urllib.request

from veronica.config import settings

KOKORO = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}


def prefetch_whisper(models: list[str]) -> None:
    """Construct each WhisperModel once so faster-whisper downloads it."""
    from faster_whisper import WhisperModel  # heavy import; only when asked

    for name in models:
        print(f"whisper {name}")
        WhisperModel(name, device="cpu", compute_type="int8")
        print(f"ok      whisper {name}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--whisper", action="store_true", help="also prefetch the English whisper models")
    p.add_argument("--hindi", action="store_true",
                   help="also prefetch the multilingual whisper models for Hindi/Hinglish (~500 MB)")
    args = p.parse_args(argv)
    settings.ensure_dirs()
    for name, url in KOKORO.items():
        dest = settings.models_dir / name
        if dest.exists():
            print(f"ok      {dest}")
            continue
        print(f"fetch   {url}")
        part = dest.with_suffix(dest.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, part)
            part.rename(dest)
        except Exception:
            part.unlink(missing_ok=True)
            raise
        print(f"saved   {dest}")

    import openwakeword
    openwakeword.utils.download_models()
    print("openwakeword models ready")

    if args.whisper:
        prefetch_whisper([settings.whisper_model, settings.partial_stt_model])
    if args.hindi:
        prefetch_whisper([settings.whisper_multilingual_model, settings.partial_stt_multilingual_model])


if __name__ == "__main__":
    main()
