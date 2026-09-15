"""Fetch Kokoro TTS model/voices and openwakeword base models into ~/.veronica/models."""
import urllib.request

from veronica.config import settings

KOKORO = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}


def main() -> None:
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


if __name__ == "__main__":
    main()
