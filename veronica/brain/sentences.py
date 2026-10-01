import re

# sentence ends at . ! ? or the Devanagari danda (।) followed by whitespace and
# an uppercase/digit/Devanagari char (so "3.14" and "p.m." don't split)
_END = re.compile(r"(?<=[.!?।])\s+(?=[A-Z0-9ऀ-ॿ])")
_MD = re.compile(r"[*_`#]+")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def has_devanagari(text: str) -> bool:
    """True if `text` contains any Devanagari (U+0900-U+097F). Lives here,
    not in speech.tts, so callers that only need the script check (the
    orchestrator) don't import kokoro_onnx for it."""
    return bool(_DEVANAGARI.search(text or ""))


class SentenceSplitter:
    """Accumulates streamed text and emits complete sentences."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text: str) -> list[str]:
        self._buf += _MD.sub("", text)
        parts = _END.split(self._buf)
        result = []

        if len(parts) > 1:
            # Found sentence boundaries
            result = [p.strip() for p in parts[:-1] if p.strip()]
            self._buf = parts[-1]

        # Check if buffer ends with sentence-ending punctuation (at end of current input)
        stripped_buf = self._buf.rstrip()
        if stripped_buf and stripped_buf[-1] in '.!?।':
            result.append(stripped_buf)
            self._buf = ""

        return result

    def flush(self) -> list[str]:
        tail = self._buf.strip()
        self._buf = ""
        return [tail] if tail else []
