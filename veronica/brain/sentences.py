import re

# sentence ends at . ! ? followed by whitespace and uppercase (so "3.14" and "p.m." don't split)
_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_MD = re.compile(r"[*_`#]+")


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
        if stripped_buf and stripped_buf[-1] in '.!?':
            result.append(stripped_buf)
            self._buf = ""

        return result

    def flush(self) -> list[str]:
        tail = self._buf.strip()
        self._buf = ""
        return [tail] if tail else []
