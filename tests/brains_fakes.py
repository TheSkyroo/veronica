"""Fakes shared by the CliBrain backend tests: a canned-output child
process and a spawn that hands it out."""
import asyncio


class FakeStdin:
    def __init__(self):
        self.lines: list[str] = []

    def write(self, data: bytes) -> None:
        self.lines.append(data.decode())

    async def drain(self) -> None:
        pass


class FakeProc:
    """Streams `lines` from stdout, then EOF (or hangs when `hang`, until
    killed). `wait()` blocks only while a hanging, un-killed child is alive."""

    def __init__(self, lines, *, exit_code=0, hang=False):
        self._lines = [l if l.endswith("\n") else l + "\n" for l in lines]
        self.returncode = None
        self.signals = []
        self.killed = False
        self._hang = hang
        self._exit = exit_code
        self.stdout = self
        self.stdin = FakeStdin()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_eof()

    def feed(self, lines) -> None:
        """persistent mode: the next turn's output."""
        self._lines += [l if l.endswith("\n") else l + "\n" for l in lines]

    async def readline(self):
        if self._lines:
            await asyncio.sleep(0)
            return self._lines.pop(0).encode()
        if self._hang and not self.killed:
            await asyncio.sleep(3600)
        self.returncode = self._exit if not self.killed else -9
        return b""

    def send_signal(self, sig):
        self.signals.append(sig)

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        while self.returncode is None:
            if not self._hang or self.killed:
                self.returncode = -9 if self.killed else self._exit
                break
            await asyncio.sleep(0.01)
        return self.returncode
