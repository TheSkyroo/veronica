import asyncio
import contextlib
import logging
import re
import time
from collections.abc import Callable

import numpy as np

from veronica.audio.chime import tone
from veronica.config import Settings

log = logging.getLogger("veronica.orchestrator")


class Orchestrator:
    CONFIRM_WORDS = frozenset({"yes", "yeah", "yep", "do it", "go ahead", "confirm", "sure"})
    DENY_WORDS = frozenset({"no", "nope", "not", "don't", "dont", "cancel", "stop", "never"})

    @staticmethod
    def is_confirmation(heard: str) -> bool:
        no_apostrophes = heard.lower().replace("'", "").replace("’", "")
        words = re.sub(r"[^a-z ]", " ", no_apostrophes).split()
        if any(w in Orchestrator.DENY_WORDS for w in words):
            return False
        for phrase in Orchestrator.CONFIRM_WORDS:
            phrase_words = phrase.split()
            n = len(phrase_words)
            for i in range(len(words) - n + 1):
                if words[i:i + n] == phrase_words:
                    return True
        return False

    def __init__(self, settings: Settings, *, wake, recorder, stt, brain, tts, player,
                 on_state: Callable[[str], None] | None = None) -> None:
        self.s = settings
        self.wake, self.recorder, self.stt = wake, recorder, stt
        self.brain, self.tts, self.player = brain, tts, player
        self._on_state = on_state or (lambda _: None)
        self.state = "idle"
        self.muted = False
        self.ready = False
        self._speech_lock = asyncio.Lock()
        self._speech_queue: asyncio.Queue | None = None

    async def warmup(self) -> None:
        """Load models before the first turn so the first answer isn't slow."""
        self._set("warming")
        t0 = time.monotonic()
        await self.tts.asynth("ok")
        if self.stt is not None:
            await self.stt.atranscribe(np.zeros(16000, dtype=np.int16))
        log.info("warmup done in %.1fs", time.monotonic() - t0)
        self.ready = True
        self._set("idle")

    def _set(self, state: str) -> None:
        self.state = state
        log.info("state=%s", state)
        self._on_state(state)

    # -- speaking -------------------------------------------------------------
    async def _say_unlocked(self, text: str) -> None:
        samples, _ = await self.tts.asynth(text)
        await self.player.play(samples)

    async def say(self, text: str) -> None:
        if self.muted:
            return
        async with self._speech_lock:
            await self._say_unlocked(text)

    async def chime(self, freq_hz: float, ms: int) -> None:
        if self.muted:
            return
        async with self._speech_lock:
            self.player.reset()
            await self.player.play(tone(freq_hz, ms))

    async def handle_text(self, text: str) -> list[str]:
        """Ask the brain and speak each sentence; synth N+1 overlaps playback of N."""
        self._set("thinking")
        t0 = time.monotonic()
        spoken: list[str] = []
        self.player.reset()
        queue: asyncio.Queue = asyncio.Queue(maxsize=2)

        async def producer():
            try:
                async for sent in self.brain.ask(text):
                    spoken.append(sent)
                    samples, _ = await self.tts.asynth(sent)
                    await queue.put((sent, samples))
            finally:
                await queue.put(None)

        # Exposed so confirm() (which the brain may await mid-stream, e.g. as
        # a tool-use confirmation gate) can wait for already-queued sentences
        # to finish playing before it speaks its own prompt — otherwise it
        # could win the _speech_lock race against the consumer below and
        # jump the queue.
        self._speech_queue = queue
        prod = asyncio.create_task(producer())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                sent, samples = item
                if len(spoken) and sent == spoken[0]:
                    self._set("speaking")
                    log.info("latency first-sentence=%.2fs", time.monotonic() - t0)
                if not self.muted:
                    async with self._speech_lock:
                        await self.player.play(samples)
                queue.task_done()
            await prod
        except BaseException:
            prod.cancel()
            with contextlib.suppress(BaseException):
                await prod
            raise
        finally:
            self._speech_queue = None
        if not spoken:
            await self.say("I have nothing to say to that.")
        return spoken

    # -- confirmation gate ----------------------------------------------------
    async def confirm(self, summary: str) -> bool:
        if self._speech_queue is not None:
            # don't jump ahead of sentences already queued for playback by
            # an in-flight handle_text pipeline.
            await self._speech_queue.join()
        async with self._speech_lock:
            self.player.reset()
            await self._say_unlocked(f"Run {summary}?")
            pcm = await self.recorder.capture(max_s=max(1, self.s.confirm_listen_s))
            if pcm is None:
                return False
            heard = await self.stt.atranscribe(pcm)
        ok = self.is_confirmation(heard)
        log.info("confirm heard=%r -> %s", heard, ok)
        return ok

    # -- one interaction ------------------------------------------------------
    async def one_turn(self) -> None:
        """Called after wake word: listen, answer, then follow-up window."""
        self._set("listening")
        await self.chime(self.s.chime_wake_hz, 120)
        pcm = await self.recorder.capture(max_s=self.s.listen_wait_s)
        if pcm is None:
            self._set("idle")
            return
        while True:
            text = await self.stt.atranscribe(pcm)
            log.info("heard=%r", text)
            if not text:
                self.player.reset()
                await self.say("Sorry, didn't catch that.")
            else:
                await self.handle_text(text)
            self._set("followup")
            await self.chime(self.s.chime_followup_hz, 100)
            pcm = await self.recorder.capture(max_s=max(1, self.s.followup_window_s))
            if pcm is None:
                break
        self._set("idle")

    async def run_forever(self) -> None:
        self._set("idle")
        while True:
            try:
                await self.wake.wait()
            except Exception:
                log.exception("wake listener failed; retrying in %s s", self.s.wake_retry_s)
                self._set("error")
                await asyncio.sleep(self.s.wake_retry_s)
                continue
            if self.muted:
                log.info("muted; skipping turn")
                continue
            try:
                await self.one_turn()
            except Exception as exc:
                log.exception("turn failed")
                try:
                    self.player.reset()
                    detail = f"{type(exc).__name__} {exc}".lower()
                    if any(k in detail for k in ("login", "logged in", "authenticat")):
                        message = "Claude Code isn't logged in."
                    else:
                        message = "Something went wrong, check the log."
                    await self.say(message)
                except Exception:
                    log.exception("failed to report error")
                finally:
                    self._set("idle")
