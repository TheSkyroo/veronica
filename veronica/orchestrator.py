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
        self._confirm_capturing = False

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
        first = True
        self.player.reset()
        queue: asyncio.Queue = asyncio.Queue(maxsize=2)

        async def producer():
            try:
                async for sent in self.brain.ask(text):
                    spoken.append(sent)
                    # Enqueue at yield time (synth kicked off but not
                    # necessarily finished) so: (a) maxsize=2 bounds how far
                    # ahead synthesis can run, and (b) a concurrent confirm()
                    # joining the queue sees this sentence as pending *before*
                    # it's been spoken, closing the production ordering race.
                    fut = asyncio.ensure_future(self.tts.asynth(sent))
                    try:
                        await queue.put((sent, fut))
                    except BaseException:
                        # if put() itself is cancelled (e.g. the queue was
                        # full when handle_text was cancelled), fut was never
                        # handed to anything that would cancel it — it'd
                        # otherwise run to completion orphaned.
                        fut.cancel()
                        raise
            finally:
                await queue.put(None)

        def _cancel_or_reap(fut: asyncio.Future) -> None:
            # Cancel a not-yet-done synth future; for one that already
            # completed (possibly with an exception) before we got to it,
            # retrieve the result instead so asyncio doesn't complain about
            # an exception that was never retrieved.
            if not fut.cancel():
                with contextlib.suppress(BaseException):
                    fut.exception()

        def _drain(q: asyncio.Queue) -> None:
            while True:
                try:
                    item = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is not None:
                    _cancel_or_reap(item[1])
                q.task_done()

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
                try:
                    if item is None:
                        break
                    sent, fut = item
                    samples, _ = await fut
                    if first:
                        first = False
                        self._set("speaking")
                        log.info("latency first-sentence=%.2fs", time.monotonic() - t0)
                    if not self.muted:
                        async with self._speech_lock:
                            await self.player.play(samples)
                finally:
                    # Accounted for whether this item played cleanly, raised,
                    # or we were cancelled mid-item — unfinished_tasks must
                    # always balance to zero so nothing can join() forever.
                    queue.task_done()
            await prod
        finally:
            # Cancellation-safe teardown: whether we exit normally, via an
            # exception raised from the loop above, or because this task
            # itself was cancelled, the producer must be stopped and the
            # queue must never be left with unbalanced put()/task_done()
            # counts — an unbalanced queue would hang any confirm() blocked
            # in queue.join() forever.
            prod.cancel()
            # Drain BEFORE awaiting prod: if the queue was full, the
            # producer's own `finally: await queue.put(None)` would block
            # forever with nobody left to consume it. Freeing space here
            # lets that put() (and thus `await prod` below) complete.
            _drain(queue)
            with contextlib.suppress(BaseException):
                await prod
            # The producer's finally may have just put its None sentinel
            # (normal exit already consumed it above, so this is a no-op
            # then); drain it too so unfinished_tasks balances to zero.
            _drain(queue)
            self._speech_queue = None
        if not spoken:
            await self.say("I have nothing to say to that.")
        return spoken

    # -- confirmation gate ----------------------------------------------------
    async def confirm(self, summary: str) -> bool:
        queue = self._speech_queue
        if queue is not None:
            # don't jump ahead of sentences already queued for playback by
            # an in-flight handle_text pipeline. Bounded by brain_timeout_s
            # as a belt-and-braces guard against ever hanging here.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(queue.join(), timeout=self.s.brain_timeout_s)
        async with self._speech_lock:
            self.player.reset()
            await self._say_unlocked(f"Run {summary}?")
            self._confirm_capturing = True
            try:
                pcm = await self.recorder.capture(max_s=max(1, self.s.confirm_listen_s))
            finally:
                self._confirm_capturing = False
            if pcm is None:
                return False
            heard = await self.stt.atranscribe(pcm)
        ok = self.is_confirmation(heard)
        log.info("confirm heard=%r -> %s", heard, ok)
        return ok

    # -- barge-in ---------------------------------------------------------------
    async def _run_with_barge(self, coro) -> bool:
        """Run a turn coroutine; return True if the wake word interrupted it."""
        turn = asyncio.ensure_future(coro)
        listener = asyncio.create_task(self.wake.wait(threshold=self.s.barge_threshold))
        try:
            done, _ = await asyncio.wait({turn, listener}, return_when=asyncio.FIRST_COMPLETED)
            if listener in done and listener.result():
                log.info("barge-in")
                self.player.stop()
                if self._confirm_capturing:
                    # confirm() is blocked in recorder.capture(); wake its
                    # thread so it returns None promptly instead of being
                    # orphaned when we cancel the turn below.
                    self.recorder.stop()
                turn.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await turn
                await self.brain.interrupt()
                return True
            if listener not in done:
                # turn finished first; the listener is still running, ask its
                # thread to exit.
                self.wake.stop()
                await listener
            if not turn.done():
                # listener resolved (without a barge) before the turn did;
                # just wait the rest of the way for the turn to finish.
                await turn
            turn.result()  # re-raise turn errors
            return False
        finally:
            # Never leave either task pending, however we got here (normal
            # return, a re-raised turn error, or an exception out of the
            # listener itself, e.g. listener.result() raising because wait()
            # raised). Only stop() a listener that's still running here (one
            # already resolved True above and stopping it again would poison
            # the next wait() with a spurious immediate False).
            if not listener.done():
                self.wake.stop()
                with contextlib.suppress(BaseException):
                    await listener
            if not turn.done():
                turn.cancel()
                with contextlib.suppress(BaseException):
                    await turn

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
                barged = await self._run_with_barge(self.handle_text(text))
                if barged:
                    self._set("listening")
                    await self.chime(self.s.chime_wake_hz, 120)
                    pcm = await self.recorder.capture(max_s=self.s.listen_wait_s)
                    if pcm is None:
                        break
                    continue
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
                detected = await self.wake.wait()
            except Exception:
                log.exception("wake listener failed; retrying in %s s", self.s.wake_retry_s)
                self._set("error")
                await asyncio.sleep(self.s.wake_retry_s)
                continue
            if not detected:
                # a stale one-shot stop() (e.g. consumed in the same frame a
                # prior barge listener ended) must not start a spurious turn.
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
