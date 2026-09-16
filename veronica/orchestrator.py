import asyncio
import contextlib
import logging
import re
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from veronica.audio.chime import tone
from veronica.brain.intents import match_intent, match_memory_intent, normalize
from veronica.config import Settings
from veronica.ui.events import envelope

log = logging.getLogger("veronica.orchestrator")


class Orchestrator:
    CONFIRM_WORDS = frozenset({"yes", "yeah", "yep", "do it", "go ahead", "confirm", "sure"})
    DENY_WORDS = frozenset({"no", "nope", "not", "don't", "dont", "cancel", "stop", "never"})
    _SPOKEN_END_PHRASES = frozenset({"thanks veronica", "thank you veronica"})

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
                 partial_stt=None, store=None,
                 on_state: Callable[[str], None] | None = None,
                 on_event: Callable[[str, Any], None] | None = None,
                 on_quit: Callable[[], None] | None = None) -> None:
        self.s = settings
        self.wake, self.recorder, self.stt = wake, recorder, stt
        self.brain, self.tts, self.player = brain, tts, player
        self.partial_stt = partial_stt
        self.store = store
        self._on_state = on_state or (lambda _: None)
        self._on_event = on_event
        self._on_quit = on_quit or (lambda: None)
        self.state = "idle"
        self._muted = False
        self._unmute_event = asyncio.Event()
        self.ready = False
        self._speech_lock = asyncio.Lock()
        self._speech_queue: asyncio.Queue | None = None
        self._confirm_capturing = False
        self._barged = False
        self._now_speaking = ""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._partial_task: asyncio.Task | None = None
        # Bumped every time a partial-eligible capture() returns, before STT
        # runs on it: any partial transcription still in flight (or one that
        # races in from the recorder thread right at that boundary) belongs
        # to a capture that's already over and must be dropped.
        self._partial_gen = 0
        if self.recorder is not None and self.partial_stt is not None:
            self.recorder.on_audio = self._on_recorder_audio
        # Own-speech suppression for the whisper wake engine: the mic's rolling
        # analysis window can still hold the tail of a just-finished sentence
        # (e.g. "...I'm Veronica") for up to wake_window_s + wake_hop_s after
        # _now_speaking is cleared, so we keep offering the last-spoken text to
        # the suppress check for that long too. self._clock is overridable in
        # tests to fake time.
        self._last_spoken = ""
        self._last_spoken_until = 0.0
        self._clock = time.monotonic
        self._announce_queue: asyncio.Queue = asyncio.Queue()

    @property
    def muted(self) -> bool:
        return self._muted

    @muted.setter
    def muted(self, value: bool) -> None:
        was_muted = self._muted
        self._muted = bool(value)
        if was_muted and not self._muted:
            # Unmuting: wake run_forever's wait (if it's parked there) so any
            # announcement that queued up while muted is delivered right
            # away instead of waiting for the next wake word. `muted` can be
            # set from a different thread (e.g. the menu bar's AppKit
            # thread), so this must go through call_soon_threadsafe rather
            # than setting the asyncio.Event directly.
            loop = self._loop
            if loop is not None:
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(self._unmute_event.set)
            else:
                self._unmute_event.set()

    async def warmup(self) -> None:
        """Load models before the first turn so the first answer isn't slow."""
        self._emit("warm", {"ready": False})
        self._set("warming")
        t0 = time.monotonic()
        await self.tts.asynth("ok")
        if self.stt is not None:
            await self.stt.atranscribe(np.zeros(16000, dtype=np.int16))
        if self.partial_stt is not None:
            await self.partial_stt.atranscribe(np.zeros(16000, dtype=np.int16))
        log.info("warmup done in %.1fs", time.monotonic() - t0)
        self.ready = True
        self._set("idle")
        self._emit("warm", {"ready": True})

    def _set(self, state: str) -> None:
        self.state = state
        log.info("state=%s", state)
        self._on_state(state)
        self._emit("state", state)

    def _emit(self, kind: str, payload) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(kind, payload)
        except Exception:
            log.exception("on_event failed for %s", kind)

    # -- speaking -------------------------------------------------------------
    def _finished_speaking(self, text: str) -> None:
        """Called right after a play() of `text` returns: keep offering it to
        the suppress check for wake_window_s + wake_hop_s more, since the
        mic's rolling analysis window can still hold its audio tail."""
        self._last_spoken = text
        self._last_spoken_until = self._clock() + self.s.wake_window_s + self.s.wake_hop_s
        self._now_speaking = ""

    def _suppress_text(self) -> str:
        if self._clock() < self._last_spoken_until:
            return f"{self._now_speaking} {self._last_spoken}"
        return self._now_speaking

    # -- live partial transcript -----------------------------------------------
    def _end_partial_window(self) -> None:
        """Called right after any partial-eligible recorder.capture() returns,
        before STT runs on the result: bumps the generation counter so a
        partial transcription still in flight (or one that races in from the
        recorder thread right at this boundary) is recognized as stale and
        dropped rather than emitted after — or worse, overwriting — this
        turn's real 'heard' text. Also cancels the in-flight task outright."""
        self._partial_gen += 1
        if self._partial_task is not None and not self._partial_task.done():
            self._partial_task.cancel()

    def _on_recorder_audio(self, pcm: np.ndarray) -> None:
        """Called from the Recorder's capture thread (not the event loop)
        every partial_hop_s of captured speech. Hands off to the loop
        thread-safely; coalescing (skip while a partial transcription is
        already running) happens there, not here."""
        loop = self._loop
        if loop is None:
            return
        gen = self._partial_gen
        try:
            loop.call_soon_threadsafe(self._schedule_partial, pcm, gen)
        except RuntimeError:
            pass  # loop closed/closing; drop this partial

    def _schedule_partial(self, pcm: np.ndarray, gen: int) -> None:
        if gen != self._partial_gen:
            return  # the capture this came from is already over
        if self._partial_task is not None and not self._partial_task.done():
            return  # a partial transcription is already in flight; coalesce
        self._partial_task = asyncio.ensure_future(self._run_partial(pcm, gen))

    async def _run_partial(self, pcm: np.ndarray, gen: int) -> None:
        try:
            text = await self.partial_stt.atranscribe(pcm)
        except Exception:
            log.exception("partial transcription failed")
            return
        if gen != self._partial_gen:
            return  # capture ended (or another one started) while transcribing
        if text:
            self._emit("heard_partial", text)

    async def _say_unlocked(self, text: str, kind: str = "sentence") -> None:
        # Called only while _speech_lock is already held (by say()/confirm()).
        # Emit right before play so a listener never sees "sentence"/"voice"
        # (or "prompt"/"voice") for audio that hasn't actually started playing
        # yet.
        samples, sr = await self.tts.asynth(text)
        self._emit("voice", {"step_ms": 50, "levels": envelope(samples, sr)})
        self._emit(kind, text)
        self._now_speaking = text
        try:
            await self.player.play(samples)
        finally:
            self._finished_speaking(text)

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
        self._barged = False   # fresh turn: any earlier barge no longer applies
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
                    samples, sr = await fut
                    if first:
                        first = False
                        self._set("speaking")
                        log.info("latency first-sentence=%.2fs", time.monotonic() - t0)
                    if not self.muted:
                        async with self._speech_lock:
                            # Emit right after acquiring the lock, immediately
                            # before play, so a listener never sees these
                            # events for audio that hasn't started yet.
                            self._emit("voice", {"step_ms": 50, "levels": envelope(samples, sr)})
                            self._emit("sentence", sent)
                            self._now_speaking = sent
                            try:
                                await self.player.play(samples)
                            finally:
                                self._finished_speaking(sent)
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
        elif self.store is not None and self.s.memory_enabled:
            self.store.add_turn(text, " ".join(spoken))
        return spoken

    # -- confirmation gate ----------------------------------------------------
    async def confirm(self, summary: str, detail: str = "") -> bool:
        if self.muted:
            log.info("confirm skipped (muted): %s", summary)
            return False
        prev = self.state
        self._set("confirming")
        result = False
        try:
            queue = self._speech_queue
            if queue is not None:
                # don't jump ahead of sentences already queued for playback by
                # an in-flight handle_text pipeline. Bounded by brain_timeout_s
                # as a belt-and-braces guard against ever hanging here.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(queue.join(), timeout=self.s.brain_timeout_s)
            if self._barged:
                # a barge landed while we were waiting for the queue to drain;
                # the turn this confirmation belongs to is already being torn
                # down, so don't speak the prompt or eat the follow-up capture.
                log.info("confirm aborted by barge")
                return result
            async with self._speech_lock:
                if self._barged:
                    # a barge landed while we were waiting to acquire the
                    # speech lock (e.g. a concurrent say()/chime() was still
                    # holding it); don't speak the prompt for a turn that's
                    # already being torn down.
                    log.info("confirm aborted by barge")
                    return result
                self.player.reset()
                await self._say_unlocked(f"Run {summary}?", kind="prompt")
                if self._barged:
                    # barged while the prompt was being spoken.
                    log.info("confirm aborted by barge")
                    return result
                # Emitted only now (after the question has actually been
                # spoken, immediately before we start listening) so the
                # listening window's countdown starts from when the user
                # could first respond, not from confirm()'s entry.
                self._emit("tool", {
                    "summary": summary, "detail": detail, "decision": "ask",
                    "timeout_ms": self.s.confirm_listen_s * 1000,
                })
                self._confirm_capturing = True
                try:
                    pcm = await self.recorder.capture(max_s=max(1, self.s.confirm_listen_s))
                finally:
                    self._confirm_capturing = False
                if pcm is None:
                    return result
                heard = await self.stt.atranscribe(pcm)
                if self._barged:
                    # barged while we were transcribing the reply.
                    log.info("confirm aborted by barge")
                    return result
                result = self.is_confirmation(heard)
                log.info("confirm heard=%r -> %s", heard, result)
        finally:
            self._set(prev)
            self._emit("tool", {"summary": summary, "decision": "allowed" if result else "declined"})
        return result

    # -- barge-in ---------------------------------------------------------------
    async def _run_with_barge(self, coro) -> bool:
        """Run a turn coroutine; return True if the wake word interrupted it."""
        turn = asyncio.ensure_future(coro)
        listener = asyncio.create_task(
            self.wake.wait(threshold=self.s.barge_threshold, suppress=self._suppress_text)
        )
        try:
            done, _ = await asyncio.wait({turn, listener}, return_when=asyncio.FIRST_COMPLETED)
            if listener in done and listener.exception() is not None:
                # mic hiccup or similar in the barge listener; the turn is
                # still good, so don't cancel it — just log and let it finish
                # normally, as if no barge listener were running at all.
                try:
                    listener.result()
                except Exception:
                    log.exception("barge listener failed")
                if not turn.done():
                    await turn
                return False
            if listener in done and listener.result():
                log.info("barge-in")
                self.player.stop()
                if self._confirm_capturing:
                    # confirm() is blocked in recorder.capture(); wake its
                    # thread so it returns None promptly instead of being
                    # orphaned when we cancel the turn below.
                    self.recorder.stop()
                self._barged = True
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
    async def _listen_after_wake(self) -> np.ndarray | None:
        """Chime (unless the wake engine's pre-roll already contains speech,
        i.e. the user spoke the command in the same breath as the wake
        word) and capture, handing that pre-roll to the recorder so it
        isn't lost."""
        self._set("listening")
        pre = self.wake.take_preroll()
        if not self.recorder.has_speech(pre):
            await self.chime(self.s.chime_wake_hz, 120)
        pcm = await self.recorder.capture(max_s=self.s.listen_wait_s, preroll=pre, partial=True)
        self._end_partial_window()
        return pcm

    async def one_turn(self) -> None:
        """Called after wake word: listen, answer, then follow-up window."""
        self._loop = asyncio.get_running_loop()
        pcm = await self._listen_after_wake()
        if pcm is None:
            self._set("idle")
            return
        is_followup = False
        while True:
            text = await self.stt.atranscribe(pcm)
            self._emit("heard", text)
            log.info("heard=%r", text)
            intent = match_intent(text)
            if intent == "end":
                if normalize(text) in self._SPOKEN_END_PHRASES:
                    self.player.reset()
                    await self.say("Okay.")
                self._emit("hud", {"mode": "hide"})
                self._set("idle")
                return
            if intent == "hud_hide":
                self._emit("hud", {"mode": "hide"})
                self._set("idle")
                return
            if intent == "mute":
                self.player.reset()
                await self.say("Muted.")
                self.muted = True
                self._emit("hud", {"mode": "hide"})
                self._set("idle")
                return
            mem = None if intent is not None else match_memory_intent(text)
            if intent in ("hud_mini", "hud_full"):
                self._emit("hud", {"mode": "mini" if intent == "hud_mini" else "full"})
                self.player.reset()
                await self.say("Okay.")
            elif intent == "quit":
                self.player.reset()
                if await self.confirm("Quit Veronica"):
                    await self.say("Goodbye.")
                    self._on_quit()
                    self._set("idle")
                    return
            elif mem is not None:
                kind, arg = mem
                self.player.reset()
                if kind == "remember":
                    if self.store is not None:
                        self.store.add_fact(arg)
                    await self.say("Got it.")
                else:
                    n = self.store.delete_fact_matching(arg) if self.store is not None else 0
                    await self.say("Forgotten." if n else "I didn't have that.")
            elif not text:
                if is_followup:
                    # A follow-up capture (not the first listen after wake,
                    # nor the re-listen after a barge) that came back empty
                    # just means the user didn't say anything more — go
                    # quiet rather than nag with "Sorry, didn't catch that."
                    # and reopen yet another follow-up window.
                    self._set("idle")
                    return
                self.player.reset()
                await self.say("Sorry, didn't catch that.")
            else:
                barged = await self._run_with_barge(self.handle_text(text))
                if barged:
                    pcm = await self._listen_after_wake()
                    is_followup = False
                    if pcm is None:
                        break
                    continue
            self._set("followup")
            await self.chime(self.s.chime_followup_hz, 100)
            pcm = await self.recorder.capture(
                max_s=max(1, self.s.followup_window_s), partial=True, skip_ms=self.s.followup_skip_ms
            )
            is_followup = True
            self._end_partial_window()
            if pcm is None:
                break
        self._set("idle")

    async def _muted_capture(self) -> None:
        """Called after a wake word fires while muted: capture exactly one
        utterance (no chime, no HUD show — state/HUD stay untouched) and
        check only whether it's the unmute phrase. Anything else (including
        silence) is ignored silently; run_forever goes straight back to
        idle either way."""
        self._loop = asyncio.get_running_loop()
        pcm = await self.recorder.capture(max_s=self.s.listen_wait_s)
        if pcm is None:
            return
        text = await self.stt.atranscribe(pcm)
        if match_intent(text) == "unmute":
            self.muted = False
            await self.say("I'm back.")

    # -- announcements ----------------------------------------------------------
    async def announce(self, text: str) -> None:
        """Queue `text` to be spoken next time we're idle (e.g. a timer
        firing): never interrupts an in-flight turn. Safe to call from any
        task (e.g. TimerService's on_fire callback)."""
        await self._announce_queue.put(text)

    async def _deliver_announcement(self, text: str) -> None:
        self._set("speaking")
        self.player.reset()
        await self.chime(self.s.chime_wake_hz, 120)
        await self.say(text)
        self._set("idle")

    async def run_forever(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._set("idle")
        while True:
            # Deliver anything queued while we were away (e.g. a timer that
            # fired mid-turn), unless muted — while muted, announcements just
            # sit in the queue (no state flicker) until unmuted.
            while not self.muted and not self._announce_queue.empty():
                await self._deliver_announcement(self._announce_queue.get_nowait())

            wake_task = asyncio.ensure_future(self.wake.wait())
            waiting_on_unmute = self.muted
            if waiting_on_unmute:
                # Don't consume the queue while muted: race the wake
                # listener against the unmute signal instead, so a queued
                # announcement is neither delivered (flicker) nor lost.
                self._unmute_event.clear()
                signal_task = asyncio.ensure_future(self._unmute_event.wait())
            else:
                signal_task = asyncio.ensure_future(self._announce_queue.get())
            done, _ = await asyncio.wait(
                {wake_task, signal_task}, return_when=asyncio.FIRST_COMPLETED
            )

            if wake_task not in done:
                # Only signal_task resolved: we're idle right now
                # (run_forever only waits here between turns), so stop the
                # wake listener, then loop back around to start a fresh one
                # — "resuming" it. An unmute signal just loops back to the
                # top, which delivers the now-unmuted queue; a real
                # announcement is spoken directly.
                self.wake.stop()
                with contextlib.suppress(BaseException):
                    await wake_task
                if not waiting_on_unmute:
                    await self._deliver_announcement(signal_task.result())
                continue

            if signal_task in done:
                if not waiting_on_unmute:
                    # Both resolved in the same tick: a genuine wake-word
                    # detection must not be swallowed by a same-tick
                    # announcement, so prefer the wake path and put the
                    # announcement back — it'll be delivered after this turn
                    # ends (top of the next iteration finds the queue
                    # non-empty).
                    self._announce_queue.put_nowait(signal_task.result())
            else:
                signal_task.cancel()
                with contextlib.suppress(BaseException):
                    await signal_task

            try:
                detected = wake_task.result()
            except Exception:
                log.exception("wake listener failed; retrying in %s s", self.s.wake_retry_s)
                self._set("error")
                await asyncio.sleep(self.s.wake_retry_s)
                self._set("idle")
                continue
            if not detected:
                # a stale one-shot stop() (e.g. consumed in the same frame a
                # prior barge listener ended) must not start a spurious turn.
                continue
            if self.muted:
                log.info("muted; capturing one utterance to check for unmute")
                await self._muted_capture()
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
