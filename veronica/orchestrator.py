import logging
import re
import time
from collections.abc import Callable

from veronica.config import Settings

log = logging.getLogger("veronica.orchestrator")


class Orchestrator:
    CONFIRM_WORDS = frozenset({"yes", "yeah", "yep", "do it", "go ahead", "confirm", "sure"})
    DENY_WORDS = frozenset({"no", "nope", "not", "don't", "dont", "cancel", "stop", "never"})

    @staticmethod
    def is_confirmation(heard: str) -> bool:
        words = re.sub(r"[^a-z ]", " ", heard.lower()).split()
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

    def _set(self, state: str) -> None:
        self.state = state
        log.info("state=%s", state)
        self._on_state(state)

    # -- speaking -------------------------------------------------------------
    async def say(self, text: str) -> None:
        samples, _ = await self.tts.asynth(text)
        await self.player.play(samples)

    async def handle_text(self, text: str) -> list[str]:
        """Ask the brain and speak each sentence as it arrives. Returns sentences."""
        self._set("thinking")
        t0 = time.monotonic()
        spoken: list[str] = []
        self.player.reset()
        async for sent in self.brain.ask(text):
            if not spoken:
                self._set("speaking")
                log.info("latency first-sentence=%.2fs", time.monotonic() - t0)
            spoken.append(sent)
            await self.say(sent)
        if not spoken:
            await self.say("I have nothing to say to that.")
        return spoken

    # -- confirmation gate ----------------------------------------------------
    async def confirm(self, summary: str) -> bool:
        self.player.stop()
        self.player.reset()
        await self.say(f"Run {summary}?")
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
        pcm = await self.recorder.capture()
        while True:
            if pcm is None:
                break
            text = await self.stt.atranscribe(pcm)
            log.info("heard=%r", text)
            if not text:
                self.player.reset()
                await self.say("Sorry, didn't catch that.")
            else:
                await self.handle_text(text)
            self._set("followup")
            pcm = await self.recorder.capture(max_s=max(1, self.s.followup_window_s))
        self._set("idle")

    async def run_forever(self) -> None:
        self._set("idle")
        while True:
            await self.wake.wait()
            try:
                await self.one_turn()
            except Exception:
                log.exception("turn failed")
                try:
                    self.player.reset()
                    await self.say("Something went wrong, check the log.")
                except Exception:
                    log.exception("failed to report error")
                finally:
                    self._set("idle")
