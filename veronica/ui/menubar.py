import asyncio
import logging
import queue
import threading

import rumps

from veronica.__main__ import build_orchestrator
from veronica.config import settings
from veronica.ui.hud import HudWindow

ICONS = {"idle": "◯", "listening": "◉", "thinking": "…", "speaking": "♪", "followup": "◎", "error": "✕", "warming": "…", "confirming": "?"}


class _NoopHud:
    """Stand-in for HudWindow when the HUD is disabled or unavailable, so
    _drain/quit never need to branch on whether a real HUD exists."""

    def push(self, event: dict) -> None:
        pass

    def on_state(self, state: str) -> None:
        pass

    def tick(self) -> None:
        pass

    def show(self) -> None:
        pass

    def hide(self) -> None:
        pass

    def close(self) -> None:
        pass


class VeronicaApp(rumps.App):
    def __init__(self) -> None:
        super().__init__("V ◯", quit_button=None)
        self._state = "idle"
        self._muted = False
        self._quitting = False
        self.menu = [rumps.MenuItem("Mute", callback=self.toggle_mute), None, rumps.MenuItem("Quit", callback=self.quit)]
        hud = HudWindow(settings) if settings.hud_enabled else None
        self._hud = hud if (hud is not None and hud.available) else _NoopHud()
        self._events: queue.Queue = queue.Queue()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._timer = rumps.Timer(self._refresh, 0.25)
        self._timer.start()
        self._hud_timer = rumps.Timer(self._drain, 1 / 30)
        self._hud_timer.start()

    # asyncio side (background thread)
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            # model construction (and any first-run whisper download) happens
            # inside build_orchestrator, so surface "warming" before calling
            # it rather than only once warmup() starts.
            self._state = "warming"
            self._orch = build_orchestrator(
                settings, on_state=self._on_state, on_event=lambda k, p: self._events.put((k, p))
            )
            self._loop.run_until_complete(self._orch.warmup())
            self._loop.run_until_complete(self._orch.run_forever())
        except Exception as e:  # surface startup failures (no mic, not logged in)
            if self._quitting:
                # loop.stop() makes run_until_complete raise "Event loop stopped
                # before Future completed" — an expected part of a clean quit,
                # not a real failure.
                return
            self._state = "error"
            self._error = str(e)
            logging.getLogger("veronica.ui").exception("menu bar background loop failed")

    def _on_state(self, state: str) -> None:
        self._state = state

    # AppKit side (main thread)
    def _refresh(self, _timer) -> None:
        if self._muted:
            self.title = "V zz"
        elif self._state == "error":
            self.title = "V ✕"
        else:
            self.title = f"V {ICONS.get(self._state, '?')}"

    def _drain(self, _timer) -> None:
        # If the backlog has grown past 1000 (the HUD/UI thread falling
        # behind the producer), drop this batch's mic level rather than
        # push a stale one: mic is a continuously-refreshed level meter,
        # so the freshest reading is always about to replace it anyway,
        # and skipping a push here is strictly cheaper than catching up.
        overflow = self._events.qsize() > 1000
        popped = []
        for _ in range(64):
            try:
                popped.append(self._events.get_nowait())
            except queue.Empty:
                break
        last_mic = None
        for kind, payload in popped:
            if kind == "mic":
                last_mic = payload
                continue
            if kind == "state":
                self._hud.on_state(payload)
            self._hud.push({"kind": kind, "payload": payload})
        if last_mic is not None and not overflow:
            self._hud.push({"kind": "mic", "payload": last_mic})
        self._hud.tick()

    def toggle_mute(self, item: rumps.MenuItem) -> None:
        self._muted = not self._muted
        item.state = self._muted
        # stop any speech; the wake loop keeps running but muting is honoured in _refresh only.
        orch = getattr(self, "_orch", None)
        if orch is not None:
            orch.muted = self._muted
            if self._muted:
                orch.player.stop()

    def quit(self, _item) -> None:
        self._quitting = True
        self._hud.close()
        orch = getattr(self, "_orch", None)
        if orch is not None:
            orch.player.close()
        self._loop.call_soon_threadsafe(self._loop.stop)
        rumps.quit_application()


def run_app() -> None:
    VeronicaApp().run()
