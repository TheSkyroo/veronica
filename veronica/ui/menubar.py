import asyncio
import logging
import threading

import rumps

from veronica.__main__ import build_orchestrator
from veronica.config import settings

ICONS = {"idle": "◯", "listening": "◉", "thinking": "…", "speaking": "♪", "followup": "◎", "error": "✕"}


class VeronicaApp(rumps.App):
    def __init__(self) -> None:
        super().__init__("V ◯", quit_button=None)
        self._state = "idle"
        self._muted = False
        self.menu = [rumps.MenuItem("Mute", callback=self.toggle_mute), None, rumps.MenuItem("Quit", callback=self.quit)]
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._timer = rumps.Timer(self._refresh, 0.25)
        self._timer.start()

    # asyncio side (background thread)
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._orch = build_orchestrator(settings, on_state=self._on_state)
            self._loop.run_until_complete(self._orch.run_forever())
        except Exception as e:  # surface startup failures (no mic, not logged in)
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

    def toggle_mute(self, item: rumps.MenuItem) -> None:
        self._muted = not self._muted
        item.state = self._muted
        # stop any speech; the wake loop keeps running but muting is honoured in _refresh only.
        orch = getattr(self, "_orch", None)
        if self._muted and orch is not None:
            orch.player.stop()

    def quit(self, _item) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        rumps.quit_application()


def run_app() -> None:
    VeronicaApp().run()
