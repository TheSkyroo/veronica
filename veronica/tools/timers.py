"""In-process timers: fire an announcement (and a notification banner) after
a delay, with no persistence — timers are lost on restart, by design."""
import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from veronica.tools import mac

log = logging.getLogger("veronica.timers")


class TimerService:
    def __init__(self, on_fire: Callable[[str], Awaitable[None]], clock: Callable[[], float] = time.monotonic):
        self._on_fire = on_fire
        self._clock = clock
        self._timers: dict[str, dict] = {}

    def set(self, minutes: float, label: str = "") -> str:
        tid = uuid.uuid4().hex[:8]
        delay_s = max(0.0, float(minutes) * 60)
        entry = {
            "id": tid,
            "label": label,
            "minutes": minutes,
            "fire_at": self._clock() + delay_s,
        }
        entry["task"] = asyncio.ensure_future(self._run(tid, delay_s))
        self._timers[tid] = entry
        return tid

    def list(self) -> list[dict]:
        now = self._clock()
        return [
            {
                "id": t["id"],
                "label": t["label"],
                "minutes": t["minutes"],
                "remaining_s": max(0.0, t["fire_at"] - now),
            }
            for t in self._timers.values()
        ]

    def cancel(self, label_or_id: str) -> bool:
        entry = self._timers.get(label_or_id)
        if entry is None:
            entry = next((t for t in self._timers.values() if t["label"] == label_or_id), None)
        if entry is None:
            return False
        entry["task"].cancel()
        self._timers.pop(entry["id"], None)
        return True

    async def _run(self, tid: str, delay_s: float) -> None:
        try:
            await asyncio.sleep(delay_s)
        except asyncio.CancelledError:
            return
        entry = self._timers.pop(tid, None)
        if entry is None:
            return
        label = entry["label"]
        text = f"Timer {label} done" if label else "Timer done"
        try:
            await self._on_fire(text)
        except Exception:
            log.exception("timer on_fire failed")
        with contextlib.suppress(Exception):
            await mac.notify.handler({"title": "Timer", "message": text})
