"""Shared test doubles.

`FakeRun` stands in for `subprocess.run`: it records every argv it's called
with and answers from a script keyed by the joined argv, so version/updater
tests never touch real git/uv.
"""
from __future__ import annotations

import subprocess


class FakeRun:
    def __init__(self, script: dict[str, tuple[int, str, str] | Exception] | None = None):
        # key: " ".join(argv) -> (returncode, stdout, stderr) or an exception to raise
        self.script = dict(script or {})
        self.calls: list[list[str]] = []
        self.kwargs: list[dict] = []

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append(argv)
        self.kwargs.append(kwargs)
        key = " ".join(argv)
        if key not in self.script:
            return subprocess.CompletedProcess(argv, 1, "", f"unscripted: {key}")
        entry = self.script[key]
        if isinstance(entry, Exception):
            raise entry
        rc, out, err = entry
        return subprocess.CompletedProcess(argv, rc, out, err)

    @property
    def argv_strings(self) -> list[str]:
        return [" ".join(c) for c in self.calls]
