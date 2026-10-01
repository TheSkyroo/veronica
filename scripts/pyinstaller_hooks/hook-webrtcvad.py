# Overrides PyInstaller's bundled contrib hook-webrtcvad.py, which hardcodes
# copy_metadata('webrtcvad') and crashes the build: we depend on the
# `webrtcvad-wheels` fork, whose module is `webrtcvad` but whose distribution
# metadata is registered under `webrtcvad-wheels` -- so the stock hook raises
# PackageNotFoundError. A user hook outranks the bundled one (PyInstaller keeps
# only the highest-priority hook per module), so this is the one that runs.
from PyInstaller.utils.hooks import copy_metadata

datas = copy_metadata("webrtcvad-wheels")
