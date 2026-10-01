"""Build-time only: force onnxruntime to load first in every interpreter
PyInstaller spawns during the build.

PyInstaller's Windows binary-dependency analysis imports *every* collected
package into one subprocess to track DLL search-path changes. Our dependency
graph has that subprocess import a winrt module (winrt.windows.media.ocr,
which loads the system WinML/onnxruntime) before it reaches onnxruntime --
and importing our bundled onnxruntime *after* WinML segfaults the process,
aborting the build. Loading onnxruntime first avoids the clash.

This dir is put on PYTHONPATH only for the PyInstaller subprocess (see
scripts/build_app.py), so site.py imports this at interpreter startup --
before PyInstaller imports anything. It is never bundled and never on the
path during normal dev/test runs.
"""
try:
    import onnxruntime  # noqa: F401
except Exception:
    # onnxruntime missing/broken is the build's problem to report, not ours.
    pass
