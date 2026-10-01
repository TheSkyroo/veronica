# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = [('C:/Users/lenovo/Desktop/batman/Projects/veronicA/veronica/build/pyinstaller/build.json', '.')]
binaries = []
hiddenimports = ['pystray._win32', 'webview.platforms.winforms', 'webview.platforms.edgechromium', 'comtypes.stream', 'pycaw.pycaw']
datas += collect_data_files('veronica')
datas += collect_data_files('faster_whisper')
datas += collect_data_files('openwakeword')
datas += collect_data_files('kokoro_onnx')
datas += collect_data_files('language_tags')
datas += collect_data_files('phonemizer')
hiddenimports += collect_submodules('veronica')
tmp_ret = collect_all('webview')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pystray')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('espeakng_loader')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('claude_agent_sdk')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('winrt')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['C:/Users/lenovo/Desktop/batman/Projects/veronicA/veronica/build/pyinstaller/veronica_entry.py'],
    pathex=['C:/Users/lenovo/Desktop/batman/Projects/veronicA/veronica'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['C:/Users/lenovo/Desktop/batman/Projects/veronicA/veronica/scripts/pyinstaller_hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Veronica',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:/Users/lenovo/Desktop/batman/Projects/veronicA/veronica/assets/Veronica.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Veronica',
)
