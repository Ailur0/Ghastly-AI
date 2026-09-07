# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    # context/ is deliberately NOT bundled. It holds interview-state.json —
    # the questions and answers from whoever last ran this from source — and
    # bundling it shipped one person's transcript to everyone who got the
    # exe. The app creates the folder it needs beside the executable on first
    # run, so nothing is lost by leaving it out.
    datas=[('.env', '.')],
    hiddenimports=['pypdf', 'PIL', 'PIL.Image', 'PIL.JpegImagePlugin', 'soundcard'],
    hookspath=[],
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
    a.binaries,
    a.datas,
    [],
    name='SystemAudioHelper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
