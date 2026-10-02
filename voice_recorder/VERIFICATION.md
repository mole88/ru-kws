# Portable recorder verification

Verified on the build host: Windows 10 x64, Python 3.12.0 build environment.

- Migrated two original sessions to `dist/recordings`: 129 files, including
  62 WAV files and 60 accepted clip manifest entries. All files match their
  pre-migration SHA-256 hashes in `dist/recordings-transfer-sha256.json`.
- Nine unit tests pass: WAV sample counts, timing regions, overflow handling,
  interrupted recording preservation, reviewed annotation validation, Audacity
  export, frozen paths, menu routing and Russian keyboard shortcuts.
- The one-file EXE passes its self-test, including bundled PortAudio loading
  and device enumeration (19 devices on the build host).
- `test_portable.py` passes after extracting the release ZIP to a temporary path
  containing spaces and Cyrillic characters, using an unrelated working
  directory, a system-only PATH and deliberately invalid PYTHONHOME/PYTHONPATH.
- Both external and embedded command files work. Menu entry/exit, CLI help,
  and conversion of a synthetic WAV's Audacity labels to reviewed JSON pass.
- The release ZIP contains no personal recordings.

Limits: no separate physical PC or clean Windows VM was available. These checks
verify independence from an installed Python environment on the build host;
they do not replace recording and playback with the recipient's microphone.
No real microphone audio was captured during automated verification.
