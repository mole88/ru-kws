"""Package only source files for uploading to Google Drive."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
destination = ROOT / "dist/ru-kws-source.zip"
destination.parent.mkdir(parents=True, exist_ok=True)
files = [ROOT / name for name in ("pyproject.toml", "README.md", "THIRD_PARTY_NOTICES.md", ".gitignore")]
for folder, suffixes in {
    "src/ru_kws": {".py", ".txt"},
    "configs": {".yaml"},
    "notebooks": {".ipynb"},
    "scripts": {".py"},
    "tests": {".py"},
}.items():
    files.extend(path for path in (ROOT / folder).rglob("*")
                 if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts)
with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(files):
        archive.write(path, "ru-kws/" + path.relative_to(ROOT).as_posix())
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
    assert "ru-kws/src/ru_kws/train.py" in archive.namelist()
print(f"{destination} ({destination.stat().st_size:,} bytes; {len(files)} files)")
