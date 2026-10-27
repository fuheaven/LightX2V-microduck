"""Verify the unmodified reference architecture before starting a training image."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[3] / "third_party/fastwam"
manifest = json.loads((root / "source.json").read_text())
for name, sha in manifest["files"].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha:
        raise ValueError(f"FastWAM source differs from pinned version: {name}")
print("Verified FastWAM", manifest["commit"], len(manifest["files"]), "files")
