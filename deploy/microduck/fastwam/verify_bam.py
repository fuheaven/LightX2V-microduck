"""Check the small vendored CPU actuator subset against its pinned provenance."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    root = Path(sys.argv[1]).resolve()
    receipt = json.loads((root / "source-manifest.json").read_text())
    if receipt["revision"] != "62bd8ce12154340be97e06f7f41a0ca8f116d967":
        raise ValueError("BAM revision changed")
    if receipt["archive_sha256"] != "71e3b7f3a791c9577b934cb13154509892341b0006414d922fca0ace53cb8371":
        raise ValueError("BAM source archive changed")
    for name, expected in receipt["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"BAM file changed: {name}")
    print(f"Verified {len(receipt['files'])} pinned BAM source/parameter/license files")


if __name__ == "__main__":
    main()
