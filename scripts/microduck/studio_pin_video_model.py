"""Record an operator-provided local H3 snapshot. No downloads or credentials."""

import argparse
import json
from pathlib import Path

from lightx2v_microduck.studio.schema import file_hash, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--revision", required=True, help="Exact upstream commit, not main")
    args = p.parse_args()
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        p.error("revision must be a full immutable commit hash")
    root = args.model.resolve()
    if not (root / "modular_model_index.json").is_file() or not (root / "transformer_ref").is_dir():
        p.error("expected a local MiniMax-H3 Ref2VA snapshot")
    files = {str(f.relative_to(root)): file_hash(f) for f in sorted(root.rglob("*")) if f.is_file() and ".cache" not in f.parts and f.name != "studio-model-manifest.json"}
    write_json(root / "studio-model-manifest.json", {"revision": args.revision, "files": files})
    print(json.dumps({"files": len(files), "revision": args.revision}))


if __name__ == "__main__":
    main()
