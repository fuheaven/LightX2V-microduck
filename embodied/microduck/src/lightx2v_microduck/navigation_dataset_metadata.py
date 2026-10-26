"""Bind language and episode identity to the same indexed navigation artifact."""

import hashlib
import json
from pathlib import Path


def read_indexed_metadata(root, entry, *, require_digest=False):
    root = Path(root).resolve()
    path = (root / entry["path"] / "episode.json").resolve()
    if not path.is_relative_to(root):
        raise ValueError("navigation metadata path leaves its dataset")
    payload = path.read_bytes()
    expected = entry.get("metadata_sha256")
    if require_digest and not expected:
        raise ValueError("navigation index requires a metadata digest for every episode")
    if expected is not None and hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("navigation episode metadata changed after indexing")
    meta = json.loads(payload)
    if any(meta.get(key) != entry[key] for key in ("task", "seed", "split")):
        raise ValueError("navigation episode identity differs from its index")
    if not isinstance(meta.get("instruction"), str) or not meta["instruction"].strip():
        raise ValueError("navigation episode requires a nonempty instruction")
    return meta
