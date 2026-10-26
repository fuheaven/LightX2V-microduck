"""Pinned, actor-independent descriptions of measured locomotion tools."""

import hashlib
import json
from pathlib import Path

from .assets import load_lock


CARD_PATH = Path(__file__).parent / "resources/official-bam-navigation-tool-response-v1.json"
CARD_SHA256 = "2134d6eed4357e0fe1bded08d6b9f27c8891362a9785c0d25a5176ab22fdb7c5"


def load_tool_response(profile="official_bam"):
    raw = CARD_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CARD_SHA256:
        raise ValueError("locomotion tool description changed from its verified calibration")
    card = json.loads(raw)
    if profile != card["profile"]:
        raise ValueError("locomotion tool calibration requires its official_bam profile")
    root = Path(__file__).parent
    for filename, digest in card["source_sha256"].items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != digest:
            raise ValueError("locomotion runtime differs from the measured tool calibration")
    policies = {entry["path"]: entry["sha256"]
                for entry in load_lock()["sources"]["microduck-policies"]["files"]}
    if any(policies.get(name) != digest for name, digest in card["policy_files"].items()):
        raise ValueError("official locomotion actor differs from the measured tool calibration")
    return card
