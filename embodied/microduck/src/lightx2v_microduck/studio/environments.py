"""Portable data-only environments; importing never inherits skill evidence."""

from .assets import asset_identity, catalog
from .scene import validate_scene
from .schema import digest


def export_environment(scene):
    scene = validate_scene(scene)
    ids = {o["asset_id"] for o in scene["objects"] if o.get("asset_id")}
    body = {
        "schema": "studio_environment_v1",
        "units": "m",
        "up_axis": "z",
        "yaw_units": "degrees",
        "scene": scene,
        "assets": [a for a in catalog()["assets"] if a["id"] in ids],
        "asset_sha256": asset_identity(scene),
        "scope": "simulation_only",
    }
    return {**body, "sha256": digest(body)}


def import_environment(value):
    if set(value) != {"schema", "units", "up_axis", "yaw_units", "scene", "assets", "asset_sha256", "scope", "sha256"}:
        raise ValueError("unsupported environment package fields")
    if value["sha256"] != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("environment checksum mismatch")
    expected = export_environment(value["scene"])
    if expected != value:
        raise ValueError("environment assets or coordinate contract do not match the installed registry")
    return expected["scene"]
