"""Original, data-only furniture recipes shared by the editor and MuJoCo.

Dimensions and positions are metres, Z up. Collision boxes are the rendered
boxes themselves; there is no second physics engine or executable asset code.
"""

from .schema import digest

MATERIALS = {
    "wood": {"name": "浅木", "rgba": [0.63, 0.46, 0.31, 1]},
    "fabric": {"name": "亚麻", "rgba": [0.66, 0.70, 0.63, 1]},
    "slate": {"name": "石墨", "rgba": [0.30, 0.38, 0.42, 1]},
}

# Each part: full size and centre, expressed in fractions of the asset bounds.
RECIPES = {
    "table": {
        "name": "桌子",
        "dimensions": [0.9, 0.55, 0.65],
        "material": "wood",
        "parts": [
            [[1, 1, 0.08], [0, 0, 0.96]],
            *[[[0.08, 0.10, 0.92], [x, y, 0.46]] for x in (-0.43, 0.43) for y in (-0.42, 0.42)],
        ],
    },
    "chair": {
        "name": "椅子",
        "dimensions": [0.42, 0.42, 0.72],
        "material": "wood",
        "parts": [
            [[1, 1, 0.08], [0, 0, 0.53]],
            [[1, 0.10, 0.43], [0, 0.45, 0.785]],
            *[[[0.10, 0.10, 0.49], [x, y, 0.245]] for x in (-0.43, 0.43) for y in (-0.43, 0.43)],
        ],
    },
    "sofa": {
        "name": "沙发",
        "dimensions": [1.3, 0.6, 0.65],
        "material": "fabric",
        "parts": [
            [[1, 1, 0.45], [0, 0, 0.225]],
            [[1, 0.20, 0.55], [0, 0.40, 0.725]],
            *[[[0.12, 0.8, 0.30], [x, -0.10, 0.60]] for x in (-0.44, 0.44)],
        ],
    },
    "cabinet": {
        "name": "柜子",
        "dimensions": [0.75, 0.35, 0.9],
        "material": "wood",
        "parts": [
            [[1, 1, 1], [0, 0, 0.5]],
        ],
    },
    "partition": {
        "name": "隔断",
        "dimensions": [1.0, 0.12, 0.8],
        "material": "slate",
        "parts": [
            [[1, 1, 1], [0, 0, 0.5]],
        ],
    },
}


def catalog():
    return {
        "schema": "studio_assets_v1",
        "units": "m",
        "up_axis": "z",
        "materials": MATERIALS,
        "assets": [{"id": key, "version": "1.0.0", **value, "sha256": digest(value), "source": "LightX2V-microduck original procedural geometry", "static": True} for key, value in RECIPES.items()],
    }


def parts(obj):
    recipe = RECIPES[obj["asset_id"]]
    if obj.get("asset_version") != "1.0.0":
        raise ValueError("unsupported furniture version")
    dims = [obj["width"], obj["depth"], obj["height"]]
    return [{"size": [v * d for v, d in zip(size, dims)], "pos": [v * d for v, d in zip(pos, dims)], "rgba": MATERIALS[obj["material"]]["rgba"]} for size, pos in recipe["parts"]]


def asset_identity(scene):
    return digest(
        {
            "recipes": {o["asset_id"]: RECIPES[o["asset_id"]] for o in scene["objects"] if o.get("asset_id")},
            "materials": {o["material"]: MATERIALS[o["material"]] for o in scene["objects"] if o.get("asset_id")},
        }
    )
