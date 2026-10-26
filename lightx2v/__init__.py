__version__ = "0.1.0"
__author__ = "LightX2V Contributors"
__license__ = "Apache 2.0"

import os as _os

# Robotics runners own their torch device and do not need video kernels, a
# distributed process group or CUDA initialization merely to import a model.
# Keep the existing video initialization unchanged for all existing callers.
_import_mode = _os.environ.get("LIGHTX2V_IMPORT_MODE", "full")
if _import_mode == "full":
    import lightx2v_platform.set_ai_device
    from lightx2v import common, models, utils
    from lightx2v.pipeline import LightX2VPipeline
elif _import_mode != "robotics":
    raise ValueError("LIGHTX2V_IMPORT_MODE must be 'full' or 'robotics'")

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "models",
    "common",
    "utils",
    "LightX2VPipeline",
]

if _import_mode == "robotics":
    __all__ = ["__version__", "__author__", "__license__"]
