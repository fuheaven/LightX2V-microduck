"""Standalone native robot runner; the video DiT registry has another API."""

from lightx2v.models.networks.tinyvla.runtime import TinyVLARuntime


class TinyVLARunner:
    """Use artifact freshness unless a diagnostic explicitly overrides it.

    The runtime preserves the legacy 50-tick fallback for artifacts without an
    inference recipe and reports the effective cadence in every prediction's
    ``semantic_refresh_steps`` field, including explicit overrides.
    """

    def __init__(self, config):
        self.runtime = TinyVLARuntime.from_directory(
            config["model_path"],
            backbone_path=config.get("backbone_path"),
            device=config.get("device", "cpu"),
            precision=config.get("precision", "fp32"),
            semantic_refresh_steps=config.get("semantic_refresh_steps"),
            compile_expert=config.get("compile_expert", False),
            vision_backend=config.get("vision_backend", "fixed_eager"),
        )

    def predict(self, *args, **kwargs):
        return self.runtime.predict(*args, **kwargs)

    def reset(self, session_id=None):
        self.runtime.reset(session_id)
