"""Minimal tool dispatcher; credentials and selected skill version stay on the server."""

import json
import os
from lightx2v_microduck.studio.sdk import Studio, TOOLS

client = Studio(os.environ["STUDIO_URL"], os.environ.get("STUDIO_TOKEN"))


def dispatch(name, arguments):
    """Pass TOOL definitions to your planner, then route its tool calls here."""
    args = json.loads(arguments) if isinstance(arguments, str) else arguments
    if name == "run_skill":
        if set(args) != {"category", "color"}:
            raise ValueError("unexpected parameters")
        return client.run(os.environ["STUDIO_SKILL_ID"], os.environ["STUDIO_VERSION_ID"], backend=os.environ.get("STUDIO_BACKEND", "lightweight"), **args)
    if name == "skill_status":
        return client.status(args["run_id"])
    if name == "cancel_skill":
        return client.cancel(args["run_id"])
    raise ValueError("unknown tool")


__all__ = ["TOOLS", "dispatch"]
