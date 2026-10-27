"""Minimal tool-safe SDK. Evaluation truth is deliberately not returned to a planner."""

import httpx


class Studio:
    def __init__(self, url, token=None):
        self.client = httpx.Client(base_url=url, headers={"Authorization": "Bearer " + token} if token else {}, timeout=30)

    def run(self, skill_id, version, *, category="ball", color="red", backend="fastwam"):
        response = self.client.post(f"/api/skills/{skill_id}/runs", json={"kind": "live", "mode": "realtime", "version": version, "policy": backend, "category": category, "color": color})
        response.raise_for_status()
        row = response.json()
        return {"run_id": row["id"], "status": row["status"]}

    def status(self, run_id):
        r = self.client.get(f"/api/runs/{run_id}", params={"tool": "true"})
        r.raise_for_status()
        return r.json()

    def cancel(self, run_id):
        r = self.client.post(f"/api/runs/{run_id}/cancel")
        r.raise_for_status()
        return {"run_id": run_id, "status": r.json()["status"], "cancel_requested": True}

    def close(self):
        self.client.close()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_skill",
            "description": "Run a registered static-object approach skill in simulation.",
            "parameters": {
                "type": "object",
                "properties": {"category": {"type": "string", "enum": ["ball", "box", "toy"]}, "color": {"type": "string", "enum": ["red", "blue", "yellow"]}},
                "required": ["category", "color"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_status",
            "description": "Read execution status, not evaluator truth.",
            "parameters": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"], "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_skill",
            "description": "Cancel and clear pending motion.",
            "parameters": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"], "additionalProperties": False},
        },
    },
]
