"""Configurable local/cloud VLM baseline using only measured permitted sensors."""

import base64
import io
import json
import os
import numpy as np
from PIL import Image
import httpx


class VLMRuntime:
    def __init__(self, kind):
        prefix = "STUDIO_VLM_LOCAL" if kind == "vlm_local" else "STUDIO_VLM_CLOUD"
        self.url = os.environ[prefix + "_URL"].rstrip("/") + "/chat/completions"
        self.model = os.environ[prefix + "_MODEL"]
        self.client = httpx.Client(timeout=30, headers={"Authorization": "Bearer " + os.environ.get(prefix + "_KEY", "")})
        self.usage = []

    def predict(self, frames, states, category, color, instruction):
        image = io.BytesIO()
        Image.fromarray(frames[-1]).save(image, format="JPEG")
        prompt = (
            "Control a simulated MicroDuck using official locomotion. Return JSON only: "
            '{"vx":0,"vy":0,"yaw":0,"head":0,"stop":false}. '
            "Bounds vx[-.15,.30],vy[-.15,.15],yaw[-1.1,1.1],head[-.45,.45]. "
            "Positive yaw turns left. RGB and measured ToF8x8 in metres are available. "
            "Find, avoid obstacles, approach then stop 0.25-0.40m from the target. "
            f"Instruction: {instruction}. ToF: {states[-1, 48:112].tolist()}. "
            f"ToF validity: {states[-1, 112:].astype(int).tolist()}. State48: {states[-1, :48].tolist()}"
        )
        response = self.client.post(
            self.url,
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image.getvalue()).decode()}}],
                    }
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()
        self.usage.append(payload.get("usage", {}))
        text = payload["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        value = json.loads(text)
        if set(value) != {"vx", "vy", "yaw", "head", "stop"} or type(value["stop"]) is not bool:
            raise ValueError("VLM response does not match the command contract")
        action = np.array([value["vx"], value["vy"], value["yaw"], value["head"], float(value["stop"])], np.float32)
        from ..navigation_action_contracts import action_contract

        action_contract().validate(action)
        return np.tile(action, (32, 1))
