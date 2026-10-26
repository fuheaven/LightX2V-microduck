"""Optional, actor-selected historical RGB for the two-image VLM interface.

The selector uses captured color components only. It never supplies a current
target position, task score or next goal. Old images retain their capture time.
"""
import base64
from dataclasses import dataclass
import io

from PIL import Image

from .navigation_hybrid import HybridRequest
from .navigation_visual_memory import CausalTargetKeyframeMemory, TargetKeyframe


VISUAL_MEMORY_PROTOCOL = "microduck_selected_color_history_v1"


@dataclass(frozen=True)
class VisualMissionRequest(HybridRequest):
    historical_target: TargetKeyframe | None = None


class SelectedTargetVisualMemory:
    def __init__(self):
        self.reset("")

    def reset(self, episode_id):
        self.episode_id = episode_id
        self.memories = {color: CausalTargetKeyframeMemory() for color in ("red", "blue", "yellow")}
        if episode_id:
            for color, memory in self.memories.items():
                memory.reset(episode_id, color)

    def append(self, observation):
        if (not self.episode_id or observation.episode_id != self.episode_id
                or observation.camera_tick != observation.tick):
            raise ValueError("visual memory requires the active episode's fresh captured image")
        for color, memory in self.memories.items():
            memory.append(observation.episode_id, color, observation.tick, observation.rgb)

    def snapshot(self, episode_id, source_tick, selected_goal):
        if episode_id != self.episode_id:
            raise ValueError("visual memory belongs to another episode")
        color = None if selected_goal is None else selected_goal["target"]
        if color is not None and color not in self.memories:
            raise ValueError("visual memory requires a model-selected color")
        frame = None if color is None else self.memories[color].snapshot(episode_id, source_tick)
        use = frame is not None and frame.capture_tick < source_tick
        receipt = dict(protocol=VISUAL_MEMORY_PROTOCOL, selected_color=color,
            capture_tick=None if frame is None else frame.capture_tick,
            age_s=None if frame is None else (source_tick - frame.capture_tick) / 50,
            historical_image_used=use,
            reason=("no_selected_goal" if color is None else "no_detected_candidate" if frame is None else
                    "current_image_has_candidate" if not use else "historical_color_candidate"),
            provenance="captured_rgb_color_component_not_current_position_or_success")
        if not use:
            return None, receipt
        frame.rgb.setflags(write=False)
        return frame, receipt


def with_historical_target(payload, request):
    """Replace only the older image; retain the exact current image and clocks."""
    receipt = request.context["visual_memory"]
    frame = request.historical_target
    if receipt["protocol"] != VISUAL_MEMORY_PROTOCOL or bool(frame is not None) != receipt["historical_image_used"]:
        raise ValueError("historical image differs from its prepared receipt")
    payload["metadata"]["visual_memory_protocol"] = VISUAL_MEMORY_PROTOCOL
    payload["messages"][0]["content"] += (
        " visual_memory may provide an older RGB image selected by a color-component detector for "
        "YOUR selected target. Its capture tick and age are explicit. The detector can mistake other "
        "colored objects for a ball. This image does not locate the target now or prove arrival. "
        "It replaces the before-action image only when the current image has no detected candidate. "
        "Use it as uncertain history, and acquire a new view when needed. Memory resets per subtask."
    )
    if frame is None:
        return payload
    obs = request.observation
    if (frame.episode_id != obs.episode_id or not 0 <= frame.capture_tick < obs.camera_tick
            or frame.capture_tick != receipt["capture_tick"]
            or frame.target_color != request.context["selected_subgoal"]["target"]):
        raise ValueError("historical image has the wrong session, time or selected target")
    content = payload["messages"][-1]["content"]
    current_label = f"Current observation; camera tick {obs.camera_tick}"
    positions = [i for i, item in enumerate(content) if item == dict(type="text", text=current_label)]
    if len(positions) != 1 or content[positions[0] + 1]["type"] != "image_url":
        raise ValueError("missing exact current camera image")
    at = positions[0]
    if request.previous is not None:
        expected = f"Before the most recent accepted action; camera tick {request.previous.camera_tick}"
        if at < 2 or content[at - 2] != dict(type="text", text=expected) or content[at - 1]["type"] != "image_url":
            raise ValueError("before-action image cannot be replaced unambiguously")
        del content[at - 2:at]
        at -= 2
    buffer = io.BytesIO()
    Image.fromarray(frame.rgb).save(buffer, format="JPEG")
    pair = [dict(type="text", text=(f"Historical {frame.target_color} color candidate; camera tick "
            f"{frame.capture_tick}; age {receipt['age_s']:.2f}s; not a current target location")),
            dict(type="image_url", image_url=dict(url="data:image/jpeg;base64," +
                                                base64.b64encode(buffer.getvalue()).decode()))]
    content[at:at] = pair
    if sum(item["type"] == "image_url" for item in content) != 2:
        raise ValueError("historical target must preserve the two-image interface")
    return payload
