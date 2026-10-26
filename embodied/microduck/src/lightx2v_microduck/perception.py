"""RGB-only detection and temporal association. No world state is accepted.

The simple tracker is an independent implementation; it does not contain code
from the unlicensed microduck-tracking repository. Track identity is uncertain
after occlusion and expiration, and is never replaced by a simulator object ID.
"""

from dataclasses import replace
import numpy as np
from PIL import Image

from .community import community_cache_root, load_catalog
from .assets import sha256_file
from .contracts import PolicyManifest, SkillIntent
from .pixel_policy import PixelRulePolicy, detect_colored_ball


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    overlap = np.maximum(0.0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(overlap))
    area = lambda box: float(np.prod(np.maximum(0.0, box[2:] - box[:2])))
    return intersection / max(area(a) + area(b) - intersection, 1e-9)


class ImageTracker:
    def __init__(self, *, max_age_ticks=25, min_iou=0.1, buffer_ratio=0.5):
        self.max_age_ticks, self.min_iou = max_age_ticks, min_iou
        self.buffer_ratio = buffer_ratio
        self.reset()

    def reset(self):
        self.tracks, self.next_id, self.last_tick = {}, 1, -1

    def update(self, detections, tick):
        if tick <= self.last_tick:
            return list(self.tracks.values())
        self.tracks = {k: v for k, v in self.tracks.items() if tick - v["seen_tick"] <= self.max_age_ticks}
        pairs = []
        for identity, track in self.tracks.items():
            shift = np.tile(track["velocity_px_per_tick"], 2) * (tick - track["seen_tick"])
            predicted = np.asarray(track["bbox"]) + shift
            for index, detection in enumerate(detections):
                if detection.get("class", "ball") == track.get("class", "ball"):

                    def expanded(box):
                        box = np.asarray(box, dtype=float)
                        padding = (box[2:] - box[:2]) * self.buffer_ratio
                        return box + np.concatenate((-padding, padding))

                    overlap = max(iou(predicted, detection["bbox"]), 0.75 * iou(expanded(predicted), expanded(detection["bbox"])))
                    pairs.append((overlap, identity, index))
        used_tracks, used_detections = set(), set()
        for overlap, identity, index in sorted(pairs, reverse=True):
            if overlap < self.min_iou or identity in used_tracks or index in used_detections:
                continue
            old, det = self.tracks[identity], dict(detections[index])
            center = lambda box: (np.asarray(box[:2]) + np.asarray(box[2:])) / 2
            velocity = (center(det["bbox"]) - center(old["bbox"])) / (tick - old["seen_tick"])
            self.tracks[identity] = {**det, "track_id": identity, "seen_tick": tick, "velocity_px_per_tick": velocity.tolist(), "hits": old["hits"] + 1}
            used_tracks.add(identity)
            used_detections.add(index)
        for index, detection in enumerate(detections):
            if index not in used_detections:
                self.tracks[self.next_id] = {**detection, "track_id": self.next_id, "seen_tick": tick, "velocity_px_per_tick": [0.0, 0.0], "hits": 1}
                self.next_id += 1
        self.last_tick = tick
        for track in self.tracks.values():
            track["age_ticks"] = tick - track["seen_tick"]
        return list(self.tracks.values())


class DuckDetector:
    """Pinned YOLO duck detector: letterbox, raw head decoding and NMS."""

    def __init__(self, asset_root=None, *, confidence=0.35, nms_iou=0.45):
        import onnxruntime as ort

        record = load_catalog()["duck-detector-v2"]
        path = community_cache_root(asset_root) / "duck-detector-v2" / "duck_detect.onnx"
        if not path.is_file() or sha256_file(path) != record["files"]["duck_detect.onnx"]["sha256"]:
            raise ValueError("Fetch and verify community:duck-detector-v2 before using RGB detection")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        if self.session.get_inputs()[0].shape != [1, 3, 320, 320] or self.session.get_outputs()[0].shape != [1, 5, 2100]:
            raise ValueError("duck detector graph contract mismatch")
        self.confidence, self.nms_iou, self.record = confidence, nms_iou, record

    def detect(self, rgb):
        height, width = rgb.shape[:2]
        scale = min(320 / width, 320 / height)
        size = (round(width * scale), round(height * scale))
        left, top = (320 - size[0]) // 2, (320 - size[1]) // 2
        canvas = Image.new("RGB", (320, 320), (114, 114, 114))
        canvas.paste(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR), (left, top))
        array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        rows = self.session.run(["output0"], {"images": array})[0][0].T
        if not np.isfinite(rows).all():
            raise FloatingPointError("nonfinite detector output")
        candidates = []
        for x, y, w, h, score in rows[rows[:, 4] >= self.confidence]:
            box = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
            box = (box - [left, top, left, top]) / scale
            box = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
            candidates.append({"bbox": box.tolist(), "score": float(score), "class": "duck"})
        kept = []
        for det in sorted(candidates, key=lambda d: d["score"], reverse=True):
            if all(iou(det["bbox"], other["bbox"]) <= self.nms_iou for other in kept):
                kept.append(det)
        return kept


class TrackedBallPolicy(PixelRulePolicy):
    def __init__(self, *, profile="official_bam"):
        self.tracker = ImageTracker()
        super().__init__(profile=profile, use_tof=False, target_color="red")
        self.manifest = replace(
            self.manifest,
            policy_id="rgb_tracked_ball",
            metadata={**self.manifest.metadata, "selection": "initial_leftmost_visible_red_then_image_track", "required_scene": "ball_tracking", "identity_after_expiration": "lost_no_reselection"},
        )

    def reset(self, episode_id=""):
        super().reset(episode_id)
        self.tracker.reset()
        self.target_id = None
        self.perception = {}

    def detect(self, observation, color):
        self.last_perception_rgb = observation.rgb
        detections = detect_colored_ball(observation.rgb, color, all_candidates=True)
        tick = observation.camera_tick if observation.camera_tick is not None else observation.tick
        tracks = self.tracker.update(detections, tick)
        current = [t for t in tracks if t["age_ticks"] == 0]
        if self.target_id is None and len(current) >= 2:
            self.target_id = min(current, key=lambda t: t["u"])["track_id"]
        selected = next((t for t in current if t["track_id"] == self.target_id), None)
        self.perception = {"frame_tick": tick, "tracks": tracks, "selected_track_id": self.target_id, "target_visible": selected is not None, "uses_privileged_state": False}
        return selected

    def predict(self, observation):
        if observation.tick < 50:
            self.last_diagnostics = {"phase": "settle_before_target_selection", "uses_privileged_state": False}
            return SkillIntent("stand", np.zeros(13, np.float32), observation.episode_id, observation.tick, 5)
        action = super().predict(observation)
        self.last_diagnostics["perception"] = self.perception
        return action


class DuckDetectionPolicy:
    """Stand/scan and detect the second duck; this is a perception experiment."""

    def __init__(self, asset_root=None, *, profile="official_bam"):
        self.detector, self.tracker = DuckDetector(asset_root), ImageTracker()
        self.manifest = PolicyManifest(
            "rgb_duck_detector_v2",
            "skill_intent",
            "rgb_proprio",
            profile,
            horizon=5,
            execution_prefix=5,
            artifacts={"checkpoint_sha256": self.detector.record["files"]["duck_detect.onnx"]["sha256"]},
            metadata={"required_scene": "duck_detection", "learned_perception": True, "not_navigation": True, "uses_language": False, "license_claims": self.detector.record["license"]},
        )
        self.reset("")

    def reset(self, episode_id):
        self.episode_id = episode_id
        self.tracker.reset()
        self.last_diagnostics = {}

    def predict(self, observation):
        if observation.rgb is None:
            raise ValueError("duck detection requires RGB")
        if observation.episode_id != self.episode_id:
            self.reset(observation.episode_id)
        tick = observation.camera_tick if observation.camera_tick is not None else observation.tick
        tracks = self.tracker.update(self.detector.detect(observation.rgb), tick)
        self.last_perception_rgb = observation.rgb
        self.last_diagnostics = {"perception": {"frame_tick": tick, "tracks": tracks, "uses_privileged_state": False}}
        return SkillIntent("stand", np.zeros(13, np.float32), observation.episode_id, observation.tick, 5)
