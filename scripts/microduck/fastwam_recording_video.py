"""Export recorded head RGB on its original 50Hz physics timeline (no rerender)."""
import argparse
from pathlib import Path

import numpy as np


def frame_indices(camera_ticks, end_tick, fps=25):
    ticks = np.asarray(camera_ticks)
    if (ticks.ndim != 1 or not len(ticks) or ticks[0] < 0
            or not np.issubdtype(ticks.dtype, np.integer) or np.any(np.diff(ticks) <= 0)
            or end_tick < ticks[-1] or fps <= 0):
        raise ValueError("Expected strictly increasing recorded camera ticks")
    presentation = np.arange(ticks[0], end_tick + 1, 50 / fps)
    return np.searchsorted(ticks, presentation, side="right") - 1


def main():
    import imageio.v2 as imageio
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.recording, allow_pickle=False) as data:
        rgb, ticks = data["rgb"], data["camera_ticks"]
        end = int(data["physics_ticks"][-1]) if "physics_ticks" in data else int(ticks[-1])
        if rgb.dtype != np.uint8 or rgb.shape != (len(ticks), 256, 256, 3):
            raise ValueError("Expected the recorded 256px head camera, without rescaling")
        indices = frame_indices(ticks, end)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(args.output, fps=25, codec="libx264", quality=8) as video:
            for index in indices:
                video.append_data(rgb[index])
    print(f"Recorded head-camera replay: {args.output}; read episode.json for task scores")


if __name__ == "__main__":
    main()
