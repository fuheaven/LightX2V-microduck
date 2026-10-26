#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
: "${FASTWAM_DATA:?Set FASTWAM_DATA to an absolute writable data directory}"
: "${GPU_ID:?Set GPU_ID to a free GPU index/UUID, or none for CPU asset/input commands}"
[[ "$FASTWAM_DATA" = /* ]] || { echo 'FASTWAM_DATA must be absolute' >&2; exit 2; }
mkdir -p "$FASTWAM_DATA"/{models,cache,tmp,results,inputs,upstream}
gpu_args=()
if [[ "$GPU_ID" != none ]]; then gpu_args=(--gpus "device=$GPU_ID"); fi
exec docker run --rm --init "${gpu_args[@]}" --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  -e MUJOCO_GL="${MUJOCO_GL:-egl}" -e PYOPENGL_PLATFORM="${MUJOCO_GL:-egl}" -e HOME=/data/cache \
  -v "$repo_root:/workspace:ro" -v "$FASTWAM_DATA:/data" \
  "${FASTWAM_IMAGE:-lightx2v-microduck:fastwam-v1}" "$@"
