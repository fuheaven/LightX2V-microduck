#!/usr/bin/env bash
# Independent UI + on-demand simulation, served directly on the chosen host port.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
: "${FASTWAM_DATA:?Set FASTWAM_DATA to an absolute writable data directory}"
: "${GPU_ID:?Set GPU_ID to one available GPU, or none for the official ONNX demo}"
[[ "$FASTWAM_DATA" = /* ]] || { echo 'FASTWAM_DATA must be absolute' >&2; exit 2; }
mkdir -p "$FASTWAM_DATA"/{models,cache,tmp,results,inputs,upstream}
gpu_args=(); server_args=(--enable-live)
if [[ "$GPU_ID" != none ]]; then
  gpu_args=(--gpus "device=$GPU_ID")
  server_args+=(--enable-fastwam)
fi
# NVIDIA 580.95.05 on the tested H200 stalls EGL readback alongside CUDA.
# Bounded OSMesa snapshot workers keep physics running without driver changes.
render=${MUJOCO_GL:-osmesa}
exec docker run --rm --init --name "${FASTWAM_WEB_NAME:-microduck-studio}" \
  "${gpu_args[@]}" --shm-size=8g --user "$(id -u):$(id -g)" \
  -p "${WEB_BIND:-127.0.0.1}:${WEB_PORT:-8770}:8770" \
  -e MUJOCO_GL="$render" -e PYOPENGL_PLATFORM="$render" -e HOME=/data/cache \
  -v "$repo_root:/workspace:ro" -v "$FASTWAM_DATA:/data" \
  "${FASTWAM_IMAGE:-lightx2v-microduck:fastwam-v1}" \
  python scripts/microduck/fastwam_visualization.py --host 0.0.0.0 --port 8770 \
  --data /data "${server_args[@]}"
