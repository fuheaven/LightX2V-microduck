#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
: "${FASTWAM_DATA:?Set FASTWAM_DATA to an absolute writable data directory}"
: "${GPU_ID:?Set GPU_ID to a free GPU index/UUID, or none for CPU asset/input commands}"
[[ "$FASTWAM_DATA" = /* ]] || { echo 'FASTWAM_DATA must be absolute' >&2; exit 2; }
mkdir -p "$FASTWAM_DATA"/{models,cache,tmp,results,inputs,upstream}
gpu_args=()
if [[ "$GPU_ID" != none ]]; then gpu_args=(--gpus "device=$GPU_ID"); fi
network_args=()
if [[ -n "${FASTWAM_NETWORK:-}" ]]; then network_args=(--network "$FASTWAM_NETWORK"); fi
proxy_args=()
for proxy_name in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy; do
  if [[ -n "${!proxy_name:-}" ]]; then proxy_args+=(-e "$proxy_name"); fi
done
exec docker run --rm --init "${gpu_args[@]}" "${network_args[@]}" "${proxy_args[@]}" --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  -e MUJOCO_GL="${MUJOCO_GL:-egl}" -e PYOPENGL_PLATFORM="${MUJOCO_GL:-egl}" -e HOME=/data/cache \
  -v "$repo_root:/workspace:ro" -v "$FASTWAM_DATA:/data" \
  "${FASTWAM_IMAGE:-lightx2v-microduck:fastwam-v1}" "$@"
