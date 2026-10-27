FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libegl1 libgl1 libglvnd0 libosmesa6 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY deploy/microduck/fastwam/requirements.lock /tmp/deploy/fastwam/requirements.lock
COPY deploy/microduck/studio/*.lock /tmp/deploy/studio/
RUN pip install --no-cache-dir -r /tmp/deploy/studio/worker.lock && pip freeze --all > /installed-packages.txt
COPY . .
RUN python deploy/microduck/fastwam/verify_bam.py third_party/bam && python deploy/microduck/studio/verify_source.py
ENV LIGHTX2V_IMPORT_MODE=robotics DTYPE=BF16 MUJOCO_GL=osmesa \
    PYTHONPATH=/workspace:/workspace/embodied/microduck/src:/workspace/third_party/bam:/workspace/third_party/fastwam/src \
    HF_HOME=/data/cache/huggingface XDG_CACHE_HOME=/data/cache TORCH_HOME=/data/cache/torch TMPDIR=/data/tmp \
    TORCHINDUCTOR_CACHE_DIR=/data/cache/inductor TRITON_CACHE_DIR=/data/cache/triton \
    MICRODUCK_ASSET_ROOT=/data/upstream STUDIO_MODEL_ROOT=/data OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    __EGL_VENDOR_LIBRARY_FILENAMES=/workspace/deploy/microduck/fastwam/10_nvidia.json
CMD ["python","-m","lightx2v_microduck.studio.worker","--help"]
