ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9
FROM ${BASE_IMAGE}
ARG PYPI_INDEX=https://pypi.org/simple
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libegl1 libgl1 libglvnd0 libosmesa6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/microduck/fastwam/requirements.lock /opt/microduck/requirements.lock
RUN python -m pip install --no-cache-dir --index-url ${PYPI_INDEX} -r /opt/microduck/requirements.lock
COPY third_party/bam /opt/bam
COPY deploy/microduck/fastwam/verify_bam.py /opt/microduck/verify_bam.py
RUN python /opt/microduck/verify_bam.py /opt/bam
RUN python -m pip freeze --all > /opt/microduck/installed-packages.txt
# Keep this environment separate from the optional TinyVLA/ROS packages.
# The native FastWAM runtime is tested with Python 3.11 via PYTHONPATH.
ENV LIGHTX2V_IMPORT_MODE=robotics DTYPE=BF16 MUJOCO_GL=egl \
    PYTHONPATH=/workspace:/workspace/embodied/microduck/src:/opt/bam \
    HF_HOME=/data/cache/huggingface XDG_CACHE_HOME=/data/cache \
    TMPDIR=/data/tmp MICRODUCK_ASSET_ROOT=/data/upstream \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
WORKDIR /workspace
CMD ["python", "scripts/microduck/fastwam_handoff.py", "--help"]
