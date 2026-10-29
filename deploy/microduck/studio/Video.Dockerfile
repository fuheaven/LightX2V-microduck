FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg fonts-dejavu-core libsndfile1 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY deploy/microduck/studio/*.lock /tmp/requirements/
RUN pip install --no-cache-dir -r /tmp/requirements/video.lock && pip check && pip freeze --all > /installed-packages.txt
COPY . .
ENV PYTHONPATH=/workspace:/workspace/embodied/microduck/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/data/cache/huggingface XDG_CACHE_HOME=/data/cache TMPDIR=/data/tmp
RUN python -c "from lightx2v_microduck.studio import video, worker; from diffusers import ModularPipeline; from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3VideoReference; from diffusers.utils.export_utils import encode_video; print('Studio video worker and H3 imports verified')"
CMD ["python", "-m", "lightx2v_microduck.studio.worker", "--help"]
