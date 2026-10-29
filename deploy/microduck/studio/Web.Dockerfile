FROM node:22.22.1-bookworm-slim AS frontend
WORKDIR /frontend
COPY apps/microduck-studio/package*.json ./
RUN npm ci --no-audit --no-fund
COPY apps/microduck-studio ./
RUN npm run build

FROM python:3.12.13-slim-bookworm
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
COPY deploy/microduck/studio/web.lock /tmp/web.lock
RUN pip install --no-cache-dir -r /tmp/web.lock && pip freeze --all > /installed-packages.txt
COPY . .
COPY --from=frontend /frontend/dist /workspace/apps/microduck-studio/dist
ENV PYTHONPATH=/workspace:/workspace/embodied/microduck/src PYTHONUNBUFFERED=1
EXPOSE 8771
CMD ["python","-m","lightx2v_microduck.studio.api","--host","0.0.0.0","--port","8771","--workspace","/workspace-artifacts"]
