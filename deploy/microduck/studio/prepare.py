"""Create a private deployment env file without printing credentials."""

import argparse
import os
import secrets
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--workspace", type=Path, required=True)
p.add_argument("--data", type=Path, required=True)
p.add_argument("--env-file", type=Path, required=True)
args = p.parse_args()
if args.env_file.exists():
    raise SystemExit("Environment file already exists; edit it explicitly.")
for path in (args.workspace, args.data, args.data / "cache", args.data / "tmp", args.env_file.parent):
    path.mkdir(parents=True, exist_ok=True)
for path in (args.workspace, args.data):
    if any(c in str(path.resolve()) for c in ("\n", "\r", "$", '"')):
        raise SystemExit("Unsupported path characters for Compose env file.")
values = {
    "STUDIO_WORKSPACE_DIR": str(args.workspace.resolve()),
    "STUDIO_DATA_DIR": str(args.data.resolve()),
    "STUDIO_UID": str(os.getuid()),
    "STUDIO_GID": str(os.getgid()),
    "STUDIO_BIND": "127.0.0.1",
    "STUDIO_PORT": "8771",
    "STUDIO_GPU_ID": "0",
    "STUDIO_WORKER_IMAGE": "microduck-studio-worker:v1",
    "STUDIO_WORKER_TOKEN": secrets.token_urlsafe(32),
    "STUDIO_TOKEN": secrets.token_urlsafe(32),
}
fd = os.open(args.env_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as f:
    f.write("".join(f'{key}="{value}"\n' for key, value in values.items()))
print("Created private deployment configuration:", args.env_file)
