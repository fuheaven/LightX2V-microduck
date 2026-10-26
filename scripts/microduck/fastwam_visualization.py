"""Standalone FastWAM gallery; optionally enable a local or remote live runner."""
import argparse
from pathlib import Path
import signal
import sys
import threading

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "embodied/microduck/src"))


def main():
    from lightx2v_microduck.visualization import make_server
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--data", type=Path, default=REPO / "artifacts/visualization")
    parser.add_argument("--enable-live", action="store_true", help="Allow local MuJoCo child processes")
    parser.add_argument("--enable-fastwam", action="store_true", help="Enable local GPU inference; implies --enable-live")
    parser.add_argument("--worker-url", help="Optional trusted live runner origin; browser still uses this service")
    args = parser.parse_args()
    if args.worker_url and (args.enable_live or args.enable_fastwam):
        parser.error("choose either a local runner or a remote worker-url")
    server = make_server(args.host, args.port, args.data,
                         enable_live=args.enable_live or args.enable_fastwam,
                         enable_fastwam=args.enable_fastwam, worker_url=args.worker_url)
    def shutdown(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(f"MicroDuck Studio: http://{args.host}:{server.server_port} (no training)", flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        server.jobs.close()
        server.server_close()


if __name__ == "__main__":
    main()
