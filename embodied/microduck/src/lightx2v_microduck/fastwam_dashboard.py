"""FastWAM dashboard ingestion belongs to the Web server lifecycle, not training."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

SYNC_SECONDS = 3600
STALE_SECONDS = SYNC_SECONDS + 300

SSH = ['ssh', '-S', '/tmp/codex-microduck-training-ssh', '-p', '2041',
       '-oBatchMode=yes', '-oConnectTimeout=10', 'fuhaiwen@10.210.8.11', 'python3 -']


def directory():
    return Path(os.environ.get('MICRODUCK_FASTWAM_DASHBOARD_DIR',
                str(Path(__file__).resolve().parents[4] / 'artifacts/fastwam-monitor')))


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def synchronize(root=None):
    root = Path(root) if root else directory()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'sync.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        try:
            source = Path(__file__).with_name('fastwam_status_source.py').read_bytes()
            result = subprocess.run(SSH, input=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=40, check=True)
            data = json.loads(result.stdout)
            if data['schema'] != 'microduck_fastwam_dashboard_v1':
                raise ValueError('unexpected schema')
            write(root / 'snapshot.json', data)
            write(root / 'connection.json', {'ok': True, 'checked_unix': time.time()})
        except Exception as exc:
            # Do not expose SSH stderr, environment details or credentials in the public page.
            write(root / 'connection.json', {'ok': False, 'checked_unix': time.time(),
                                           'error_type': type(exc).__name__})


def next_sync_delay(root=None, now=None):
    """Preserve the hourly cadence across a Web-process restart."""
    root = Path(root) if root else directory()
    now = time.time() if now is None else now
    try:
        checked = json.loads((root / 'connection.json').read_text())['checked_unix']
        if type(checked) not in (int, float) or not math.isfinite(checked):
            return 0
        return min(SYNC_SECONDS, max(0, SYNC_SECONDS - (now - checked)))
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def start_sync():
    stop = threading.Event()
    def worker():
        if stop.wait(next_sync_delay()):
            return
        while not stop.is_set():
            synchronize()
            stop.wait(SYNC_SECONDS)
    threading.Thread(target=worker, name='fastwam-read-only-sync', daemon=True).start()
    return stop


def dashboard(root=None, now=None):
    root = Path(root) if root else directory()
    now = time.time() if now is None else now
    try:
        data = json.loads((root / 'snapshot.json').read_text())
    except FileNotFoundError:
        return {'available': False, 'freshness': {'state': 'unavailable'}, 'refresh_seconds': SYNC_SECONDS}
    try:
        connection = json.loads((root / 'connection.json').read_text())
    except FileNotFoundError:
        connection = {}
    age = max(0, now - data['observed_unix'])
    data['available'] = True
    data['freshness'] = {'state': 'disconnected' if connection.get('ok') is False else 'stale' if age > STALE_SECONDS else 'fresh',
                         'age_seconds': age, 'observed_unix': data['observed_unix']}
    data['refresh_seconds'] = SYNC_SECONDS
    # Never infer an active allocation from a report's status alone.
    running = [c for c in data['active'] if c['running'] and c['pid_live']]
    active_names = {c['name'] for c in running}
    for run in data['runs']:
        run['live'] = 'microduck-fastwam-' + run['id'] in active_names
        run['execution'] = next((c for c in data.get('completed_containers', [])
                                if c['name'] == 'microduck-fastwam-' + run['id']), None)
    data['current_run_id'] = next((r['id'] for r in reversed(data['runs']) if r['live']),
                                  data['runs'][-1]['id'] if data['runs'] else None)
    data['current_job'] = running[0] if running else None
    data['current_evaluation'] = next((e for e in data.get('pending_evaluations', [])
                                     if 'microduck-fastwam-' + e['id'] in active_names), None)
    selection = data.get('model_selection') or {}
    data['retained'] = selection.get('retained')
    data['latest_candidate'] = selection.get('latest')
    # Legacy snapshots lack a verified current selection. Do not relabel their
    # historical best result or most recent report as the current model.
    data['retained_checkpoint'] = (data['retained'] or {}).get('checkpoint_sha256')
    # Large internal queue argv is not a product-facing datum.
    q = data.pop('queue', {})
    data['queue'] = {k: q.get(k) for k in ('status', 'stage', 'container', 'completed_jobs',
                                         'total_jobs', 'process_state', 'observed_unix')}
    return data
