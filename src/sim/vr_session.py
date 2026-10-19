"""Dependency-light shared VR recipe and per-session source/task identity."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys


TASKS = ('stack_cube', 'blind_peg_insertion')
VIEW_PROFILES = {'table_front': 'table_front', 'robot_head': 'blindpeg_head', 'table_overview': 'table_overview'}
ENVIRONMENT = {
    'AIRSEAI_XR_VIEW_PROFILE': 'table_front',
    'AIRSEAI_RECORDING_DURABLE': '1',
    'AIRSEAI_XR_B_FINISH_ONLY': '1',
    'AIRSEAI_XR_FORCE_VIEWPORT': '1',
    'PYTHONNOUSERSITE': '1',
    'PYTHONUNBUFFERED': '1',
}


def _source_identity(repo):
    """Hash runtime source bytes, including local candidate changes."""
    paths = [repo/'pyproject.toml']
    for folder in ('src', 'benchmark', 'scripts'):
        paths.extend(p for p in (repo/folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix in ('.py', '.sh', '.toml'))
    files = {p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(set(paths)) if p.is_file()}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'sha256': digest, 'files': files}


def make_plan(task, session_dir, *, episodes=5, seed=0, python=None, repo_root=None, view_profile='table_front', media_bin=None):
    """Build a preview without touching disk or importing Isaac/torch/OpenXR."""
    if task not in TASKS:
        raise ValueError('task must be stack_cube or blind_peg_insertion')
    if type(episodes) is not int or episodes < 1 or type(seed) is not int or seed < 0:
        raise ValueError('episodes must be positive and seed nonnegative integers')
    if view_profile not in VIEW_PROFILES:
        raise ValueError('Unknown VR view profile')
    repo = Path(repo_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    session = Path(session_dir).expanduser().resolve()
    if session.exists():
        raise FileExistsError(f'Session already exists: {session}')
    entry = repo/'benchmark/teleop/run_teleop.py'
    if not entry.is_file():
        raise ValueError(f'Native VR entry missing: {entry}')
    interpreter = str(Path(python or sys.executable).expanduser().resolve())
    if not Path(interpreter).is_file():
        raise ValueError(f'Python interpreter missing: {interpreter}')
    environment = dict(ENVIRONMENT)
    path_parts = []
    if media_bin is not None:
        media_path = Path(media_bin).expanduser().resolve()
        if not media_path.is_dir():
            raise ValueError(f'media bin directory missing: {media_path}')
        path_parts.append(str(media_path))
    if os.environ.get('PATH'):
        path_parts.append(os.environ['PATH'])
    # Keep activated media wrappers ahead of raw binaries beside the interpreter.
    path_parts.append(str(Path(interpreter).parent))
    environment['PATH'] = os.pathsep.join(path_parts)
    environment['AIRSEAI_XR_VIEW_PROFILE'] = VIEW_PROFILES[view_profile]
    placement = None
    if view_profile == 'table_front':
        environment['AIRSEAI_XR_VIEW_PLACEMENT_FILE'] = str(session/'view_placement.json')
        placement = {'schema_version': 1, 'request_id': 'initial-table-front',
                     'world_down_m': 0.30, 'world_yaw_deg': 180.0,
                     'world_toward_viewer_m': 0.45,
                     'robot_pivot': [0.0, 0.0, 0.0]}
    environment['PYTHONPATH'] = str(repo/'src') + (
        os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
    command = [interpreter, '-u', str(entry), '--task', task, '--teleop_device', 'vr',
        '--device', 'cuda:0', '--seed', str(seed), '--num_envs', '1', '--episodes', str(episodes),
        '--horizon_s', '180', '--dataset', str(session/'dataset'), '--out', str(session/'report'),
        '--keep-unclean', '--vr-joint-speed-deg-s', '60', '--vr-position-gain', '1',
        '--pos_speed', '0.20', '--rot_speed', '0.9', '--jaw_speed', '0.02',
        '--write_mode', 'background', '--write_queue_size', '1', '--episode_write_workers', '1',
        '--no-previews', '--headless']
    return {'schema_version': 1, 'task': task, 'view_profile': view_profile, 'view_placement': placement,
        'session_dir': str(session), 'repo_root': str(repo),
        'command': command, 'environment': environment, 'source': _source_identity(repo),
        'report_path': str(session/'report'/f'{task}_teleop.json'),
        'initial_states': 'task-generated reset; no bundled 22-joint bank selected',
        'expected_robot_joints': None, 'physics': 'task-native; blind-peg VR override stays task-specific',
        'status': 'preview', 'task_success': None}


def _write_manifest(path, manifest):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _media_preflight(environment):
    """Check actual selected recording tools before creating any session output."""
    tools = {}
    for name in ('ffmpeg', 'ffprobe'):
        found = shutil.which(name, path=environment.get('PATH', ''))
        if found is None:
            raise ValueError(f'{name} missing from recording PATH; activate the runtime or select --media-bin')
        path = Path(found).absolute()
        try:
            result = subprocess.run([str(path), '-version'], env=environment, text=True,
                                    errors='replace', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
            output = (result.stdout or '').strip()
            version = output.splitlines()[0] if output else ''
            if result.returncode != 0 or not version.startswith(name + ' version'):
                raise ValueError(f'{name} -version failed ({result.returncode}): {output[:500]}')
            hasher = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ValueError(f'{name} -version preflight failed: {error}') from error
        tools[name] = {'path': str(path), 'version': version, 'sha256': digest}
    return tools


def run_session(plan):
    """Execute in the selected source tree; never reuse an existing session."""
    repo, session = Path(plan['repo_root']), Path(plan['session_dir'])
    if _source_identity(repo) != plan['source']:
        raise ValueError('Candidate source changed since preview; build a fresh plan')
    environment = dict(os.environ)
    environment.update(plan['environment'])
    media_tools = _media_preflight(environment)
    session.mkdir(parents=True, exist_ok=False)
    manifest = dict(plan, status='running', media_tools=media_tools,
                    started_utc=datetime.now(timezone.utc).isoformat())
    path = session/'session.json'
    _write_manifest(path, manifest)
    if plan.get('view_placement') is not None:
        _write_manifest(session/'view_placement.json', plan['view_placement'])
    child = None
    reaped = False
    launcher_returncode = None
    try:
        # POSIX console interrupts reach the launcher only; forward once to
        # this new session's process group. Windows shares the existing console
        # group, which already delivers Ctrl-C to the native collector. Sending
        # SIGINT with Popen.send_signal is unsupported there and can terminate it.
        options = {} if sys.platform == 'win32' else {'start_new_session': True}
        child = subprocess.Popen(plan['command'], cwd=str(repo), env=environment, **options)
        manifest['child_pid'] = child.pid
        _write_manifest(path, manifest)
        while True:
            try:
                returncode = child.wait()
                reaped = True
                break
            except KeyboardInterrupt:
                manifest['interrupt_count'] = manifest.get('interrupt_count', 0) + 1
                manifest['status'] = 'stopping'
                manifest.setdefault('interrupt_requested_utc', datetime.now(timezone.utc).isoformat())
                _write_manifest(path, manifest)
                if manifest['interrupt_count'] == 1 and sys.platform != 'win32':
                    try:
                        os.killpg(child.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass  # Child exited between wait and signal forwarding.
                    except OSError as error:
                        manifest['interrupt_forward_error'] = str(error)
                        _write_manifest(path, manifest)
                # Keep waiting through repeated interrupts. Native collector
                # owns stop, writer drain, merge, recovery and report cleanup.
        manifest['returncode'] = returncode
        manifest['status'] = 'failed' if returncode else 'invalid_report'
        if returncode == 0:
            try:
                report = json.loads(Path(plan['report_path']).read_text('utf-8'))
                if not isinstance(report, dict) or report.get('task') != plan['task']:
                    raise ValueError('Report task differs from session identity')
                if report.get('status') != 'complete':
                    raise ValueError('Collector did not report complete')
                manifest['status'] = 'complete'
            except (OSError, ValueError) as error:
                manifest['report_error'] = str(error)
        manifest['report_status'] = manifest['status']
        if manifest.get('interrupt_count'):
            manifest['status'] = 'interrupted'
            launcher_returncode = 130
    except Exception as error:
        # Unexpected supervision failure must not masquerade as a running or
        # completed session. Do not kill an un-reaped collector on this path.
        manifest['status'] = 'failed' if child is None else 'supervision_failed'
        if child is not None and not reaped:
            manifest['process_state'] = 'unknown'
        manifest['error'] = str(error)
        manifest.setdefault('returncode', None)
        launcher_returncode = 1
    finally:
        if child is None or reaped:
            manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        if launcher_returncode is None:
            launcher_returncode = manifest['returncode'] or (0 if manifest['status'] == 'complete' else 2)
        manifest['launcher_returncode'] = launcher_returncode
        _write_manifest(path, manifest)
    return launcher_returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=TASKS)
    parser.add_argument('--session-dir', required=True)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--media-bin', help='explicit directory containing recording ffmpeg/ffprobe wrappers')
    parser.add_argument('--view-profile', choices=tuple(VIEW_PROFILES), default='table_front')
    parser.add_argument('--execute', action='store_true', help='start native VR; default only prints a preview')
    args = parser.parse_args(argv)
    try:
        plan = make_plan(args.task, args.session_dir, episodes=args.episodes, seed=args.seed,
                         python=args.python, view_profile=args.view_profile, media_bin=args.media_bin)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.execute:
        try:
            return run_session(plan)
        except ValueError as error:
            parser.error(str(error))
    # ASCII JSON also round-trips inherited PATH entries across Windows console encodings.
    print(json.dumps(plan, ensure_ascii=True, indent=2))
    return 0
