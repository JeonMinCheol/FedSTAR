"""Launch the six FedRG model families across idle GPUs, with separate results."""

import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parent
FAMILIES = tuple(f'HtFE-img-{n}' for n in (2, 3, 4, 5, 8, 9))


def total_defaults():
    values = dict(dataset='Cifar100', t='3', global_rounds='500', batch_size='128',
                  join_ratio='1', num_clients='20', num_classes='100', local_epoch='1')
    for line in (ROOT / 'total.sh').read_text().splitlines():
        match = re.match(r'^\s*([a-z_]+)=(.*)$', line)
        if match and match[1] in values:
            tokens = shlex.split(match[2], comments=True)
            if len(tokens) != 1 or '$' in tokens[0]:
                raise ValueError(f'Cannot read literal setting from total.sh: {match[1]}')
            values[match[1]] = tokens[0]
    return values


def query_gpus():
    command = ['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits']
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError('GPU discovery failed; no experiments started. Run this launcher in the '
                           'host terminal with GPU device access.\n' + (result.stderr or result.stdout).strip())
    gpus = [dict(index=int(row[0]), uuid=row[1].strip(), free_mb=int(row[2]))
            for row in csv.reader(io.StringIO(result.stdout)) if row]
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
                          capture_output=True, text=True, timeout=15)
    if apps.returncode:
        raise RuntimeError('Cannot inspect existing GPU jobs; no experiments started')
    busy = {row[0].strip() for row in csv.reader(io.StringIO(apps.stdout)) if len(row) >= 2}
    for gpu in gpus:
        gpu['busy'] = gpu['uuid'] in busy
    return sorted(gpus, key=lambda gpu: gpu['index'])


def live_families(dataset):
    """Inspect only this user's Python training processes in this repository."""
    found = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid() or (proc / 'cwd').resolve() != ROOT:
                continue
            tokens = (proc / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            if not tokens or 'python' not in Path(tokens[0]).name or not any(
                    Path(token).name == 'main.py' for token in tokens[1:]):
                continue
            def argument(*names):
                value = None
                for index, token in enumerate(tokens):
                    if token in names and index + 1 < len(tokens):
                        value = tokens[index + 1]
                    for name in names:
                        if token.startswith(name + '='):
                            value = token[len(name) + 1:]
                return value
            if argument('-algo', '--algorithm') == 'FedRG' and argument('-data', '--dataset') == dataset:
                model = argument('-m', '--model_family')
                if model in FAMILIES:
                    found[model] = int(proc.name)
        except (OSError, ValueError, UnicodeError):
            continue
    return found


def distribute(gpus, families):
    if not gpus:
        raise ValueError('No idle GPU with at least 3 GiB free is available')
    groups = {gpu['index']: [] for gpu in gpus}
    if len(gpus) == 5:
        # Two smaller configurations share one device sequentially; large 8/9
        # configurations each have their own device for the entire experiment.
        layout = ((FAMILIES[0], FAMILIES[1]), (FAMILIES[2],), (FAMILIES[3],),
                  (FAMILIES[4],), (FAMILIES[5],))
        for gpu, models in zip(gpus, layout):
            groups[gpu['index']] = [model for model in models if model in families]
    elif len(gpus) >= len(FAMILIES):
        for gpu, family in zip(gpus, FAMILIES):
            if family in families:
                groups[gpu['index']].append(family)
    else:
        weights = dict(zip(FAMILIES, (2, 3, 3, 5, 6, 6)))
        loads = {gpu['index']: 0 for gpu in gpus}
        for family in sorted(families, key=lambda name: weights[name], reverse=True):
            gpu = min(gpus, key=lambda item: (loads[item['index']], -item['free_mb'], item['index']))
            groups[gpu['index']].append(family)
            loads[gpu['index']] += weights[family]
    return [{**gpu, 'families': groups[gpu['index']]} for gpu in gpus if groups[gpu['index']]]


def training_command(settings, family, gpu_uuid, batch):
    return [settings['python'], '-u', str(ROOT / 'main.py'), '-dev', 'cuda', '-did', gpu_uuid,
            '-algo', 'FedRG', '-data', settings['dataset'], '-m', family, '-fd', '512',
            '-t', str(settings['times']), '-gr', str(settings['global_rounds']),
            '-lbs', str(settings['batch_size']), '-ls', str(settings['local_epochs']),
            '-nc', str(settings['num_clients']), '-ncl', str(settings['num_classes']),
            '-jr', str(settings['join_ratio']), '-lr', '0.01', '-ab', '1', '-eg', '10',
            '-slr', '0.01', '--rg_graph_target', 'uniform', '--node_dim', '40',
            '--adjalpha', '3', '--gc_epoch', '10', '--gnn_learning_rate', '0.001',
            '--gnn_weight_decay', '0.0001', '--max_parallel_clients', '1',
            '--keep_checkpoints', 'false', '-go', f'{batch}_{family}']


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def worker(manifest_path, gpu_index):
    manifest = json.loads(manifest_path.read_text())
    group = next(group for group in manifest['groups'] if group['index'] == gpu_index)
    status_path = manifest_path.parent / f'gpu-{gpu_index}.json'
    jobs = [{'family': family, 'status': 'queued'} for family in group['families']]
    failures = 0
    for job in jobs:
        # Recheck before each queued experiment in case a manual run was started.
        existing = live_families(manifest['settings']['dataset'])
        if job['family'] in existing:
            job.update(status='already_running', pid=existing[job['family']])
            write_json(status_path, jobs)
            continue
        log_path = manifest_path.parent / f'{job["family"]}.out'
        command = training_command(manifest['settings'], job['family'], group['uuid'], manifest['batch'])
        environment = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=group['uuid'])
        # Cap independent CPU thread pools so five concurrent processes do not
        # create hundreds of OpenMP/MKL workers on the host.
        environment.setdefault('OMP_NUM_THREADS', '4')
        environment.setdefault('MKL_NUM_THREADS', '4')
        with log_path.open('x') as log:
            log.write('Command: ' + shlex.join(command) + '\n')
            log.flush()
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            job.update(status='running', pid=process.pid, gpu=gpu_index,
                       gpu_uuid=group['uuid'], log=str(log_path))
            write_json(status_path, jobs)
            print(f'Started {job["family"]}: GPU {gpu_index}, PID {process.pid}, log {log_path}', flush=True)
            returncode = process.wait()
        job.update(status='finished' if returncode == 0 else 'failed', exit_code=returncode)
        failures += returncode != 0
        write_json(status_path, jobs)
        print(f'{job["family"]}: {job["status"]} (exit {returncode})', flush=True)
    return int(failures > 0)


def supervise(manifest_path, lock_fd):
    # The inherited lock prevents another sweep launcher from starting duplicate
    # queued jobs during gaps between experiments/repetitions.
    os.fstat(lock_fd)
    manifest = json.loads(manifest_path.read_text())
    children = []
    for group in manifest['groups']:
        command = [manifest['settings']['python'], '-u', str(Path(__file__).resolve()),
                   '--worker', str(manifest_path), '--gpu-index', str(group['index'])]
        children.append(subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL))
    codes = [child.wait() for child in children]
    write_json(manifest_path.parent / 'finished.json', {'worker_exit_codes': codes, 'finished_at': time.time()})
    print('All FedRG GPU queues finished.', flush=True)
    return int(any(codes))


def launch(args):
    settings = {key: getattr(args, key) for key in ('python', 'dataset', 'times', 'global_rounds',
                'batch_size', 'local_epochs', 'num_clients', 'num_classes', 'join_ratio')}
    dataset = ROOT / 'dataset' / args.dataset
    config = json.loads((dataset / 'config.json').read_text())
    if config['num_clients'] != args.num_clients or config['num_classes'] != args.num_classes:
        raise ValueError('Requested client/class counts do not match the dataset config')
    existing = live_families(args.dataset)
    families = [family for family in FAMILIES if family not in existing]
    for family, pid in existing.items():
        print(f'Already running: {family}, PID {pid}; will not start a duplicate')
    if not families:
        return 0
    gpus = query_gpus()
    if args.gpus:
        selected = {int(value) for value in args.gpus.split(',')}
        if not selected <= {gpu['index'] for gpu in gpus}:
            raise ValueError('A requested GPU index was not found by nvidia-smi')
        gpus = [gpu for gpu in gpus if gpu['index'] in selected]
    # Do not put new experiments on a GPU already running another compute job.
    gpus = [gpu for gpu in gpus if not gpu['busy'] and gpu['free_mb'] >= 3072]
    groups = distribute(gpus, families)
    for group in groups:
        print(f'GPU {group["index"]} ({group["uuid"]}): ' + ' -> '.join(group['families']))
    print(f'Settings: {settings}', flush=True)
    if args.dry_run:
        return 0
    # Confirm CUDA access with the actual training interpreter before launching
    # any jobs or claiming success. This does not instantiate client models.
    for group in groups:
        result = subprocess.run([args.python, str(ROOT / 'main.py'), '-did', group['uuid'], '--check_device'],
                                cwd=ROOT, capture_output=True, text=True, timeout=90)
        if result.returncode:
            raise RuntimeError(f'GPU {group["index"]} CUDA preflight failed; no jobs started.\n'
                               + result.stdout + result.stderr)
        print(result.stdout.strip(), flush=True)
    log_root = ROOT / 'logs' / 'FedRG'
    log_root.mkdir(parents=True, exist_ok=True)
    lock = (log_root / '.sweep.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another FedRG sweep supervisor is already active; no jobs started')
    batch = time.strftime('%Y%m%d-%H%M%S', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
    folder = log_root / batch
    folder.mkdir()
    manifest_path = folder / 'manifest.json'
    write_json(manifest_path, dict(batch=batch, settings=settings, groups=groups, reused=existing))
    with (folder / 'supervisor.out').open('x') as log:
        process = subprocess.Popen([args.python, '-u', str(Path(__file__).resolve()),
                                    '--supervise', str(manifest_path), '--lock-fd', str(lock.fileno())],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, pass_fds=(lock.fileno(),))
    (folder / 'supervisor.pid').write_text(str(process.pid) + '\n')
    print(f'Sweep supervisor started: PID {process.pid}\nLogs and GPU queue status: {folder}', flush=True)
    # Verify workers have actually created their training processes. The jobs
    # continue independently after this command returns.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            if process.returncode == 0 and (folder / 'finished.json').exists():
                print('All queued experiments have already finished successfully.', flush=True)
                return 0
            raise RuntimeError('Supervisor exited during startup; inspect ' + str(folder / 'supervisor.out'))
        paths = [folder / f'gpu-{group["index"]}.json' for group in groups]
        if all(path.exists() for path in paths):
            jobs = [json.loads(path.read_text())[0] for path in paths]
            if any(job['status'] == 'failed' for job in jobs):
                raise RuntimeError('A training process failed during startup; inspect logs in ' + str(folder))
            # Confirm main.py has actually bound CUDA, not merely that a Python
            # PID was created. Full progress and eventual exit codes stay in logs.
            if any(job['status'] == 'running' and '[Device] selected GPU ' not in
                   Path(job['log']).read_text(errors='replace') for job in jobs):
                time.sleep(0.5)
                continue
            for job in jobs:
                print(f'{job["family"]}: {job["status"]}, PID={job.get("pid")}', flush=True)
            return 0
        time.sleep(0.5)
    raise RuntimeError('Workers have not confirmed startup yet; inspect ' + str(folder / 'supervisor.out'))


def main():
    defaults = total_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--dataset', default=defaults['dataset'])
    parser.add_argument('--times', type=int, default=int(defaults['t']))
    parser.add_argument('--global-rounds', type=int, default=int(defaults['global_rounds']))
    parser.add_argument('--batch-size', type=int, default=int(defaults['batch_size']))
    parser.add_argument('--local-epochs', type=int, default=int(defaults['local_epoch']))
    parser.add_argument('--num-clients', type=int, default=int(defaults['num_clients']))
    parser.add_argument('--num-classes', type=int, default=int(defaults['num_classes']))
    parser.add_argument('--join-ratio', type=float, default=float(defaults['join_ratio']))
    parser.add_argument('--gpus', help='Optional comma-separated nvidia-smi GPU indices')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--gpu-index', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--supervise', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker, args.gpu_index)
    if args.supervise:
        return supervise(args.supervise, args.lock_fd)
    return launch(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
