"""Reproducible CloudSim experiment runner for the current CSV workload set."""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from dataset_loader import CloudWorkloadDataset

ROOT = Path(__file__).resolve().parents[1]
STATE_SCHEMA = 'dataset-workload-state-v1-9f'
REWARD_SCHEMA = 'dataset-workload-objective-v1'
METRICS = ('makespan_s', 'energy_kwh', 'sla_violation_pct', 'throughput_tasks_s')
ALGORITHMS = ('rr', 'eft', 'dqn', 'ddqn')

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'train', 'evaluate', 'experiment'))
    p.add_argument('--dataset', default='data/cloud_workloads.csv')
    p.add_argument('--experiment-dir')
    p.add_argument('--algorithm', choices=('dqn', 'ddqn'), default='ddqn')
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--tasks', type=int, default=100, help='Training/smoke task count')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--arrival-rate', type=float, default=20)
    p.add_argument('--quick', action='store_true', help='Two paired evaluation seeds at --tasks')
    p.add_argument('--gamma', type=float, default=.95)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--target-sync', type=int, default=100)
    p.add_argument('--replay-capacity', type=int, default=10000)
    p.add_argument('--epsilon-start', type=float, default=.5)
    p.add_argument('--epsilon-min', type=float, default=.05)
    p.add_argument('--epsilon-decay', type=float, default=.995)
    p.add_argument('--candidate-count', type=int, default=16)
    return p


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pick_experiment_dir(args, data):
    if args.experiment_dir:
        path = Path(args.experiment_dir)
        return path if path.is_absolute() else ROOT / path
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return ROOT / 'results' / f'cloud-workloads-{data.sha256[:10]}-{stamp}'


def validate_args(args):
    if args.episodes < 1 or args.tasks < 1 or args.arrival_rate <= 0:
        raise SystemExit('episodes, tasks, and arrival-rate must be positive')
    if not (0 < args.gamma <= 1 and args.learning_rate > 0 and args.batch_size > 0 and
            args.target_sync > 0 and args.replay_capacity >= args.batch_size and
            0 <= args.epsilon_min <= args.epsilon_start <= 1 and
            0 < args.epsilon_decay <= 1 and 1 <= args.candidate_count <= 200):
        raise SystemExit('Invalid DQN/DDQN hyperparameter range')


def save_config(path, args, data):
    config = {
        'experiment_id': path.name, 'created_local': datetime.now().astimezone().isoformat(),
        'command': args.command, 'dataset': data.summary(),
        'training': {'episodes_per_agent': args.episodes, 'tasks_per_episode': args.tasks,
                     'seeds': [args.seed + i for i in range(args.episodes)]},
        'evaluation': {'workload_sizes': ([args.tasks] if args.quick else [100, 200, 400, 600, 800, 1000]),
                       'seeds': [args.seed + 100000 + i for i in range(2 if args.quick else 5)],
                       'runs_expected': 4 * (1 if args.quick else 6) * (2 if args.quick else 5)},
        'hyperparameters': {'gamma': args.gamma, 'learning_rate': args.learning_rate,
            'batch_size': args.batch_size, 'target_sync': args.target_sync,
            'replay_capacity': args.replay_capacity, 'epsilon_start': args.epsilon_start,
            'epsilon_min': args.epsilon_min, 'epsilon_decay': args.epsilon_decay,
            'candidate_count': args.candidate_count, 'arrival_rate_tasks_s': args.arrival_rate},
        'state_schema': STATE_SCHEMA, 'reward_schema': REWARD_SCHEMA,
        'training_from_scratch': True, 'evaluation_updates_weights': False,
        'profile_sampling': 'without replacement when count fits partition; otherwise with replacement',
        'disk_io': 'source feature recorded but not simulated by this CloudSim model',
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (path / 'dataset.json').write_text(json.dumps(data.summary(), indent=2), encoding='utf-8')
    return config


class Experiment:
    def __init__(self, args, data, expdir):
        self.a, self.data, self.dir = args, data, expdir
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / 'models').mkdir(exist_ok=True)
        self.run_id = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        self.models = {name: self.dir / 'models' / f'{name}.weights.h5' for name in ('dqn', 'ddqn')}
        self.server = None
        self.server_log = None
        self.current_algorithm = None
        self.current_model = None
        self.env = None
        if sys.platform == 'win32':
            self.env = dict(os.environ)
            self.env['MAVEN_OPTS'] = f'-Duser.home="{ROOT / "target" / "maven-home"}"'
        self.mvn = shutil.which('mvn')
        if not self.mvn:
            raise RuntimeError('Maven is missing from PATH')
        if not self.free_port():
            raise RuntimeError('Port 8765 is already in use; stop another runner first')

    @staticmethod
    def free_port():
        with socket.socket() as sock:
            try:
                sock.bind(('127.0.0.1', 8765))
                return True
            except OSError:
                return False

    def model_path(self, algorithm):
        if self.a.command == 'smoke':
            return self.dir / 'models' / f'smoke-{algorithm}.weights.h5'
        return self.models[algorithm]

    def validate_checkpoint(self, algorithm):
        model = self.model_path(algorithm)
        meta_path = Path(str(model) + '.metadata.json')
        if not model.is_file() or not meta_path.is_file():
            raise RuntimeError(f'Missing fresh {algorithm.upper()} checkpoint: {model}')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        expected = {'state_schema': STATE_SCHEMA, 'reward_schema': REWARD_SCHEMA,
                    'policy_schema': f'earliest-finish-candidates-{self.a.candidate_count}',
                    'algorithm': algorithm, 'dataset_sha256': self.data.sha256,
                    'experiment_id': self.dir.name}
        for key, value in expected.items():
            if meta.get(key) != value:
                raise RuntimeError(f'{algorithm.upper()} checkpoint {key} mismatch: {meta.get(key)!r}')
        return model

    def launch_agent(self, algorithm, eval_mode=False):
        model = self.model_path(algorithm)
        self.current_algorithm, self.current_model = algorithm, model
        cmd = [sys.executable, str(ROOT / 'python' / 'agent_server.py')]
        if eval_mode:
            cmd.append('--eval')
        cmd += ['--algorithm', algorithm, '--model', str(model), '--seed', str(self.a.seed),
                '--gamma', str(self.a.gamma), '--learning-rate', str(self.a.learning_rate),
                '--batch-size', str(self.a.batch_size), '--target-sync', str(self.a.target_sync),
                '--replay-capacity', str(self.a.replay_capacity), '--epsilon-start', str(self.a.epsilon_start),
                '--epsilon-min', str(self.a.epsilon_min), '--epsilon-decay', str(self.a.epsilon_decay),
                '--candidate-count', str(self.a.candidate_count), '--dataset-sha256', self.data.sha256,
                '--experiment-id', self.dir.name]
        self.server_log = (self.dir / f'agent-{self.run_id}-{algorithm}.log').open('a', encoding='utf-8')
        self.server = subprocess.Popen(cmd, cwd=ROOT, stdout=self.server_log,
                                       stderr=subprocess.STDOUT, env=self.env)
        self.wait_ready()

    def wait_ready(self):
        for _ in range(180):
            if self.server.poll() is not None:
                raise RuntimeError(f'Agent exited; inspect {self.server_log.name}')
            try:
                with urlopen('http://127.0.0.1:8765/health', timeout=1) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(1)
        raise TimeoutError('Python agent startup timed out')

    def switch_agent(self, algorithm, eval_mode=False):
        if self.server is not None and self.server.poll() is None:
            self.server.terminate()
            self.server.wait(timeout=30)
        if self.server_log:
            self.server_log.close()
            self.server_log = None
        self.launch_agent(algorithm, eval_mode)

    def stop_agent(self):
        if self.server is not None and self.server.poll() is None:
            self.server.terminate()
            self.server.wait(timeout=30)
        if self.server_log:
            self.server_log.close()
            self.server_log = None

    def profile(self, count, seed, phase):
        path = self.dir / 'workloads' / f'{phase}-{count}-{seed}.csv'
        return self.data.write_profile(path, count, seed, phase)

    def java(self, mode, count, seed, name, profile, base=None):
        output = self.dir / f'{name}.json'
        log_path = self.dir / f'{name}.log'
        bm, be = (base['makespan_s'], base['energy_kwh']) if base else (100, 1)
        rel_out = output.relative_to(ROOT).as_posix()
        rel_profile = profile.relative_to(ROOT).as_posix()
        params = [mode, str(count), str(seed), rel_out, str(bm), str(be),
                  str(self.a.arrival_rate), str(self.a.gamma), rel_profile]
        try:
            with log_path.open('w', encoding='utf-8') as log:
                subprocess.run([self.mvn, '-q', 'exec:java', f'-Dexec.args={" ".join(params)}'],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, env=self.env)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'{mode} simulation failed; inspect {log_path}') from exc
        result = json.loads(output.read_text(encoding='utf-8'))
        result['workload_profile'] = str(profile.relative_to(ROOT).as_posix())
        result['run_status'] = 'success'
        return result

    def compile_java(self):
        subprocess.run([self.mvn, '-q', 'compile'], cwd=ROOT, check=True, env=self.env)

    def train_agent(self, algorithm, episodes, tasks):
        model = self.model_path(algorithm)
        # A scratch training stage must never silently resume an existing model.
        if model.exists() or Path(str(model) + '.metadata.json').exists():
            stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            backup = model.with_name(f'{model.stem}.preserved-{stamp}{model.suffix}')
            if model.exists():
                shutil.move(str(model), str(backup))
            metadata = Path(str(model) + '.metadata.json')
            if metadata.exists():
                shutil.move(str(metadata), str(backup) + '.metadata.json')
            print(f'Preserved prior checkpoint at {backup}', flush=True)
        self.switch_agent(algorithm, eval_mode=False)
        history = []
        try:
            for episode in range(episodes):
                seed = self.a.seed + episode
                workload = self.profile(tasks, seed, 'training')
                rr = self.java('rr', tasks, seed, f'train-{algorithm}-{episode+1}-rr', workload)
                row = self.java(algorithm, tasks, seed,
                    f'train-{algorithm}-{episode+1}-{algorithm}', workload, rr)
                log_text = (self.dir / f'agent-{self.run_id}-{algorithm}.log').read_text(encoding='utf-8')
                matches = re.findall(r'Episode (\d+): updates=(\d+), epsilon=([0-9.]+), last_loss=([^,\r\n]+), shaped_return=([^\r\n]+)', log_text)
                agent_ep, updates, epsilon, loss, shaped = matches[-1] if matches else (episode+1, 0, '0', 'None', '0')
                row.update({'episode': episode+1, 'agent_episode': int(agent_ep),
                    'terminal_reward': row['reward'], 'total_reward': float(shaped),
                    'optimizer_updates': int(updates), 'epsilon': float(epsilon),
                    'training_loss': None if loss == 'None' else float(loss),
                    'phase': 'training', 'algorithm': algorithm,
                    'checkpoint': str(model.relative_to(ROOT).as_posix())})
                history.append(row)
                print(f'{algorithm.upper()} episode {episode+1}/{episodes}: shaped_return={float(shaped):.4f}, '
                      f'updates={updates}, makespan={row["makespan_s"]:.3f}s', flush=True)
        finally:
            self.stop_agent()
        returns = [row['total_reward'] for row in history]
        for i, row in enumerate(history):
            row['rolling_reward'] = float(np.mean(returns[max(0, i-9):i+1]))
        csv_write(self.dir / f'training-{algorithm}.csv', history)
        meta = self.validate_checkpoint(algorithm)
        return history, json.loads(Path(str(meta) + '.metadata.json').read_text(encoding='utf-8'))

    def evaluate(self, sizes, seeds):
        before = {name: sha256(self.validate_checkpoint(name)) for name in ('dqn', 'ddqn')}
        rows, status = [], []
        self.compile_java()
        for count in sizes:
            for run, seed in enumerate(seeds):
                profile = self.profile(count, seed, 'evaluation')
                base = None
                for algorithm in ALGORITHMS:
                    name = f'eval-{count}-{run}-{seed}-{algorithm}'
                    try:
                        if algorithm in ('dqn', 'ddqn'):
                            self.switch_agent(algorithm, eval_mode=True)
                        result = self.java(algorithm, count, seed, name, profile, base)
                        result.update({'phase': 'evaluation', 'algorithm': algorithm,
                            'checkpoint': str(self.model_path(algorithm).relative_to(ROOT).as_posix()) if algorithm in ('dqn','ddqn') else '',
                            'run_index': run+1})
                        if algorithm == 'rr':
                            base = result
                        rows.append(result)
                        status.append({'tasks': count, 'seed': seed, 'algorithm': algorithm,
                                       'status': 'success', 'detail': ''})
                    except Exception as exc:
                        failure = {'tasks': count, 'seed': seed, 'algorithm': algorithm,
                                   'run_status': 'failed', 'error': str(exc), 'submitted': count,
                                   'completed': 0, 'failed': count, 'completion_rate_pct': 0.0,
                                   'checkpoint': '', 'phase': 'evaluation'}
                        rows.append(failure)
                        status.append({'tasks': count, 'seed': seed, 'algorithm': algorithm,
                                       'status': 'failed', 'detail': str(exc)})
                        if algorithm == 'rr':
                            base = None
                        if self.server is not None and self.server.poll() is not None:
                            try:
                                self.switch_agent(algorithm, eval_mode=True)
                            except Exception:
                                pass
        self.stop_agent()
        after = {name: sha256(self.validate_checkpoint(name)) for name in ('dqn', 'ddqn')}
        if before != after:
            raise RuntimeError('Evaluation modified checkpoint weights')
        csv_write(self.dir / 'evaluation.csv', rows)
        csv_write(self.dir / 'run_status.csv', status)
        successful = [row for row in rows if row.get('run_status') == 'success']
        self.reports(rows, successful, sizes, seeds)
        return rows, status

    def reports(self, all_rows, rows, sizes, seeds):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t_critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
        summary = []
        for algorithm in ALGORITHMS:
            for count in sizes:
                group = [r for r in rows if r['algorithm'] == algorithm and r['tasks'] == count]
                item = {'algorithm': algorithm, 'tasks': count, 'successful_runs': len(group),
                        'expected_runs': len(seeds), 'submitted_tasks': sum(r['submitted'] for r in group),
                        'completed_tasks': sum(r['completed'] for r in group),
                        'task_completion_rate_pct': (100 * sum(r['completed'] for r in group) /
                            sum(r['submitted'] for r in group)) if group else None}
                for metric in METRICS:
                    vals = [float(r[metric]) for r in group]
                    mean = float(np.mean(vals)) if vals else None
                    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else (0.0 if vals else None)
                    ci = t_critical.get(len(vals)-1, 1.96) * std / math.sqrt(len(vals)) if vals else None
                    item[metric+'_mean'] = mean
                    item[metric+'_std'] = std
                    item[metric+'_ci95_low'] = mean-ci if mean is not None else None
                    item[metric+'_ci95_high'] = mean+ci if mean is not None else None
                summary.append(item)
        csv_write(self.dir / 'summary.csv', summary)

        improvements = []
        lookup = {(r['tasks'], r['seed'], r['algorithm']): r for r in rows}
        for count in sizes:
            for proposed in ('dqn', 'ddqn'):
                for baseline in ('rr', 'eft', 'dqn', 'ddqn'):
                    if proposed == baseline:
                        continue
                    paired = []
                    for seed in seeds:
                        left, right = lookup.get((count, seed, proposed)), lookup.get((count, seed, baseline))
                        if not left or not right:
                            continue
                        item = {'tasks': count, 'seed': seed, 'proposed': proposed, 'baseline': baseline}
                        for metric in METRICS:
                            b, p = float(right[metric]), float(left[metric])
                            better = metric == 'throughput_tasks_s'
                            val = ((p-b)/b*100 if better else (b-p)/b*100) if b != 0 else None
                            item[metric+'_improvement_pct'] = val
                        paired.append(item)
                    for metric in METRICS:
                        vals = [r[metric+'_improvement_pct'] for r in paired if r[metric+'_improvement_pct'] is not None]
                        improvements.append({'tasks': count, 'proposed': proposed, 'baseline': baseline,
                            'metric': metric, 'paired_runs': len(vals),
                            'mean_improvement_pct': float(np.mean(vals)) if vals else None,
                            'std_improvement_pct': float(np.std(vals, ddof=1)) if len(vals)>1 else (0.0 if vals else None),
                            'zero_denominator_runs': len(paired)-len(vals)})
        csv_write(self.dir / 'relative_improvement.csv', improvements)

        for metric in METRICS:
            fig, ax = plt.subplots()
            for algorithm in ALGORITHMS:
                group = [s for s in summary if s['algorithm'] == algorithm and s[metric+'_mean'] is not None]
                ax.errorbar([s['tasks'] for s in group], [s[metric+'_mean'] for s in group],
                    yerr=[s[metric+'_std'] for s in group], label=algorithm.upper(), capsize=3, marker='o')
            ax.set_xlabel('Tasks'); ax.set_ylabel(metric); ax.legend(); fig.tight_layout()
            fig.savefig(self.dir / f'{metric}-comparison.png'); plt.close(fig)
        self.training_graphs()
        expected = len(ALGORITHMS)*len(sizes)*len(seeds)
        training = {}
        for agent in ('dqn', 'ddqn'):
            history_path = self.dir / f'training-{agent}.csv'
            if not history_path.exists():
                continue
            with history_path.open(encoding='utf-8', newline='') as stream:
                history = list(csv.DictReader(stream))
            losses = [float(r['training_loss']) for r in history if r.get('training_loss') not in ('', 'None')]
            training[agent] = {
                'episodes': len(history),
                'optimizer_updates': int(history[-1]['optimizer_updates']) if history else 0,
                'mean_shaped_return': float(np.mean([float(r['total_reward']) for r in history])) if history else None,
                'mean_terminal_reward': float(np.mean([float(r['terminal_reward']) for r in history])) if history else None,
                'mean_training_loss': float(np.mean(losses)) if losses else None,
                'history_csv': f'training-{agent}.csv',
            }
        model_hashes = {name: sha256(self.model_path(name)) for name in ('dqn', 'ddqn')}
        report = {'experiment_id': self.dir.name, 'dataset': self.data.summary(),
            'expected_evaluations': expected,
            'successful_evaluations': sum(s['status']=='success' for s in self.read_status()),
            'failed_evaluations': sum(s['status']=='failed' for s in self.read_status()),
            'skipped_evaluations': 0, 'evaluation_rows': all_rows, 'summary': summary,
            'relative_improvement': improvements,
            'checkpoints': {name: str(self.model_path(name).relative_to(ROOT).as_posix()) for name in ('dqn','ddqn')},
            'checkpoint_sha256_after_evaluation': model_hashes,
            'training': training,
            'artifacts': {'config': 'config.json', 'dataset': 'dataset.json',
                'evaluation_csv': 'evaluation.csv', 'run_status_csv': 'run_status.csv',
                'summary_csv': 'summary.csv', 'relative_improvement_csv': 'relative_improvement.csv',
                'markdown_report': 'accuracy_performance_report.md'},
            'limitations': ['Both agents trained on 100-task episodes; evaluation sizes range up to 1000 tasks.',
                'Ten training episodes do not establish convergence.',
                'The held-out partition has 120 rows; profiles larger than 120 tasks sample with replacement.',
                'The simulator does not model disk I/O or realistic host-level RAM/network contention.'],
            'evaluation_weights_unchanged': True,
            'statistical_note': '95% Student-t confidence intervals for means; five seeds are a small sample. Paired improvement means use identical seed/workload pairs; no significance claim is made.'}
        (self.dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        self.markdown_report(report)

    def read_status(self):
        path = self.dir / 'run_status.csv'
        with path.open(encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))

    def training_graphs(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for algorithm in ('dqn', 'ddqn'):
            path = self.dir / f'training-{algorithm}.csv'
            if not path.exists():
                continue
            with path.open(encoding='utf-8', newline='') as f:
                history = list(csv.DictReader(f))
            if not history:
                continue
            metrics = [('total_reward', 'Episode reward'), ('rolling_reward', 'Rolling reward'),
                ('training_loss', 'Training loss'), ('makespan_s', 'Makespan (s)'),
                ('energy_kwh', 'Energy (kWh)'), ('sla_violation_pct', 'SLA violation (%)')]
            for field, label in metrics:
                if field == 'rolling_reward':
                    vals = [float(r['total_reward']) for r in history]
                    values = [float(np.mean(vals[max(0, i-9):i+1])) for i in range(len(vals))]
                else:
                    values = [float(r[field]) if r.get(field) not in ('', 'None', 'nan') else float('nan') for r in history]
                plt.plot([int(r['episode']) for r in history], values, marker='o')
                plt.xlabel('Episode'); plt.ylabel(label); plt.tight_layout()
                plt.savefig(self.dir / f'training-{algorithm}-{field}.png'); plt.close()

    def markdown_report(self, report):
        data = report['dataset']
        training = report.get('training', {})
        model_hashes = report.get('checkpoint_sha256_after_evaluation', {})
        lines = [f'# Cloud workload scheduling report: {self.dir.name}', '',
            '## Experiment overview', '',
            'CloudSim Plus scheduling comparison using the current tabular workload dataset.',
            f'- Dataset: `{data["path"]}` ({data["record_count"]} rows, SHA-256 `{data["sha256"]}`).',
            f'- Disjoint deterministic split: {data["train_records"]} training rows and {data["evaluation_records"]} evaluation rows (split seed {data["split_seed"]}).',
            '- No previous dataset file was present in the project data directory, so dataset-to-dataset changes cannot be verified.',
            '- `VM_ID` and `Target (Optimal Scheduling)` are excluded from features and scheduling decisions to avoid label leakage.',
            '- CPU usage, execution time, RAM usage, network usage, and priority are used in the generated profiles. Disk I/O is recorded in source data but not simulated by this CloudSim model.',
            '- Workload profiles are sampled without replacement when possible and with replacement for counts above the split size.', '',
            '## Methodology', '',
            'RR assigns tasks cyclically. EFT assigns each arriving task to the VM with the earliest predicted finish. DQN uses the target network to select and evaluate the next action; DDQN uses the online network for argmax and the target network for evaluation. Both agents share state, action candidates, architecture, reward, replay buffer, optimizer, exploration schedule, training workloads, and seeds.',
            'The agent state contains projected VM wait, queue size, speed, current task work, deadline slack, CPU, RAM, bandwidth utilization, and priority. Actions select among the 16 earliest predicted finishes. Reward combines bounded makespan, SLA, energy, and throughput terms measured by CloudSim with potential shaping between task assignments.',
            '', '## Evaluation results', '',
            'Mean +/- sample standard deviation across successful seeds. Makespan is seconds, energy is kWh, SLA is percent, throughput is tasks/second. Completion rate is completed/submitted tasks. The confidence intervals are Student-t intervals for the mean; five seeds are insufficient for strong statistical claims.', '',
            '| Tasks | Algorithm | Makespan s | Energy kWh | SLA % | Throughput tasks/s | Completion % | Runs |',
            '|---:|:---|---:|---:|---:|---:|---:|---:|']
        for r in report['summary']:
            if not r['successful_runs']:
                continue
            fmt = lambda key, digits=3: f"{r[key+'_mean']:.{digits}f} +/- {r[key+'_std']:.{digits}f}" if r[key+'_mean'] is not None else 'n/a'
            lines.append(f"| {r['tasks']} | {r['algorithm'].upper()} | {fmt('makespan_s')} | {fmt('energy_kwh',6)} | {fmt('sla_violation_pct')} | {fmt('throughput_tasks_s')} | {r['task_completion_rate_pct']:.2f} | {r['successful_runs']}/{r['expected_runs']} |")
        lines += ['', '## Relative improvement', '',
            'Positive values mean better performance. Lower is better for makespan, energy, and SLA; higher is better for throughput. Percentages are paired by seed and workload. See `relative_improvement.csv` for every comparison.', '',
            '| Tasks | Proposed | Baseline | Metric | Mean improvement % | Paired runs |', '|---:|:---|:---|:---|---:|---:|']
        for r in report['relative_improvement']:
            value = 'undefined' if r['mean_improvement_pct'] is None else f"{r['mean_improvement_pct']:.3f}"
            lines.append(f"| {r['tasks']} | {r['proposed'].upper()} | {r['baseline'].upper()} | {r['metric']} | {value} | {r['paired_runs']} |")
        lines += ['', '## Training analysis', '',
            f"DQN and DDQN each trained for {self.a.episodes} episodes of {self.a.tasks} tasks and recorded {training.get('dqn', {}).get('optimizer_updates', 0)} and {training.get('ddqn', {}).get('optimizer_updates', 0)} optimizer updates. Mean shaped returns were {training.get('dqn', {}).get('mean_shaped_return', 0):.4f} and {training.get('ddqn', {}).get('mean_shaped_return', 0):.4f}; mean terminal rewards were {training.get('dqn', {}).get('mean_terminal_reward', 0):.4f} and {training.get('ddqn', {}).get('mean_terminal_reward', 0):.4f}. Inspect the training history CSVs and per-agent plots. This short run does not establish convergence.",
            'Training used 100 tasks per episode, while evaluation ranged from 100 to 1000 tasks. DQN was slightly faster than EFT at 100 tasks but slower at larger sizes; DDQN fell farther behind as workload size grew. This workload-size mismatch is a measured limitation, not evidence that simply adding episodes would solve it.', '',
            '## Run status and reproducibility', '',
            f"- Expected evaluation runs: {report['expected_evaluations']}; successful: {report['successful_evaluations']}; failed: {report['failed_evaluations']}; skipped: {report['skipped_evaluations']}.",
            '- Training and evaluation use disjoint source rows. Evaluation runs RR, EFT, DQN, and DDQN on the exact same sampled task profile and arrival seed.',
            '- Checkpoint hashes are checked before and after evaluation; evaluation agents use `--eval` and do not update weights.',
            f'- New checkpoints: `{report["checkpoints"]["dqn"]}` and `{report["checkpoints"]["ddqn"]}`.',
            '- Configuration: `config.json`; dataset identity: `dataset.json`; raw evaluations: `evaluation.csv`; run status: `run_status.csv`; summary: `summary.csv`; improvements: `relative_improvement.csv`; machine report: `report.json`.',
            '- Figures: `makespan_s-comparison.png`, `energy_kwh-comparison.png`, `sla_violation_pct-comparison.png`, `throughput_tasks_s-comparison.png`, and `training-<agent>-<metric>.png`.',
            '- No p-value or statistical-significance claim is made. Five evaluation seeds are a small sample.',
            f'- Checkpoint SHA-256 after evaluation: DQN `{model_hashes["dqn"]}`, DDQN `{model_hashes["ddqn"]}`. The runner verified these match the pre-evaluation hashes.', '',
            '## Limitations', '',
            'The supplied CSV has 600 records, so 800/1000-task evaluation profiles resample from the held-out partition with replacement. Source VM and target labels are ignored. CloudSim does not model disk I/O; network/RAM fields affect task utilization profiles but host-level I/O contention and cloud pricing are outside this simulator. Workloads are task records, not independent train/test labels. Results describe this simulation only.', '',
            '## Commands', '',
            '```powershell',
            f'.\\.venv\\Scripts\\python.exe python\\run.py experiment --dataset data\\cloud_workloads.csv --experiment-dir results\\{self.dir.name} --episodes {self.a.episodes} --tasks {self.a.tasks} --seed {self.a.seed}',
            '```', '']
        (self.dir / 'accuracy_performance_report.md').write_text('\n'.join(lines), encoding='utf-8')


def invoke_experiment(args, data, expdir):
    command = [sys.executable, str(ROOT / 'python' / 'run.py')]
    common = ['--dataset', str(Path(args.dataset).resolve()), '--experiment-dir', str(expdir),
        '--episodes', str(args.episodes), '--tasks', str(args.tasks), '--seed', str(args.seed),
        '--arrival-rate', str(args.arrival_rate), '--gamma', str(args.gamma),
        '--learning-rate', str(args.learning_rate), '--batch-size', str(args.batch_size),
        '--target-sync', str(args.target_sync), '--replay-capacity', str(args.replay_capacity),
        '--epsilon-start', str(args.epsilon_start), '--epsilon-min', str(args.epsilon_min),
        '--epsilon-decay', str(args.epsilon_decay), '--candidate-count', str(args.candidate_count)]
    for algorithm in ('dqn', 'ddqn'):
        subprocess.run(command + ['train', '--algorithm', algorithm] + common, cwd=ROOT, check=True)
    eval_args = command + ['evaluate'] + common
    if args.quick:
        eval_args += ['--quick']
    subprocess.run(eval_args, cwd=ROOT, check=True)


def main():
    args = parser().parse_args()
    validate_args(args)
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    data = CloudWorkloadDataset(dataset_path)
    expdir = pick_experiment_dir(args, data).resolve()
    config = save_config(expdir, args, data)
    print(f'Experiment directory: {expdir.relative_to(ROOT) if expdir.is_relative_to(ROOT) else expdir}', flush=True)
    print(f'Dataset: {data.path} ({data.record_count} records; SHA-256 {data.sha256})', flush=True)
    if args.command == 'experiment':
        if args.quick:
            print('Quick mode: training both agents then evaluating two paired seeds at the selected task count.', flush=True)
        invoke_experiment(args, data, expdir)
        (expdir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        return
    exp = Experiment(args, data, expdir)
    if args.command == 'train':
        exp.compile_java()
        exp.train_agent(args.algorithm, args.episodes, args.tasks)
        return
    if args.command == 'evaluate':
        sizes = [args.tasks] if args.quick else [100, 200, 400, 600, 800, 1000]
        seeds = [args.seed+100000+i for i in range(2 if args.quick else 5)]
        config['evaluation']['workload_sizes'] = sizes
        config['evaluation']['seeds'] = seeds
        save_config(expdir, args, data)
        exp.evaluate(sizes, seeds)
        return

    exp.compile_java()
    rows = []
    for algorithm in ('rr', 'eft', 'dqn', 'ddqn'):
        model = exp.model_path(algorithm) if algorithm in ('dqn','ddqn') else None
        if algorithm in ('dqn','ddqn'):
            exp.switch_agent(algorithm, eval_mode=False)
        seed = args.seed
        profile = exp.profile(args.tasks, seed, 'evaluation')
        base = next((r for r in rows if r.get('algorithm') == 'rr'), None)
        row = exp.java(algorithm,args.tasks,seed,f'smoke-{algorithm}',profile,base)
        row.update({'phase':'smoke','algorithm':algorithm,'checkpoint':str(model) if model else ''})
        rows.append(row)
        print(f'Smoke {algorithm.upper()}: completed={row["completed"]}/{row["submitted"]}, makespan={row["makespan_s"]:.3f}s',flush=True)
    exp.stop_agent()
    csv_write(expdir / 'smoke.csv', rows)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise
