"""Validate and turn the cloud workload table into CloudSim task profiles.

VM_ID and Target are deliberately excluded: they are labels from the source
table and are not valid observations for this online scheduling experiment.
"""
import csv
import hashlib
import random
from pathlib import Path


REQUIRED = (
    'Task_ID', 'CPU_Usage (%)', 'RAM_Usage (MB)', 'Disk_IO (MB/s)',
    'Network_IO (MB/s)', 'Priority', 'Execution_Time (s)',
)
PROFILE_FIELDS = ('length_mi', 'cpu_util', 'ram_util', 'bw_util', 'priority', 'execution_s', 'source_task_id')


class CloudWorkloadDataset:
    def __init__(self, path, split_seed=2026, train_fraction=.8):
        self.path = Path(path).resolve()
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        with self.path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not set(REQUIRED).issubset(reader.fieldnames):
                raise ValueError(f'{self.path} must contain columns: {", ".join(REQUIRED)}')
            rows = list(reader)
        if len(rows) < 2:
            raise ValueError('Dataset must contain at least two task records')
        seen_ids = set()
        seen_profiles = set()
        duplicate_profiles = 0
        parsed = []
        for line, row in enumerate(rows, start=2):
            try:
                task_id = int(row['Task_ID'])
                cpu = float(row['CPU_Usage (%)'])
                ram = float(row['RAM_Usage (MB)'])
                disk = float(row['Disk_IO (MB/s)'])
                network = float(row['Network_IO (MB/s)'])
                priority = int(row['Priority'])
                execution = float(row['Execution_Time (s)'])
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Invalid or missing workload value on CSV line {line}') from exc
            if task_id in seen_ids:
                raise ValueError(f'Duplicate Task_ID {task_id} on CSV line {line}')
            seen_ids.add(task_id)
            if not (0 < cpu <= 100 and 0 < ram <= 16384 and disk >= 0 and
                    0 < network <= 50 and 1 <= priority <= 3 and execution > 0):
                raise ValueError(f'Out-of-range workload value on CSV line {line}: {row}')
            profile = (cpu, ram, disk, network, priority, execution)
            if profile in seen_profiles:
                duplicate_profiles += 1
            seen_profiles.add(profile)
            parsed.append({
                'task_id': task_id, 'cpu': cpu / 100.0, 'ram': ram / 16384.0,
                'disk': disk, 'network': network / 50.0, 'priority': priority,
                'execution': execution,
            })
        self.record_count = len(parsed)
        self.duplicate_workload_records = duplicate_profiles
        self.columns = tuple(reader.fieldnames)
        # Stable, reproducible disjoint split. Target and VM_ID never affect it.
        shuffled = list(parsed)
        random.Random(split_seed).shuffle(shuffled)
        split = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_fraction)))
        self.train_rows, self.evaluation_rows = shuffled[:split], shuffled[split:]
        self.split_seed = split_seed

    def write_profile(self, path, count, seed, phase):
        if count <= 0:
            raise ValueError('Task count must be positive')
        source = self.train_rows if phase == 'training' else self.evaluation_rows
        rng = random.Random(seed)
        selected = rng.sample(source, count) if count <= len(source) else [rng.choice(source) for _ in range(count)]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=PROFILE_FIELDS)
            writer.writeheader()
            for row in selected:
                # At 1500 MIPS, this length and utilization yield the recorded
                # execution_s; faster/slower VMs scale execution proportionally.
                writer.writerow({
                    'length_mi': round(row['execution'] * 1500 * row['cpu']),
                    'cpu_util': row['cpu'], 'ram_util': row['ram'],
                    'bw_util': row['network'], 'priority': row['priority'],
                    'execution_s': row['execution'], 'source_task_id': row['task_id'],
                })
        return path

    def summary(self):
        return {
            'path': str(self.path), 'sha256': self.sha256,
            'record_count': self.record_count, 'columns': list(self.columns),
            'duplicate_task_profiles': self.duplicate_workload_records,
            'train_records': len(self.train_rows),
            'evaluation_records': len(self.evaluation_rows),
            'split_seed': self.split_seed, 'target_used': False, 'vm_id_used': False,
            'disk_io_simulated': False,
        }
