# CloudSim Plus and DQN scheduling

An independently implemented CloudSim Plus research prototype. It is not the authors' code and is not a verified reproduction of any paper's results.

For copy-paste PowerShell commands to validate, train, evaluate, and inspect runs, see [RUN_GUIDE.md](RUN_GUIDE.md).

## Requirements

Windows PowerShell, Java/JDK 17, Maven, Python 3.11 64-bit, and the packages in `requirements.txt`. The project uses CloudSim Plus 6.3.0 and TensorFlow 2.13.0. Run commands from the project root. The Maven and pip caches need network access on first setup.

```powershell
java -version
javac -version
mvn -version
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import tensorflow, numpy, matplotlib; print(tensorflow.__version__, numpy.__version__, matplotlib.__version__)"
```

## Validate and smoke test

```powershell
.\.venv\Scripts\python.exe -m py_compile python\agent_server.py python\run.py python\dataset_loader.py python\dqn_targets.py python\train.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe python\run.py smoke --dataset data\cloud_workloads.csv --tasks 100 --seed 42
```

The smoke test compiles Java, then runs RR, EFT, DQN, and DDQN on the same dataset-derived 100-task profile. Each policy must complete every submitted task.

## Current dataset and workload construction

The active dataset is `data/cloud_workloads.csv` (600 task rows). `python/dataset_loader.py` checks required columns, missing/invalid values, unique task IDs, and supported numeric ranges; it records a SHA-256 identity for each experiment. The workload fields CPU usage, execution time, RAM use, network I/O, and priority are used in CloudSim task profiles. The `VM_ID` and `Target (Optimal Scheduling)` columns are excluded from observations and scheduling so the source labels cannot leak into decisions. Disk I/O is recorded but not simulated because this CloudSim model has no disk-contention model.

A stable 80/20 shuffled split (split seed 2026) separates training records from evaluation records. Profiles are sampled without replacement when the requested task count fits a partition, and with replacement above its capacity. The 800- and 1000-task evaluation workloads therefore reuse held-out records. This is a small dataset and those larger profiles are not 800/1000 independent source records.

The source provides CPU utilization and execution time. Cloudlet length is derived so execution time at 1500 MIPS matches the source record, and CPU/RAM/bandwidth utilization models use the dataset fields. Priorities are included in agent state. Deadline is arrival plus 1.5 times source execution time plus 2 seconds. These are documented simulator assumptions. Dataset records do not model a production cloud's disk, network, VM placement, or pricing behavior.

## Fresh DQN and DDQN training plus full evaluation

This command creates a new dataset-hash experiment directory, trains both models independently from scratch, then evaluates RR, EFT, DQN, and DDQN over five paired seeds at 100, 200, 400, 600, 800, and 1000 tasks (120 expected runs).

```powershell
.\.venv\Scripts\python.exe python\run.py experiment --dataset data\cloud_workloads.csv --episodes 10 --tasks 100 --seed 200
```

The runner puts new checkpoints in that experiment's `models` directory. It never initializes them from the older `models/` checkpoints. Existing results are retained. Ten episodes are a short baseline and do not establish convergence.

For a smaller end-to-end check (two training episodes and two paired seeds at 100 tasks):

```powershell
.\.venv\Scripts\python.exe python\run.py experiment --dataset data\cloud_workloads.csv --episodes 2 --tasks 100 --seed 200 --quick
```

## Run individual stages

Use one experiment directory for both agents and evaluation. Select a fresh directory name when beginning a new dataset or experiment.

```powershell
$experiment = 'results\cloud-workloads-current-v1'
.\.venv\Scripts\python.exe python\run.py train --algorithm dqn --dataset data\cloud_workloads.csv --experiment-dir $experiment --episodes 10 --tasks 100 --seed 200
.\.venv\Scripts\python.exe python\run.py train --algorithm ddqn --dataset data\cloud_workloads.csv --experiment-dir $experiment --episodes 10 --tasks 100 --seed 200
.\.venv\Scripts\python.exe python\run.py evaluate --dataset data\cloud_workloads.csv --experiment-dir $experiment --seed 900
```

Evaluation verifies dataset and checkpoint metadata, loads both newly trained models, disables exploration and optimizer updates, and checks that checkpoint hashes do not change. For a faster held-out check, add `--quick --tasks 100` to the evaluate command.

The old `python/train.py` entry point now delegates to the real CloudSim DQN trainer. It no longer generates simulated metrics or imports missing agent/reward modules.

## Results and reports

Each experiment directory contains:

- `config.json`, `dataset.json`, and sampled input profiles under `workloads/`.
- `models/dqn.weights.h5` and `models/ddqn.weights.h5` with metadata sidecars.
- `training-dqn.csv`, `training-ddqn.csv`, optimizer/update logs, and reward/loss/performance plots.
- `evaluation.csv`, task-level CSVs, Java logs, Python agent logs, and `run_status.csv`.
- `summary.csv` with mean, sample standard deviation, and 95% Student-t confidence intervals.
- `relative_improvement.csv` with paired-seed improvements. Positive means better: lower makespan/energy/SLA, higher throughput. Zero denominators are marked undefined.
- `report.json`, `accuracy_performance_report.md`, and four evaluation comparison plots.

Inspect a chosen run in PowerShell:

```powershell
$experiment = 'results\cloud-workloads-current-v1'
Import-Csv "$experiment\evaluation.csv" | Format-Table
Import-Csv "$experiment\summary.csv" | Format-Table
Import-Csv "$experiment\relative_improvement.csv" | Format-Table
Import-Csv "$experiment\run_status.csv" | Group-Object status
Get-Content "$experiment\accuracy_performance_report.md"
Get-ChildItem $experiment -Filter '*.png' | Select-Object Name,Length
Invoke-Item $experiment
```

SLA is reported as a percentage, energy in kWh, makespan in seconds, and throughput in tasks/second. Completion rate is completed/submitted tasks; it is not classification accuracy. Failed runs remain in `run_status.csv`; the report does not substitute invented values for them. Five seeds are too few for strong statistical claims, so the report does not claim statistical significance.

## Algorithms and network bridge

Java owns CloudSim Plus, datacenters, hosts, VMs, arrivals, dispatch decisions, completion checks, deadlines, and energy integration. Python owns TensorFlow networks, epsilon-greedy selection, replay, optimizer updates, and checkpoints. Java calls a local Python HTTP server.

The agent state has nine features for each of 200 VMs: projected wait, queued task count, VM speed, current task work, deadline slack, and the task CPU, RAM, bandwidth, and priority values. The policy chooses among the 16 VMs with the earliest predicted finish. DQN's target is `r + gamma * (1-done) * max_a Q_target(s_next,a)`. DDQN selects the next action with the online network and evaluates it with the target network. Terminal transitions do not bootstrap. Both agents share architecture, state, action candidates, reward, replay, optimizer, and training seeds.

The dense network is 1800 -> 128 -> 64 -> 200 (ReLU hidden layers). Defaults: Adam 0.001, gamma 0.95, replay capacity 10000, batch size 64, target sync every 100 optimizer updates, epsilon 0.5 with 0.995 per-episode decay and 0.05 floor. All are configurable through `python/run.py`.

## Simulation assumptions and limits

- EFT means earliest predicted finish for the arriving task, not batch Min-Min. PSO, Dueling DQN, and multi-agent DRL are not implemented.
- The simulator uses 50 hosts and 200 heterogeneous one-core VMs. Tasks arrive at a Poisson rate (default 20 per second). Every run verifies submitted tasks equal completed tasks; an incomplete simulation fails rather than silently recording success.
- Energy uses the documented linear host power approximation; all hosts are modeled as powered on. CloudSim task RAM/bandwidth utilization does not model realistic host-level memory or network contention. Disk I/O contention, switching, migration, cooling, and cloud pricing are absent.
- The paper's settings and reward equations are underspecified and partly inconsistent. This implementation documents its choices; it does not promise paper-level metrics or exact reproduction.
- The prior synthetic workload benchmark and original checkpoints remain separate historical artifacts. Only the new experiment directory should be used for current dataset conclusions.

## Troubleshooting

- Compile/build failures: inspect the first Maven error and verify JDK 17. Maven dependencies need to be available in its local cache.
- TensorFlow import errors: use `.venv` Python 3.11 64-bit and install `requirements.txt`.
- Port 8765 in use: stop the earlier runner before launching another.
- Simulation failures: inspect the per-run Java log and the Python agent log in that experiment directory.
