# Cloud Task Scheduling Run Guide (PowerShell)

Run these commands from the project root. The canonical dataset is `data\cloud_workloads.csv`; the runner validates its contents and records its SHA-256 in each experiment. Do not copy or regenerate it.

## 1. Open the project and check the environment

```powershell
Set-Location 'C:\Users\Neil Aggarwal\Downloads\cloud-drl-vscode-ddqn\cloud-drl-vscode'
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import tensorflow, numpy, matplotlib; print('TensorFlow', tensorflow.__version__, 'NumPy', numpy.__version__, 'Matplotlib', matplotlib.__version__)"
java -version
javac -version
mvn -version
Get-FileHash data\cloud_workloads.csv -Algorithm SHA256
(Import-Csv data\cloud_workloads.csv).Count
```

Expected dataset SHA-256: `48277be7e3cc6af1d30cba694648184a1bbd7f20faf5189eb3e355daa7b762ad`; expected record count: `600`. Verified here: Python 3.11.9, JDK 17, TensorFlow 2.13.0, NumPy 1.24.3, and Matplotlib 3.7.2.

For a direct Maven test invocation, point Maven's Java user home to a writable project-local directory. The Python runner applies this setting automatically on Windows:

```powershell
$mavenHome = Join-Path (Get-Location) 'target\maven-home'
$env:MAVEN_OPTS = '-Duser.home="' + $mavenHome + '"'
mvn -q test
```

## 2. Validate Python and run the smoke test

```powershell
.\.venv\Scripts\python.exe -m py_compile python\agent_server.py python\run.py python\dataset_loader.py python\dqn_targets.py python\train.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
$smokeRun = "results\smoke-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
.\.venv\Scripts\python.exe python\run.py smoke --dataset data\cloud_workloads.csv --experiment-dir $smokeRun --tasks 100 --seed 42
```

The smoke command compiles the Java simulator and runs RR, EFT, DQN, and DDQN. Successful output reports `completed=100/100` for every policy. Its experiment directory contains `smoke.csv`, JSON metrics, task-level CSVs, and simulator/agent logs.

## 3. Train DQN and DDQN, then evaluate their checkpoints

Use one new experiment directory for both checkpoints and its evaluation. The timestamp prevents replacing earlier experiment outputs.

```powershell
$experiment = "results\cloud-workloads-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
.\.venv\Scripts\python.exe python\run.py train --algorithm dqn --dataset data\cloud_workloads.csv --experiment-dir $experiment --episodes 10 --tasks 100 --seed 200
.\.venv\Scripts\python.exe python\run.py train --algorithm ddqn --dataset data\cloud_workloads.csv --experiment-dir $experiment --episodes 10 --tasks 100 --seed 200
```

Each training command prints per-episode shaped return, optimizer-update count, and makespan. Checkpoints and metadata are saved at `$experiment\models\dqn.weights.h5` and `$experiment\models\ddqn.weights.h5`. The runner preserves a checkpoint if training is repeated at that path, but start with a fresh directory to keep runs separate.

Evaluate the saved checkpoints against RR and EFT on two paired 100-task seeds:

```powershell
.\.venv\Scripts\python.exe python\run.py evaluate --dataset data\cloud_workloads.csv --experiment-dir $experiment --seed 900 --quick --tasks 100
```

Evaluation requires **both** DQN and DDQN checkpoints in the selected experiment directory. It verifies dataset and checkpoint metadata, loads the models with evaluation mode enabled, disables exploration and optimizer updates, and checks checkpoint hashes before and after. Remove `--quick --tasks 100` for five seeds at each of 100, 200, 400, 600, 800, and 1000 tasks (120 expected policy runs).

## 4. Run the full experiment in one command

This creates both agents from scratch, then evaluates all four algorithms. Quick mode keeps the evaluation to two paired seeds at 100 tasks:

```powershell
$experiment = "results\cloud-workloads-quick-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
.\.venv\Scripts\python.exe python\run.py experiment --dataset data\cloud_workloads.csv --experiment-dir $experiment --episodes 10 --tasks 100 --seed 200 --quick
```

For the complete evaluation suite, omit `--quick`. It runs five paired seeds at six workload sizes (120 policy evaluations), after training both agents. RR and EFT are heuristic baselines and are not trained.

## 5. Inspect outputs

```powershell
Get-ChildItem $experiment -Force
Get-ChildItem "$experiment\models" -Force
Import-Csv "$experiment\training-dqn.csv" | Format-Table episode,total_reward,optimizer_updates,makespan_s
Import-Csv "$experiment\training-ddqn.csv" | Format-Table episode,total_reward,optimizer_updates,makespan_s
Import-Csv "$experiment\evaluation.csv" | Format-Table algorithm,tasks,seed,makespan_s,energy_kwh,sla_violation_pct,throughput_tasks_s,completed,submitted
Import-Csv "$experiment\summary.csv" | Format-Table
Import-Csv "$experiment\run_status.csv" | Group-Object status
Get-Content "$experiment\accuracy_performance_report.md"
Get-ChildItem $experiment -Filter '*.png' | Select-Object Name,Length
```

The run directory also contains sampled inputs under `workloads\`, per-run JSON and task CSV files, `relative_improvement.csv`, `report.json`, and `agent-*.log` / simulator `*.log` files. Metrics currently aggregated by the runner are makespan, energy, SLA violation percentage, and throughput. The task-level CSV provides submission, start, finish, and deadline times; waiting-time and utilization aggregates are not currently reported.

## Troubleshooting

- **Missing checkpoint or metadata:** evaluate only from the experiment directory where both DQN and DDQN were trained. Check `models\*.weights.h5` and matching `.metadata.json` sidecars.
- **Maven tries `C:\.m2`:** run through `python\run.py` (it sets a project-local Maven home). For direct Maven commands, use the `MAVEN_OPTS` setup in section 1.
- **Maven dependency download fails:** Maven Central must be reachable on first use; retry when network access is available.
- **Port 8765 is busy:** stop the earlier runner/agent process before starting another run.
- **Dataset validation error:** use the unchanged `data\cloud_workloads.csv`; the loader reports missing columns, invalid values, or out-of-range rows.
- **Java run failure or incomplete tasks:** open that run's `.log` and JSON in the experiment directory. The runner marks a failed process or incomplete completion as failure instead of counting it as a successful result.
- **Experiment directory already exists:** choose a new timestamped directory so prior CSVs and reports remain intact.
