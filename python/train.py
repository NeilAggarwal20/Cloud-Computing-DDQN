"""Compatibility entry point for the real CloudSim-backed DQN trainer."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
raise SystemExit(subprocess.call([
    sys.executable, str(ROOT / 'python' / 'run.py'), 'train', '--algorithm', 'dqn',
    *sys.argv[1:],
], cwd=ROOT))
