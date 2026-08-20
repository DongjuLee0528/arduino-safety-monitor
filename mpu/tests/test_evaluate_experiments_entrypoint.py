import subprocess
import sys
from pathlib import Path


def test_evaluate_experiments_module_entrypoint_imports_mpu():
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-m", "mpu.ai.runs.evaluate_experiments", "--help"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--exp-a-dir" in proc.stdout
