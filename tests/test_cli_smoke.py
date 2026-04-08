from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_run_single_scenario_cli_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "run"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_single_scenario.py",
            "--output-dir",
            str(output_dir),
            "--run-id",
            "test",
            "--seed",
            "123",
            "--dt-s",
            "2.0",
            "--pf-particles",
            "32",
            "--map-grid-size",
            "81",
            "--disable-pf-feedback",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert (output_dir / "maritime_baseline_test.npz").exists()
    assert (output_dir / "maritime_baseline_test_summary.json").exists()
    assert (output_dir / "maritime_baseline_test_metrics.json").exists()
    assert (output_dir / "maritime_baseline_test_config.json").exists()
