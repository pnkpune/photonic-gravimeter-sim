from __future__ import annotations

import json
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

    metrics = json.loads(
        (output_dir / "maritime_baseline_test_metrics.json").read_text()
    )
    assert metrics["pf_position_error"] is not None


def test_run_single_scenario_regional_map_cli_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "regional_run"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_single_scenario.py",
            "--scenario",
            "norwegian_margin_maritime",
            "--regional-map",
            "norwegian_margin",
            "--output-dir",
            str(output_dir),
            "--run-id",
            "regional",
            "--seed",
            "123",
            "--dt-s",
            "2.0",
            "--pf-particles",
            "32",
            "--map-matcher",
            "sequence",
            "--use-gradiometer",
            "--use-sequence-lag-smoother",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert (output_dir / "norwegian_margin_maritime_regional.npz").exists()
    assert (output_dir / "norwegian_margin_maritime_regional_summary.json").exists()
    assert (output_dir / "norwegian_margin_maritime_regional_metrics.json").exists()
    assert (output_dir / "norwegian_margin_maritime_regional_config.json").exists()

    config = json.loads(
        (output_dir / "norwegian_margin_maritime_regional_config.json").read_text()
    )
    metrics = json.loads(
        (output_dir / "norwegian_margin_maritime_regional_metrics.json").read_text()
    )
    assert "norwegian_margin_gravity_map.npz" in config["runner"]["map_source"]
    assert config["runner"]["regional_map"]["region_name"] == "norwegian_margin"
    assert metrics["sequence_position_error"] is not None
    assert metrics["lag_smoothed_position_error"] is not None
