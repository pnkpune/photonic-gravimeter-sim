from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_regional_benchmark_summary_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "benchmark"
    scenario_path = root / "configs/scenarios/norwegian_margin_maritime.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_filters.py",
            "--profile",
            "regional_core",
            "--scenario",
            str(scenario_path),
            "--regional-map",
            "norwegian_margin",
            "--output-dir",
            str(output_dir),
            "--seed",
            "123",
            "--dt-s",
            "2.0",
            "--pf-particles",
            "24",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr

    summary_path = output_dir / "norwegian_margin_maritime_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text())
    assert payload["profile"] == "regional_core"
    assert payload["scenario_output_name"] == "norwegian_margin_maritime"
    assert payload["regional_map"] == "norwegian_margin"

    labels = {row["label"] for row in payload["runs"]}
    assert labels == {
        "ins_only",
        "observe_only",
        "observe_plus_gradient",
        "sequence_plus_gradient",
        "sequence_plus_gradient_lag_smoothed",
    }

    for row in payload["runs"]:
        assert row["horizontal_rmse_m"] is not None
        assert row["cep95_m"] is not None
