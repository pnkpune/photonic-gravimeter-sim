#!/usr/bin/env python3
"""Benchmark PF and sequence-based map-matching modes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RUNNER = PROJECT_ROOT / "scripts" / "run_single_scenario.py"


def _run_one(
    label: str,
    *,
    output_dir: Path,
    seed: int,
    scenario: str,
    extra_flags: list[str] | None = None,
) -> dict[str, Any]:
    cmd = [
        PYTHON,
        str(RUNNER),
        "--scenario", scenario,
        "--seed", str(seed),
        "--run-id", label,
        "--output-dir", str(output_dir),
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else src_path + os.pathsep + existing_pythonpath
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)

    metrics_path = output_dir / f"{scenario}_{label}_metrics.json"
    archive_path = output_dir / f"{scenario}_{label}.npz"
    metrics = json.loads(metrics_path.read_text())
    pos = metrics["ins_position_error"]
    pf = metrics.get("pf_position_error") or {}
    seq = metrics.get("sequence_position_error") or {}
    integrity = metrics.get("integrity") or {}
    directional_applied = None
    directional_allowed = None
    sequence_applied = None
    sequence_allowed = None

    if archive_path.exists():
        with np.load(archive_path, allow_pickle=False) as data:
            if "estimator_custom_streams_json" in data:
                streams = json.loads(str(np.asarray(data["estimator_custom_streams_json"]).item()))
                rows = streams.get("pf_directional_feedback", [])
                directional_allowed = sum(1 for row in rows if row.get("feedback_allowed"))
                directional_applied = sum(1 for row in rows if row.get("applied"))
                seq_rows = streams.get("sequence_feedback", [])
                sequence_allowed = sum(1 for row in seq_rows if row.get("feedback_allowed"))
                sequence_applied = sum(1 for row in seq_rows if row.get("applied"))

    return {
        "label": label,
        "horizontal_rmse_m": pos["horizontal_rmse_m"],
        "cep95_m": pos["cep95_m"],
        "horizontal_max_m": pos["horizontal_max_m"],
        "vertical_rmse_m": pos["vertical_rmse_m"],
        "pf_horizontal_rmse_m": pf.get("horizontal_rmse_m"),
        "pf_cep95_m": pf.get("cep95_m"),
        "sequence_horizontal_rmse_m": seq.get("horizontal_rmse_m"),
        "sequence_cep95_m": seq.get("cep95_m"),
        "hmi_horizontal": integrity.get("fraction_hazardously_misleading_horizontal"),
        "directional_allowed_updates": directional_allowed,
        "directional_applied_updates": directional_applied,
        "sequence_allowed_updates": sequence_allowed,
        "sequence_applied_updates": sequence_applied,
        "metrics_path": str(metrics_path),
    }


def main() -> int:
    output_dir = PROJECT_ROOT / "data" / "outputs" / "reports" / "priority3_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = "maritime_baseline"
    seed = 42
    configs = [
        {"label": "observe_only", "extra_flags": []},
        {"label": "observe_plus_gradient", "extra_flags": ["--use-gradiometer"]},
        {
            "label": "directional_plus_gradient",
            "extra_flags": ["--use-gradiometer", "--use-directional-feedback"],
        },
        {
            "label": "sequence_only",
            "extra_flags": ["--map-matcher", "sequence"],
        },
        {
            "label": "sequence_plus_gradient",
            "extra_flags": ["--map-matcher", "sequence", "--use-gradiometer"],
        },
        {
            "label": "sequence_plus_gradient_feedback",
            "extra_flags": [
                "--map-matcher", "sequence",
                "--use-gradiometer",
                "--use-sequence-feedback",
            ],
        },
    ]

    rows: list[dict[str, Any]] = []
    for cfg in configs:
        print(f"\n=== {cfg['label']} ===", flush=True)
        row = _run_one(
            cfg["label"],
            output_dir=output_dir,
            seed=seed,
            scenario=scenario,
            extra_flags=cfg["extra_flags"],
        )
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    summary_path = output_dir / f"{scenario}_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2) + "\n")

    print("\n=== SUMMARY ===")
    print(
        f"{'label':<28} {'INS_RMSE':>10} {'PF_RMSE':>10} {'SEQ_RMSE':>10} "
        f"{'INS_CEP95':>10} {'PF_CEP95':>10} {'SEQ_CEP95':>10} {'HMI%':>10} {'PF_APPLY':>10} {'SEQ_APPLY':>10}"
    )
    for row in rows:
        hmi = row["hmi_horizontal"]
        hmi_pct = float('nan') if hmi is None else 100.0 * float(hmi)
        pf_rmse = float("nan") if row["pf_horizontal_rmse_m"] is None else float(row["pf_horizontal_rmse_m"])
        pf_cep95 = float("nan") if row["pf_cep95_m"] is None else float(row["pf_cep95_m"])
        seq_rmse = float("nan") if row["sequence_horizontal_rmse_m"] is None else float(row["sequence_horizontal_rmse_m"])
        seq_cep95 = float("nan") if row["sequence_cep95_m"] is None else float(row["sequence_cep95_m"])
        pf_applied = float("nan") if row["directional_applied_updates"] is None else float(row["directional_applied_updates"])
        seq_applied = float("nan") if row["sequence_applied_updates"] is None else float(row["sequence_applied_updates"])
        print(
            f"{row['label']:<28} {row['horizontal_rmse_m']:10.3f} {pf_rmse:10.3f} {seq_rmse:10.3f} "
            f"{row['cep95_m']:10.3f} {pf_cep95:10.3f} {seq_cep95:10.3f} {hmi_pct:10.3f} {pf_applied:10.0f} {seq_applied:10.0f}"
        )
    print(f"\nSaved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
