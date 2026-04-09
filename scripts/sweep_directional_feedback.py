#!/usr/bin/env python3
"""Sweep directional feedback gate configs on maritime_baseline and report.

Monkey-patches the single-scenario runner to inject a custom
DirectionalFeedbackSpec, runs the scenario under each config, and tabulates
the resulting horizontal navigation metrics.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for p in (str(SRC_DIR), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_single_scenario as runner_script  # type: ignore
from gravnav.estimators.feedback_policy import DirectionalFeedbackSpec


def run_one(label: str, df_spec: DirectionalFeedbackSpec | None, seed: int = 42):
    use_directional = df_spec is not None
    argv = [
        "run_single_scenario.py",
        "--scenario", "maritime_baseline",
        "--seed", str(seed),
        "--run-id", label,
        "--output-dir", "/tmp/bench_sweep",
    ]
    if use_directional:
        argv.append("--use-directional-feedback")

    orig_argv = sys.argv
    orig_build = runner_script._build_runner_config
    try:
        sys.argv = argv
        if use_directional:
            def patched(a):
                c = orig_build(a)
                c.map_match.directional_feedback_spec = df_spec
                return c
            runner_script._build_runner_config = patched
        rc = runner_script.main()
        assert rc == 0
    finally:
        sys.argv = orig_argv
        runner_script._build_runner_config = orig_build

    metrics_path = Path("/tmp/bench_sweep") / f"maritime_baseline_{label}_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    p = metrics["ins_position_error"]
    return {
        "label": label,
        "horizontal_rmse_m": p["horizontal_rmse_m"],
        "horizontal_cep95_m": p["cep95_m"],
        "horizontal_max_m": p["horizontal_max_m"],
        "north_rmse": p["north"]["rmse"],
        "east_rmse": p["east"]["rmse"],
        "vertical_rmse": p["down"]["rmse"],
        "hmi_horizontal": metrics["integrity"]["fraction_hazardously_misleading_horizontal"],
    }


def main():
    configs: dict[str, DirectionalFeedbackSpec | None] = {
        "baseline_observe": None,
        "A_soft_nudge": DirectionalFeedbackSpec(
            horizontal_only=True,
            min_eigenvalue_ratio=4.0,
            max_correction_norm_m=3.0,
            base_inflation=10.0,
            adaptive_inflation=True,
        ),
        "B_very_gentle": DirectionalFeedbackSpec(
            horizontal_only=True,
            min_eigenvalue_ratio=4.0,
            max_correction_norm_m=1.5,
            base_inflation=20.0,
            adaptive_inflation=True,
        ),
        "C_strict_gates": DirectionalFeedbackSpec(
            horizontal_only=True,
            min_eigenvalue_ratio=8.0,
            max_correction_norm_m=2.0,
            base_inflation=15.0,
            adaptive_inflation=True,
            persistence_count=3,
        ),
        "D_tiny_step": DirectionalFeedbackSpec(
            horizontal_only=True,
            min_eigenvalue_ratio=4.0,
            max_correction_norm_m=0.5,
            base_inflation=30.0,
            adaptive_inflation=True,
        ),
    }
    results = []
    for label, spec in configs.items():
        print(f"\n=== {label} ===", flush=True)
        r = run_one(label, spec)
        results.append(r)
        print(r, flush=True)

    print("\n\n=== SUMMARY (maritime_baseline seed=42) ===")
    print(f"{'label':<18} {'H_RMSE':>8} {'CEP95':>8} {'Hmax':>8} {'N':>7} {'E':>7} {'V':>6} {'HMI%':>7}")
    for r in results:
        print(
            f"{r['label']:<18} {r['horizontal_rmse_m']:8.1f} {r['horizontal_cep95_m']:8.1f} "
            f"{r['horizontal_max_m']:8.1f} {r['north_rmse']:7.1f} {r['east_rmse']:7.1f} "
            f"{r['vertical_rmse']:6.2f} {r['hmi_horizontal']*100:7.2f}"
        )


if __name__ == "__main__":
    main()
