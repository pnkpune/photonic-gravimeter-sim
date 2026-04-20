#!/usr/bin/env python3
"""Benchmark PF, sequence, and lag-smoothed map-matching modes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gravnav.utils.config import load_config_mapping

PYTHON = sys.executable
RUNNER = PROJECT_ROOT / "scripts" / "run_single_scenario.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "reports" / "priority3_benchmark"


def _scenario_output_name(scenario: str) -> str:
    text = str(scenario).strip()
    path = Path(text).expanduser()
    if path.exists() and path.is_file():
        try:
            mapping = load_config_mapping(path)
        except Exception:
            return path.stem
        name = mapping.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        return path.stem
    return text


def _run_one(
    label: str,
    *,
    output_dir: Path,
    seed: int,
    scenario: str,
    base_flags: list[str],
    extra_flags: list[str] | None = None,
) -> dict[str, Any]:
    cmd = [
        PYTHON,
        str(RUNNER),
        "--scenario",
        scenario,
        "--seed",
        str(seed),
        "--run-id",
        label,
        "--output-dir",
        str(output_dir),
        *base_flags,
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else src_path + os.pathsep + existing_pythonpath
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)

    scenario_name = _scenario_output_name(scenario)
    metrics_path = output_dir / f"{scenario_name}_{label}_metrics.json"
    archive_path = output_dir / f"{scenario_name}_{label}.npz"
    metrics = json.loads(metrics_path.read_text())
    pos = metrics["ins_position_error"]
    lag = metrics.get("lag_smoothed_position_error") or {}
    pf = metrics.get("pf_position_error") or {}
    seq = metrics.get("sequence_position_error") or {}
    integrity = metrics.get("integrity") or {}
    lag_integrity = metrics.get("lag_smoothed_integrity") or {}
    directional_applied = None
    directional_allowed = None
    sequence_applied = None
    sequence_allowed = None
    sequence_trust_allowed = None
    sequence_trust_positive = None
    sequence_runtime_budget_rejections = None
    sequence_runtime_budget_cooldown_rejections = None
    sequence_runtime_budget_max_update_rejections = None
    sequence_applied_alpha_values: list[float] = []

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
                sequence_trust_allowed = sum(
                    1 for row in seq_rows if row.get("trust_allowed") is True
                )
                sequence_trust_positive = sum(
                    1
                    for row in seq_rows
                    if row.get("trust_allowed") is True
                )
                sequence_runtime_budget_rejections = sum(
                    1 for row in seq_rows if row.get("runtime_budget_allowed") is False
                )
                sequence_runtime_budget_cooldown_rejections = sum(
                    1
                    for row in seq_rows
                    if row.get("runtime_budget_rejection_reason")
                    == "learned_gain_cooldown_active"
                )
                sequence_runtime_budget_max_update_rejections = sum(
                    1
                    for row in seq_rows
                    if row.get("runtime_budget_rejection_reason")
                    == "learned_gain_max_applied_updates_reached"
                )
                sequence_applied_alpha_values = [
                    float(row["gain_alpha_applied"])
                    for row in seq_rows
                    if row.get("applied")
                    and row.get("gain_alpha_applied") is not None
                ]

    return {
        "label": label,
        "horizontal_rmse_m": pos["horizontal_rmse_m"],
        "cep95_m": pos["cep95_m"],
        "horizontal_max_m": pos["horizontal_max_m"],
        "vertical_rmse_m": pos["vertical_rmse_m"],
        "lag_smoothed_horizontal_rmse_m": lag.get("horizontal_rmse_m"),
        "lag_smoothed_cep95_m": lag.get("cep95_m"),
        "lag_smoothed_vertical_rmse_m": lag.get("vertical_rmse_m"),
        "pf_horizontal_rmse_m": pf.get("horizontal_rmse_m"),
        "pf_cep95_m": pf.get("cep95_m"),
        "sequence_horizontal_rmse_m": seq.get("horizontal_rmse_m"),
        "sequence_cep95_m": seq.get("cep95_m"),
        "hmi_horizontal": integrity.get("fraction_hazardously_misleading_horizontal"),
        "lag_hmi_horizontal": lag_integrity.get("fraction_hazardously_misleading_horizontal"),
        "directional_allowed_updates": directional_allowed,
        "directional_applied_updates": directional_applied,
        "sequence_allowed_updates": sequence_allowed,
        "sequence_applied_updates": sequence_applied,
        "sequence_trust_allowed_updates": sequence_trust_allowed,
        "sequence_trust_positive_updates": sequence_trust_positive,
        "sequence_runtime_budget_rejections": sequence_runtime_budget_rejections,
        "sequence_runtime_budget_cooldown_rejections": (
            sequence_runtime_budget_cooldown_rejections
        ),
        "sequence_runtime_budget_max_update_rejections": (
            sequence_runtime_budget_max_update_rejections
        ),
        "sequence_median_gain_alpha_applied": (
            None
            if not sequence_applied_alpha_values
            else float(np.median(np.asarray(sequence_applied_alpha_values, dtype=np.float64)))
        ),
        "metrics_path": str(metrics_path),
    }


def _profile_configs(profile: str) -> list[dict[str, Any]]:
    if profile == "priority3_full":
        return [
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
                "label": "sequence_plus_gradient_lag_smoothed",
                "extra_flags": [
                    "--map-matcher",
                    "sequence",
                    "--use-gradiometer",
                    "--use-sequence-lag-smoother",
                ],
            },
            {
                "label": "sequence_plus_gradient_replay_feedback",
                "extra_flags": [
                    "--map-matcher",
                    "sequence",
                    "--use-gradiometer",
                    "--use-sequence-feedback",
                    "--sequence-feedback-mode",
                    "lag_replay",
                    "--sequence-feedback-geometry",
                    "directional_horizontal",
                ],
            },
            {
                "label": "sequence_plus_gradient_replay_feedback_relaxed",
                "extra_flags": [
                    "--map-matcher",
                    "sequence",
                    "--use-gradiometer",
                    "--use-sequence-feedback",
                    "--sequence-feedback-mode",
                    "lag_replay",
                    "--sequence-feedback-geometry",
                    "full_horizontal",
                    "--sequence-feedback-min-peak-prob",
                    "0.03",
                    "--sequence-feedback-max-horizontal-std-m",
                    "120",
                    "--sequence-feedback-max-correction-m",
                    "50",
                    "--sequence-feedback-inflation",
                    "10.0",
                ],
            },
            {
                "label": "sequence_plus_gradient_replay_directional_safe",
                "extra_flags": [
                    "--map-matcher",
                    "sequence",
                    "--use-gradiometer",
                    "--use-sequence-feedback",
                    "--sequence-feedback-mode",
                    "lag_replay",
                    "--sequence-feedback-geometry",
                    "directional_horizontal",
                    "--sequence-feedback-min-eigenvalue-ratio",
                    "1.15",
                    "--sequence-feedback-min-peak-prob",
                    "0.03",
                    "--sequence-feedback-max-horizontal-std-m",
                    "120",
                    "--sequence-feedback-max-correction-m",
                    "30",
                    "--sequence-feedback-inflation",
                    "10.0",
                ],
            },
            {
                "label": "sequence_plus_gradient_transfer_feedback",
                "extra_flags": [
                    "--map-matcher",
                    "sequence",
                    "--use-gradiometer",
                    "--use-sequence-feedback",
                    "--sequence-feedback-mode",
                    "bias_transfer",
                    "--sequence-feedback-geometry",
                    "directional_horizontal",
                ],
            },
        ]
    if profile == "regional_core":
        return [
            {"label": "ins_only", "extra_flags": ["--disable-map-match"]},
            {"label": "observe_only", "extra_flags": []},
            {"label": "observe_plus_gradient", "extra_flags": ["--use-gradiometer"]},
            {
                "label": "sequence_plus_gradient",
                "extra_flags": ["--map-matcher", "sequence", "--use-gradiometer"],
            },
            {
                "label": "sequence_plus_gradient_lag_smoothed",
                "extra_flags": [
                    "--map-matcher",
                    "sequence",
                    "--use-gradiometer",
                    "--use-sequence-lag-smoother",
                ],
            },
        ]
    if profile == "hybrid_feedback":
        return [
            {"label": "live_ins", "extra_flags": ["--disable-map-match"]},
            {
                "label": "sequence_replay_trust_only",
                "trust_row": True,
                "extra_flags": [
                    "--map-matcher",
                    "sequence",
                    "--use-gradiometer",
                    "--use-sequence-feedback",
                    "--sequence-feedback-mode",
                    "lag_replay",
                    "--sequence-feedback-geometry",
                    "directional_horizontal",
                    "--sequence-feedback-inflation",
                    "6.0",
                    "--sequence-feedback-trust-gate-source",
                    "both",
                    "--sequence-feedback-apply-trust-covariance-scale",
                ],
            },
            {
                "label": "sequence_replay_gain025",
                "trust_row": True,
                "fixed_gain_alpha_override": 0.25,
                "extra_flags": [
                    "--map-matcher",
                    "sequence",
                    "--use-gradiometer",
                    "--use-sequence-feedback",
                    "--sequence-feedback-mode",
                    "lag_replay",
                    "--sequence-feedback-geometry",
                    "directional_horizontal",
                    "--sequence-feedback-inflation",
                    "6.0",
                    "--sequence-feedback-trust-gate-source",
                    "both",
                    "--sequence-feedback-apply-trust-covariance-scale",
                ],
            },
            {
                "label": "sequence_replay_gain050",
                "trust_row": True,
                "fixed_gain_alpha_override": 0.50,
                "extra_flags": [
                    "--map-matcher",
                    "sequence",
                    "--use-gradiometer",
                    "--use-sequence-feedback",
                    "--sequence-feedback-mode",
                    "lag_replay",
                    "--sequence-feedback-geometry",
                    "directional_horizontal",
                    "--sequence-feedback-inflation",
                    "6.0",
                    "--sequence-feedback-trust-gate-source",
                    "both",
                    "--sequence-feedback-apply-trust-covariance-scale",
                ],
            },
            {
                "label": "sequence_replay_learned_gain",
                "trust_row": True,
                "apply_trust_gain_alpha": True,
                "learned_gain_cooldown_s": 600.0,
                "learned_gain_max_applied_updates": 1,
                "extra_flags": [
                    "--map-matcher",
                    "sequence",
                    "--use-gradiometer",
                    "--use-sequence-feedback",
                    "--sequence-feedback-mode",
                    "lag_replay",
                    "--sequence-feedback-geometry",
                    "directional_horizontal",
                    "--sequence-feedback-inflation",
                    "6.0",
                    "--sequence-feedback-trust-gate-source",
                    "both",
                    "--sequence-feedback-apply-trust-covariance-scale",
                ],
            },
        ]
    raise ValueError(f"Unsupported benchmark profile {profile!r}.")


def _hybrid_bias_transfer_configs() -> list[dict[str, Any]]:
    return [
        {
            "label": "sequence_bias_transfer_heuristic",
            "extra_flags": [
                "--map-matcher",
                "sequence",
                "--use-gradiometer",
                "--use-sequence-feedback",
                "--sequence-feedback-mode",
                "bias_transfer",
                "--sequence-feedback-geometry",
                "directional_horizontal",
            ],
        },
        {
            "label": "sequence_bias_transfer_trust_gated",
            "trust_row": True,
            "extra_flags": [
                "--map-matcher",
                "sequence",
                "--use-gradiometer",
                "--use-sequence-feedback",
                "--sequence-feedback-mode",
                "bias_transfer",
                "--sequence-feedback-geometry",
                "directional_horizontal",
                "--sequence-feedback-inflation",
                "6.0",
                "--sequence-feedback-trust-gate-source",
                "both",
                "--sequence-feedback-apply-trust-covariance-scale",
            ],
        },
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmark comparisons across map-matching modes.")
    parser.add_argument(
        "--profile",
        choices=("priority3_full", "regional_core", "hybrid_feedback"),
        default="priority3_full",
        help="Named benchmark configuration set.",
    )
    parser.add_argument(
        "--scenario",
        default="maritime_baseline",
        help="Scenario name or explicit scenario config path passed through to run_single_scenario.py.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Root RNG seed.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where per-run artifacts and the benchmark summary are written.",
    )
    parser.add_argument(
        "--map-path",
        default=None,
        help="Optional processed NPZ map passed through to the single-run CLI.",
    )
    parser.add_argument(
        "--regional-map",
        choices=("norwegian_margin",),
        default=None,
        help="Optional regional map name resolved through the dataset layer.",
    )
    parser.add_argument(
        "--demo-pack-manifest",
        default=None,
        help="Optional demo-pack manifest passed through to run_single_scenario.py.",
    )
    parser.add_argument(
        "--regional-map-raw-path",
        default=None,
        help="Optional raw CSV override used when building the regional map cache.",
    )
    parser.add_argument(
        "--regional-map-force-reprocess",
        action="store_true",
        help="Rebuild the processed regional map cache before running the benchmark.",
    )
    parser.add_argument(
        "--dt-s",
        type=float,
        default=None,
        help="Optional truth sample interval override passed through to the single-run CLI.",
    )
    parser.add_argument(
        "--pf-particles",
        type=int,
        default=None,
        help="Optional PF particle-count override passed through to the single-run CLI.",
    )
    parser.add_argument(
        "--sequence-feedback-trust-model-path",
        default=None,
        help="Optional trust-model NPZ used by hybrid_feedback trust-gated runs.",
    )
    parser.add_argument(
        "--sequence-feedback-min-trust-probability",
        type=float,
        default=0.70,
        help="Trust threshold passed through to trust-gated sequence feedback runs.",
    )
    parser.add_argument(
        "--sequence-feedback-apply-trust-gain-alpha",
        action="store_true",
        help="Apply trust-model gain alpha during trust-gated hybrid feedback runs.",
    )
    parser.add_argument(
        "--sequence-feedback-trust-gain-alpha-min",
        type=float,
        default=0.0,
        help="Minimum trust-derived gain alpha passed to trust-gated runs.",
    )
    parser.add_argument(
        "--sequence-feedback-trust-gain-alpha-max",
        type=float,
        default=1.0,
        help="Maximum trust-derived gain alpha passed to trust-gated runs.",
    )
    parser.add_argument(
        "--sequence-feedback-fixed-gain-alpha-override",
        type=float,
        default=None,
        help="Optional fixed gain alpha override passed to trust-gated sequence-feedback runs.",
    )
    parser.add_argument(
        "--sequence-feedback-learned-gain-cooldown-s",
        type=float,
        default=0.0,
        help=(
            "Optional learned-gain-only cooldown passed through to trust-gated "
            "sequence-feedback runs."
        ),
    )
    parser.add_argument(
        "--sequence-feedback-learned-gain-max-applied-updates",
        type=int,
        default=None,
        help=(
            "Optional learned-gain-only cap on accepted sequence-feedback updates "
            "passed through to trust-gated runs."
        ),
    )
    parser.add_argument(
        "--sequence-feedback-max-predicted-error-delta-m",
        type=float,
        default=0.0,
        help="Maximum predicted replay error delta passed to trust-gated hybrid rows.",
    )
    parser.add_argument(
        "--sequence-feedback-min-projected-std-m",
        type=float,
        default=1.0,
        help="Minimum projected replay standard deviation passed to trust-gated hybrid rows.",
    )
    parser.add_argument(
        "--include-bias-transfer-rows",
        action="store_true",
        help="Append heuristic and trust-gated bias-transfer rows after the primary hybrid rows.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.map_path is not None and args.regional_map is not None:
        parser.error("--map-path and --regional-map are mutually exclusive.")
    if args.demo_pack_manifest is not None and (
        args.map_path is not None or args.regional_map is not None
    ):
        parser.error(
            "--demo-pack-manifest cannot be combined with --map-path or --regional-map."
        )
    if (
        args.profile == "hybrid_feedback"
        and args.sequence_feedback_trust_model_path is None
    ):
        parser.error(
            "--sequence-feedback-trust-model-path is required with --profile hybrid_feedback."
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_flags: list[str] = []
    if args.map_path is not None:
        base_flags.extend(["--map-path", args.map_path])
    if args.regional_map is not None:
        base_flags.extend(["--regional-map", args.regional_map])
    if args.demo_pack_manifest is not None:
        base_flags.extend(["--demo-pack-manifest", args.demo_pack_manifest])
        base_flags.extend(
            ["--use-bathymetry", "--use-magnetics", "--use-current-correction"]
        )
    if args.regional_map_raw_path is not None:
        base_flags.extend(["--regional-map-raw-path", args.regional_map_raw_path])
    if args.regional_map_force_reprocess:
        base_flags.append("--regional-map-force-reprocess")
    if args.dt_s is not None:
        base_flags.extend(["--dt-s", str(args.dt_s)])
    if args.pf_particles is not None:
        base_flags.extend(["--pf-particles", str(args.pf_particles)])

    rows: list[dict[str, Any]] = []
    configs = _profile_configs(args.profile)
    if args.profile == "hybrid_feedback" and args.include_bias_transfer_rows:
        configs = [*configs, *_hybrid_bias_transfer_configs()]
    for cfg in configs:
        print(f"\n=== {cfg['label']} ===", flush=True)
        extra_flags = list(cfg["extra_flags"])
        if bool(cfg.get("trust_row")):
            extra_flags.extend(
                [
                    "--sequence-feedback-trust-model-path",
                    str(args.sequence_feedback_trust_model_path),
                    "--sequence-feedback-min-trust-probability",
                    str(args.sequence_feedback_min_trust_probability),
                    "--sequence-feedback-max-predicted-error-delta-m",
                    str(args.sequence_feedback_max_predicted_error_delta_m),
                    "--sequence-feedback-min-projected-std-m",
                    str(args.sequence_feedback_min_projected_std_m),
                ]
            )
            if bool(cfg.get("apply_trust_gain_alpha")) or args.sequence_feedback_apply_trust_gain_alpha:
                extra_flags.extend(
                    [
                        "--sequence-feedback-apply-trust-gain-alpha",
                        "--sequence-feedback-trust-gain-alpha-min",
                        str(args.sequence_feedback_trust_gain_alpha_min),
                        "--sequence-feedback-trust-gain-alpha-max",
                        str(args.sequence_feedback_trust_gain_alpha_max),
                    ]
                )
                learned_gain_cooldown_s = cfg.get(
                    "learned_gain_cooldown_s",
                    args.sequence_feedback_learned_gain_cooldown_s,
                )
                if float(learned_gain_cooldown_s) > 0.0:
                    extra_flags.extend(
                        [
                            "--sequence-feedback-learned-gain-cooldown-s",
                            str(float(learned_gain_cooldown_s)),
                        ]
                    )
                learned_gain_max_applied_updates = cfg.get(
                    "learned_gain_max_applied_updates",
                    args.sequence_feedback_learned_gain_max_applied_updates,
                )
                if learned_gain_max_applied_updates is not None:
                    extra_flags.extend(
                        [
                            "--sequence-feedback-learned-gain-max-applied-updates",
                            str(int(learned_gain_max_applied_updates)),
                        ]
                    )
            fixed_gain_alpha_override = cfg.get(
                "fixed_gain_alpha_override",
                args.sequence_feedback_fixed_gain_alpha_override,
            )
            if fixed_gain_alpha_override is not None:
                extra_flags.extend(
                    [
                        "--sequence-feedback-fixed-gain-alpha-override",
                        str(fixed_gain_alpha_override),
                    ]
                )
        row = _run_one(
            cfg["label"],
            output_dir=output_dir,
            seed=int(args.seed),
            scenario=args.scenario,
            base_flags=base_flags,
            extra_flags=extra_flags,
        )
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    summary_payload = {
        "profile": args.profile,
        "scenario": args.scenario,
        "scenario_output_name": _scenario_output_name(str(args.scenario)),
        "seed": int(args.seed),
        "regional_map": args.regional_map,
        "map_path": args.map_path,
        "runs": rows,
    }
    summary_path = output_dir / f"{_scenario_output_name(str(args.scenario))}_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")

    print("\n=== SUMMARY ===")
    print(
        f"{'label':<34} {'INS_RMSE':>10} {'LAG_RMSE':>10} {'PF_RMSE':>10} {'SEQ_RMSE':>10} "
        f"{'INS_CEP95':>10} {'LAG_CEP95':>10} {'PF_CEP95':>10} {'SEQ_CEP95':>10} "
        f"{'HMI%':>10} {'LAG_HMI%':>10} {'PF_APPLY':>10} {'SEQ_APPLY':>10}"
    )
    for row in rows:
        hmi = row["hmi_horizontal"]
        hmi_pct = float("nan") if hmi is None else 100.0 * float(hmi)
        lag_hmi = row["lag_hmi_horizontal"]
        lag_hmi_pct = float("nan") if lag_hmi is None else 100.0 * float(lag_hmi)
        lag_rmse = float("nan") if row["lag_smoothed_horizontal_rmse_m"] is None else float(row["lag_smoothed_horizontal_rmse_m"])
        lag_cep95 = float("nan") if row["lag_smoothed_cep95_m"] is None else float(row["lag_smoothed_cep95_m"])
        pf_rmse = float("nan") if row["pf_horizontal_rmse_m"] is None else float(row["pf_horizontal_rmse_m"])
        pf_cep95 = float("nan") if row["pf_cep95_m"] is None else float(row["pf_cep95_m"])
        seq_rmse = float("nan") if row["sequence_horizontal_rmse_m"] is None else float(row["sequence_horizontal_rmse_m"])
        seq_cep95 = float("nan") if row["sequence_cep95_m"] is None else float(row["sequence_cep95_m"])
        pf_applied = float("nan") if row["directional_applied_updates"] is None else float(row["directional_applied_updates"])
        seq_applied = float("nan") if row["sequence_applied_updates"] is None else float(row["sequence_applied_updates"])
        print(
            f"{row['label']:<34} {row['horizontal_rmse_m']:10.3f} {lag_rmse:10.3f} {pf_rmse:10.3f} {seq_rmse:10.3f} "
            f"{row['cep95_m']:10.3f} {lag_cep95:10.3f} {pf_cep95:10.3f} {seq_cep95:10.3f} "
            f"{hmi_pct:10.3f} {lag_hmi_pct:10.3f} {pf_applied:10.0f} {seq_applied:10.0f}"
        )
    print(f"\nSaved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
