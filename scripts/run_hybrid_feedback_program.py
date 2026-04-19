#!/usr/bin/env python3
"""
Run the hybrid ML-gated sequence-feedback benchmark program end to end.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

BUILD_CORPUS_SCRIPT = PROJECT_ROOT / "scripts" / "build_sequence_feedback_corpus.py"
TRAIN_TRUST_SCRIPT = PROJECT_ROOT / "scripts" / "train_sequence_feedback_trust_model.py"
BENCHMARK_SCRIPT = PROJECT_ROOT / "scripts" / "benchmark_filters.py"

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "outputs" / "reports" / "hybrid_feedback_program"
)
DEFAULT_FEEDBACK_MANIFESTS = (
    PROJECT_ROOT / "data/bathymetry/processed/norwegian_margin_maritime_demo_pack.json",
    PROJECT_ROOT / "data/bathymetry/processed/helgeland_offshore_demo_pack.json",
    PROJECT_ROOT / "data/bathymetry/processed/mid_atlantic_ridge_public_demo_pack.json",
)


@dataclass(frozen=True)
class BenchmarkTarget:
    name: str
    scenario_path: Path
    demo_pack_manifest: Path
    acceptance_mode: str


DEFAULT_BENCHMARK_TARGETS = (
    BenchmarkTarget(
        name="norwegian_margin_maritime",
        scenario_path=PROJECT_ROOT / "configs/scenarios/norwegian_margin_maritime.json",
        demo_pack_manifest=PROJECT_ROOT
        / "data/bathymetry/processed/norwegian_margin_maritime_demo_pack.json",
        acceptance_mode="beat_live",
    ),
    BenchmarkTarget(
        name="helgeland_offshore",
        scenario_path=PROJECT_ROOT / "configs/scenarios/helgeland_offshore_maritime.json",
        demo_pack_manifest=PROJECT_ROOT
        / "data/bathymetry/processed/helgeland_offshore_demo_pack.json",
        acceptance_mode="beat_live",
    ),
    BenchmarkTarget(
        name="mid_atlantic_ridge",
        scenario_path=PROJECT_ROOT
        / "configs/scenarios/mid_atlantic_ridge_public_demo_maritime.json",
        demo_pack_manifest=PROJECT_ROOT
        / "data/bathymetry/processed/mid_atlantic_ridge_public_demo_pack.json",
        acceptance_mode="nonregress_live_5pct",
    ),
)
TARGET_BY_NAME = {target.name: target for target in DEFAULT_BENCHMARK_TARGETS}
HYBRID_BENCHMARK_LABELS = (
    "live_ins",
    "sequence_lag_smoothed",
    "sequence_replay_heuristic",
    "sequence_replay_trust_gated",
    "sequence_bias_transfer_heuristic",
    "sequence_bias_transfer_trust_gated",
)


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _relative_to_root(path: str | Path) -> str:
    resolved = _resolve_path(path)
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _float_values(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        out.append(float(value))
    return out


def primary_metrics_from_row(row: dict[str, Any]) -> dict[str, float | None]:
    label = str(row["label"])
    if label == "sequence_lag_smoothed":
        return {
            "horizontal_rmse_m": (
                None
                if row.get("lag_smoothed_horizontal_rmse_m") is None
                else float(row["lag_smoothed_horizontal_rmse_m"])
            ),
            "cep95_m": (
                None
                if row.get("lag_smoothed_cep95_m") is None
                else float(row["lag_smoothed_cep95_m"])
            ),
            "hmi_horizontal": (
                None
                if row.get("lag_hmi_horizontal") is None
                else float(row["lag_hmi_horizontal"])
            ),
        }
    return {
        "horizontal_rmse_m": (
            None if row.get("horizontal_rmse_m") is None else float(row["horizontal_rmse_m"])
        ),
        "cep95_m": None if row.get("cep95_m") is None else float(row["cep95_m"]),
        "hmi_horizontal": (
            None if row.get("hmi_horizontal") is None else float(row["hmi_horizontal"])
        ),
    }


def aggregate_benchmark_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row["label"]), []).append(row)

    summary: dict[str, dict[str, Any]] = {}
    for label, label_rows in by_label.items():
        primary = [primary_metrics_from_row(row) for row in label_rows]
        rmse_values = _float_values([metrics["horizontal_rmse_m"] for metrics in primary])
        cep95_values = _float_values([metrics["cep95_m"] for metrics in primary])
        hmi_values = _float_values([metrics["hmi_horizontal"] for metrics in primary])
        applied_updates = _float_values(
            [row.get("sequence_applied_updates") for row in label_rows]
        )
        allowed_updates = _float_values(
            [row.get("sequence_allowed_updates") for row in label_rows]
        )
        trust_positive_updates = _float_values(
            [row.get("sequence_trust_positive_updates") for row in label_rows]
        )

        summary[label] = {
            "label": label,
            "num_rows": int(len(label_rows)),
            "median_horizontal_rmse_m": _median_or_none(rmse_values),
            "median_cep95_m": _median_or_none(cep95_values),
            "median_hmi_horizontal": _median_or_none(hmi_values),
            "hmi_zero_all_rows": bool(
                hmi_values and all(abs(value) <= 1.0e-12 for value in hmi_values)
            ),
            "median_sequence_applied_updates": _median_or_none(applied_updates),
            "total_sequence_applied_updates": int(sum(applied_updates)),
            "median_sequence_allowed_updates": _median_or_none(allowed_updates),
            "total_sequence_allowed_updates": int(sum(allowed_updates)),
            "median_sequence_trust_positive_updates": _median_or_none(
                trust_positive_updates
            ),
            "total_sequence_trust_positive_updates": int(sum(trust_positive_updates)),
            "per_seed_rows": label_rows,
        }
    return summary


def evaluate_target_acceptance(
    target: BenchmarkTarget,
    label_summary: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if "live_ins" not in label_summary:
        raise KeyError(f"Target {target.name!r} is missing the live_ins row.")
    if "sequence_replay_trust_gated" not in label_summary:
        raise KeyError(
            f"Target {target.name!r} is missing the sequence_replay_trust_gated row."
        )

    live = label_summary["live_ins"]
    trust = label_summary["sequence_replay_trust_gated"]
    live_rmse = live["median_horizontal_rmse_m"]
    live_cep95 = live["median_cep95_m"]
    trust_rmse = trust["median_horizontal_rmse_m"]
    trust_cep95 = trust["median_cep95_m"]

    beats_live = bool(
        live_rmse is not None
        and live_cep95 is not None
        and trust_rmse is not None
        and trust_cep95 is not None
        and trust_rmse < live_rmse
        and trust_cep95 < live_cep95
    )
    within_live_5pct = bool(
        live_rmse is not None
        and trust_rmse is not None
        and trust_rmse <= 1.05 * live_rmse
    )
    zero_hmi = bool(trust["hmi_zero_all_rows"])
    applied_nonzero = int(trust["total_sequence_applied_updates"]) > 0

    if target.acceptance_mode == "beat_live":
        accepted = bool(beats_live and zero_hmi and applied_nonzero)
    elif target.acceptance_mode == "nonregress_live_5pct":
        accepted = bool(within_live_5pct and zero_hmi)
    else:
        raise ValueError(
            f"Unsupported acceptance mode {target.acceptance_mode!r} for {target.name!r}."
        )

    return {
        "target_name": target.name,
        "acceptance_mode": target.acceptance_mode,
        "accepted": accepted,
        "beats_live": beats_live,
        "within_live_5pct": within_live_5pct,
        "zero_hmi_all_rows": zero_hmi,
        "applied_updates_nonzero": applied_nonzero,
        "live_ins_median_horizontal_rmse_m": live_rmse,
        "live_ins_median_cep95_m": live_cep95,
        "trust_gated_median_horizontal_rmse_m": trust_rmse,
        "trust_gated_median_cep95_m": trust_cep95,
        "trust_gated_total_sequence_applied_updates": int(
            trust["total_sequence_applied_updates"]
        ),
    }


def _write_markdown_report(
    path: Path,
    *,
    summary: dict[str, Any],
) -> None:
    scenario_summaries = summary["benchmark_summaries"]
    target_acceptance = summary["target_acceptance"]
    lines = [
        "# Hybrid Feedback Program Report",
        "",
        "## Decision",
        "",
        f"- accepted: `{summary['accepted']}`",
        f"- primary targets passed: `{summary['primary_targets_passed']}`",
        f"- external target passed: `{summary['external_target_passed']}`",
        "",
        "## Acceptance Table",
        "",
        "| Target | Mode | Accepted | Beat Live INS | Within 5% of Live INS | Zero HMI | Applied Updates > 0 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for acceptance in target_acceptance:
        lines.append(
            f"| `{acceptance['target_name']}` | "
            f"`{acceptance['acceptance_mode']}` | "
            f"`{acceptance['accepted']}` | "
            f"`{acceptance['beats_live']}` | "
            f"`{acceptance['within_live_5pct']}` | "
            f"`{acceptance['zero_hmi_all_rows']}` | "
            f"`{acceptance['applied_updates_nonzero']}` |"
        )

    lines.extend(
        [
            "",
            "## Scenario Medians",
            "",
            "| Scenario | Label | Median RMSE [m] | Median CEP95 [m] | HMI Zero All | Total Applied Updates |",
            "| --- | --- | ---: | ---: | --- | ---: |",
        ]
    )
    for scenario in scenario_summaries:
        label_summary = scenario["label_summary"]
        for label in (
            "live_ins",
            "sequence_lag_smoothed",
            "sequence_replay_heuristic",
            "sequence_replay_trust_gated",
            "sequence_bias_transfer_heuristic",
            "sequence_bias_transfer_trust_gated",
        ):
            if label not in label_summary:
                continue
            row = label_summary[label]
            rmse = row["median_horizontal_rmse_m"]
            cep95 = row["median_cep95_m"]
            lines.append(
                f"| `{scenario['target_name']}` | "
                f"`{label}` | "
                f"{'nan' if rmse is None else f'{rmse:.3f}'} | "
                f"{'nan' if cep95 is None else f'{cep95:.3f}'} | "
                f"`{row['hmi_zero_all_rows']}` | "
                f"{int(row['total_sequence_applied_updates'])} |"
            )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- feedback corpus: `{summary['feedback_corpus_path']}`",
            f"- trust model: `{summary['trust_model_output_path']}`",
            f"- trust model summary: `{summary['trust_model_summary_path']}`",
            f"- trust model CV summary: `{summary['trust_model_cv_summary_path']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def find_benchmark_summary_file(seed_dir: Path) -> Path:
    suffixes = tuple(f"_{label}_summary.json" for label in HYBRID_BENCHMARK_LABELS)
    candidates = sorted(
        path
        for path in seed_dir.glob("*_summary.json")
        if not path.name.endswith(suffixes)
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one aggregate benchmark summary in {seed_dir}, found {candidates}."
        )
    return candidates[0]


def _run_benchmark_one_seed(
    *,
    target: BenchmarkTarget,
    seed: int,
    output_dir: Path,
    trust_model_path: Path,
    min_trust_probability: float,
    max_predicted_error_delta_m: float | None,
    min_projected_std_m: float | None,
    apply_trust_gain_alpha: bool,
    trust_gain_alpha_min: float,
    trust_gain_alpha_max: float,
    fixed_gain_alpha_override: float | None,
    dt_s: float | None,
) -> Path:
    seed_dir = output_dir / target.name / f"seed_{int(seed)}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        str(BENCHMARK_SCRIPT),
        "--profile",
        "hybrid_feedback",
        "--scenario",
        str(target.scenario_path),
        "--demo-pack-manifest",
        str(target.demo_pack_manifest),
        "--output-dir",
        str(seed_dir),
        "--seed",
        str(int(seed)),
        "--sequence-feedback-trust-model-path",
        str(trust_model_path),
        "--sequence-feedback-min-trust-probability",
        str(float(min_trust_probability)),
    ]
    if max_predicted_error_delta_m is not None:
        cmd.extend(
            [
                "--sequence-feedback-max-predicted-error-delta-m",
                str(float(max_predicted_error_delta_m)),
            ]
        )
    if min_projected_std_m is not None:
        cmd.extend(
            [
                "--sequence-feedback-min-projected-std-m",
                str(float(min_projected_std_m)),
            ]
        )
    if apply_trust_gain_alpha:
        cmd.extend(
            [
                "--sequence-feedback-apply-trust-gain-alpha",
                "--sequence-feedback-trust-gain-alpha-min",
                str(float(trust_gain_alpha_min)),
                "--sequence-feedback-trust-gain-alpha-max",
                str(float(trust_gain_alpha_max)),
            ]
        )
    if fixed_gain_alpha_override is not None:
        cmd.extend(
            [
                "--sequence-feedback-fixed-gain-alpha-override",
                str(float(fixed_gain_alpha_override)),
            ]
        )
    if dt_s is not None:
        cmd.extend(["--dt-s", str(float(dt_s))])
    _run_command(cmd)
    return find_benchmark_summary_file(seed_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the hybrid sequence-feedback corpus, train the trust model, "
            "and benchmark trust-gated lag replay across the target scenarios."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where corpus, trust-model, and benchmark artifacts are written.",
    )
    parser.add_argument(
        "--feedback-corpus-path",
        default=None,
        help="Optional output NPZ path for the feedback corpus.",
    )
    parser.add_argument(
        "--trust-model-output-path",
        default=None,
        help="Optional output NPZ path for the trust model.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional JSON summary path. Defaults under --output-dir.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional markdown report path. Defaults under --output-dir.",
    )
    parser.add_argument(
        "--feedback-demo-pack-manifests",
        nargs="+",
        default=[_relative_to_root(path) for path in DEFAULT_FEEDBACK_MANIFESTS],
        help="Demo-pack manifests used to build the feedback-training corpus.",
    )
    parser.add_argument(
        "--benchmark-targets",
        nargs="+",
        choices=sorted(TARGET_BY_NAME.keys()),
        default=[target.name for target in DEFAULT_BENCHMARK_TARGETS],
        help="Named benchmark targets to run.",
    )
    parser.add_argument(
        "--corpus-seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Seeds used when building the feedback corpus.",
    )
    parser.add_argument(
        "--benchmark-seeds",
        nargs="+",
        type=int,
        default=[42, 123, 777],
        help="Seeds used for the mission-facing hybrid benchmarks.",
    )
    parser.add_argument(
        "--dt-s",
        type=float,
        default=None,
        help="Optional dt override passed to both corpus-build and benchmark runs.",
    )
    parser.add_argument("--horizon-s", type=float, default=60.0)
    parser.add_argument("--max-events-per-region-seed", type=int, default=64)
    parser.add_argument(
        "--minimum-useful-improvement-m",
        type=float,
        default=10.0,
        help=(
            "Minimum forward-horizon RMSE improvement required before a replay "
            "candidate is labeled useful in the feedback corpus."
        ),
    )
    parser.add_argument(
        "--gain-alpha-candidates",
        nargs="+",
        type=float,
        default=[0.1, 0.25, 0.5, 0.75, 1.0],
        help="Candidate correction gains swept when labeling the best safe replay gain.",
    )
    parser.add_argument("--l2", type=float, default=1.0e-2)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--trust-threshold", type=float, default=0.5)
    parser.add_argument("--gain-alpha-l2", type=float, default=1.0e-2)
    parser.add_argument(
        "--covariance-scale-reference-error-m",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--sequence-feedback-min-trust-probability",
        type=float,
        default=0.5,
        help="Trust threshold applied during the live hybrid-feedback benchmarks.",
    )
    parser.add_argument(
        "--sequence-feedback-max-predicted-error-delta-m",
        type=float,
        default=0.0,
        help=(
            "Maximum trust-model predicted replay error delta allowed during the "
            "live hybrid-feedback benchmarks."
        ),
    )
    parser.add_argument(
        "--sequence-feedback-min-projected-std-m",
        type=float,
        default=1.0,
        help=(
            "Minimum projected replay standard deviation allowed during the live "
            "hybrid-feedback benchmarks."
        ),
    )
    parser.add_argument(
        "--sequence-feedback-apply-trust-gain-alpha",
        action="store_true",
        help="Apply trust-model gain alpha during the live hybrid-feedback benchmarks.",
    )
    parser.add_argument(
        "--sequence-feedback-trust-gain-alpha-min",
        type=float,
        default=0.0,
        help="Minimum trust-derived gain alpha used during the live benchmarks.",
    )
    parser.add_argument(
        "--sequence-feedback-trust-gain-alpha-max",
        type=float,
        default=1.0,
        help="Maximum trust-derived gain alpha used during the live benchmarks.",
    )
    parser.add_argument(
        "--sequence-feedback-fixed-gain-alpha-override",
        type=float,
        default=None,
        help=(
            "Optional fixed gain alpha override used during the live benchmarks. "
            "When set it overrides trust-derived gain prediction."
        ),
    )
    parser.add_argument(
        "--skip-corpus-build",
        action="store_true",
        help="Skip corpus build and reuse --feedback-corpus-path.",
    )
    parser.add_argument(
        "--skip-train-model",
        action="store_true",
        help="Skip trust-model training and reuse --trust-model-output-path.",
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Skip benchmark execution and only write build/train artifacts.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feedback_corpus_path = (
        output_dir / "sequence_feedback_corpus.npz"
        if args.feedback_corpus_path is None
        else _resolve_path(args.feedback_corpus_path)
    )
    trust_model_output_path = (
        output_dir / "sequence_feedback_trust_model.npz"
        if args.trust_model_output_path is None
        else _resolve_path(args.trust_model_output_path)
    )
    summary_path = (
        output_dir / "hybrid_feedback_program_summary.json"
        if args.summary_path is None
        else _resolve_path(args.summary_path)
    )
    report_path = (
        output_dir / "hybrid_feedback_program_report.md"
        if args.report_path is None
        else _resolve_path(args.report_path)
    )

    if not args.skip_corpus_build:
        cmd = [
            PYTHON,
            str(BUILD_CORPUS_SCRIPT),
            "--demo-pack-manifests",
            *[str(_resolve_path(path)) for path in args.feedback_demo_pack_manifests],
            "--output-path",
            str(feedback_corpus_path),
            "--horizon-s",
            str(float(args.horizon_s)),
            "--max-events-per-region-seed",
            str(int(args.max_events_per_region_seed)),
            "--minimum-useful-improvement-m",
            str(float(args.minimum_useful_improvement_m)),
            "--gain-alpha-candidates",
            *[str(float(alpha)) for alpha in args.gain_alpha_candidates],
            "--seeds",
            *[str(int(seed)) for seed in args.corpus_seeds],
        ]
        if args.dt_s is not None:
            cmd.extend(["--dt-s", str(float(args.dt_s))])
        _run_command(cmd)

    if not args.skip_train_model:
        cmd = [
            PYTHON,
            str(TRAIN_TRUST_SCRIPT),
            "--corpus-path",
            str(feedback_corpus_path),
            "--model-output-path",
            str(trust_model_output_path),
            "--l2",
            str(float(args.l2)),
            "--learning-rate",
            str(float(args.learning_rate)),
            "--max-iter",
            str(int(args.max_iter)),
            "--trust-threshold",
            str(float(args.trust_threshold)),
            "--gain-alpha-l2",
            str(float(args.gain_alpha_l2)),
            "--covariance-scale-reference-error-m",
            str(float(args.covariance_scale_reference_error_m)),
        ]
        _run_command(cmd)

    benchmark_summaries: list[dict[str, Any]] = []
    target_acceptance: list[dict[str, Any]] = []
    if not args.skip_benchmarks:
        for target_name in args.benchmark_targets:
            target = TARGET_BY_NAME[str(target_name)]
            per_seed_rows: list[dict[str, Any]] = []
            per_seed_summary_paths: list[str] = []
            for seed in args.benchmark_seeds:
                summary_file = _run_benchmark_one_seed(
                    target=target,
                    seed=int(seed),
                    output_dir=output_dir / "benchmarks",
                    trust_model_path=trust_model_output_path,
                    min_trust_probability=float(
                        args.sequence_feedback_min_trust_probability
                    ),
                    max_predicted_error_delta_m=(
                        None
                        if args.sequence_feedback_max_predicted_error_delta_m is None
                        else float(args.sequence_feedback_max_predicted_error_delta_m)
                    ),
                    min_projected_std_m=(
                        None
                        if args.sequence_feedback_min_projected_std_m is None
                        else float(args.sequence_feedback_min_projected_std_m)
                    ),
                    apply_trust_gain_alpha=bool(
                        args.sequence_feedback_apply_trust_gain_alpha
                    ),
                    trust_gain_alpha_min=float(
                        args.sequence_feedback_trust_gain_alpha_min
                    ),
                    trust_gain_alpha_max=float(
                        args.sequence_feedback_trust_gain_alpha_max
                    ),
                    fixed_gain_alpha_override=(
                        None
                        if args.sequence_feedback_fixed_gain_alpha_override is None
                        else float(args.sequence_feedback_fixed_gain_alpha_override)
                    ),
                    dt_s=(None if args.dt_s is None else float(args.dt_s)),
                )
                payload = json.loads(summary_file.read_text(encoding="utf-8"))
                per_seed_rows.extend(dict(row) for row in payload["runs"])
                per_seed_summary_paths.append(_relative_to_root(summary_file))

            label_summary = aggregate_benchmark_rows(per_seed_rows)
            acceptance = evaluate_target_acceptance(target, label_summary)
            benchmark_summaries.append(
                {
                    "target_name": target.name,
                    "scenario_path": _relative_to_root(target.scenario_path),
                    "demo_pack_manifest": _relative_to_root(target.demo_pack_manifest),
                    "per_seed_summary_paths": per_seed_summary_paths,
                    "label_summary": label_summary,
                }
            )
            target_acceptance.append(acceptance)

    primary_targets = [
        row
        for row in target_acceptance
        if row["acceptance_mode"] == "beat_live"
    ]
    external_targets = [
        row
        for row in target_acceptance
        if row["acceptance_mode"] == "nonregress_live_5pct"
    ]

    trust_model_summary_path = trust_model_output_path.with_name(
        trust_model_output_path.stem + "_summary.json"
    )
    trust_model_cv_summary_path = trust_model_output_path.with_name(
        trust_model_output_path.stem + "_cv_summary.json"
    )
    summary = {
        "entry_point": _relative_to_root(__file__),
        "output_dir": _relative_to_root(output_dir),
        "feedback_corpus_path": _relative_to_root(feedback_corpus_path),
        "trust_model_output_path": _relative_to_root(trust_model_output_path),
        "trust_model_summary_path": _relative_to_root(trust_model_summary_path),
        "trust_model_cv_summary_path": _relative_to_root(trust_model_cv_summary_path),
        "benchmark_summaries": benchmark_summaries,
        "target_acceptance": target_acceptance,
        "primary_targets_passed": bool(
            primary_targets and all(bool(row["accepted"]) for row in primary_targets)
        ),
        "external_target_passed": bool(
            not external_targets
            or all(bool(row["accepted"]) for row in external_targets)
        ),
        "accepted": bool(
            (primary_targets and all(bool(row["accepted"]) for row in primary_targets))
            and (
                not external_targets
                or all(bool(row["accepted"]) for row in external_targets)
            )
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_markdown_report(report_path, summary=summary)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
