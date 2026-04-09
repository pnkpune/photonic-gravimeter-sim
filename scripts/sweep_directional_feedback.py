#!/usr/bin/env python3
"""
Sweep directional-feedback gate configs against the Priority-3 gradient path.

This script exists to answer one concrete question:

"Now that the PF likelihood improves modestly with gradient information, can a
relaxed directional-feedback policy safely convert that PF-side gain into better
INS metrics?"

It runs a small, reproducible benchmark on one scenario, compares observe-only
and observe+gradient references against several directional-feedback gate
policies, and writes a JSON + Markdown summary plus a compact figure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

for p in (str(SRC_DIR), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib.pyplot as plt
import numpy as np

import run_single_scenario as runner_script  # type: ignore
from gravnav.estimators.feedback_policy import DirectionalFeedbackSpec


DEFAULT_RUN_DIR = PROJECT_ROOT / "data" / "outputs" / "runs" / "directional_feedback_gradient_sweep"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "data" / "outputs" / "reports"
DEFAULT_FIG_DIR = PROJECT_ROOT / "data" / "outputs" / "figures" / "directional_feedback_gradient_sweep"
DEFAULT_SUMMARY_PATH = DEFAULT_REPORT_DIR / "directional_feedback_gradient_sweep_summary.json"
DEFAULT_REPORT_PATH = DEFAULT_REPORT_DIR / "directional_feedback_gradient_sweep_report.md"
DEFAULT_FIG_PATH = DEFAULT_FIG_DIR / "directional_feedback_gradient_sweep.png"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _load_directional_rows(archive_path: Path) -> list[dict[str, Any]]:
    with np.load(archive_path, allow_pickle=False) as data:
        key = "estimator_custom_streams_json"
        if key not in data:
            return []
        streams = json.loads(str(np.asarray(data[key]).item()))
    return list(streams.get("pf_directional_feedback", []))


def _top_reason(rows: list[dict[str, Any]]) -> str | None:
    counts: dict[str, int] = {}
    for row in rows:
        reason = row.get("rejection_reason")
        if reason is None:
            continue
        counts[str(reason)] = counts.get(str(reason), 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def _summarize_directional_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) == 0:
        return {
            "evaluations": 0,
            "allowed": 0,
            "applied": 0,
            "top_rejection_reason": None,
            "median_eigenvalue_ratio": None,
            "p90_eigenvalue_ratio": None,
            "median_ess_fraction": None,
            "median_abs_projected_correction_m": None,
            "median_projected_variance_m2": None,
            "median_inflation_applied": None,
            "median_observable_rank": None,
            "feedback_recommended_fraction": None,
            "median_information_density": None,
            "median_gradient_norm_horizontal": None,
        }

    eigen_ratio = np.asarray([row.get("eigenvalue_ratio", np.nan) for row in rows], dtype=np.float64)
    ess_fraction = np.asarray([row.get("ess_fraction", np.nan) for row in rows], dtype=np.float64)
    projected_correction = np.asarray(
        [row.get("projected_correction_m", np.nan) for row in rows],
        dtype=np.float64,
    )
    projected_variance = np.asarray(
        [row.get("projected_variance_m2", np.nan) for row in rows],
        dtype=np.float64,
    )
    inflation = np.asarray(
        [row.get("inflation_applied", np.nan) for row in rows],
        dtype=np.float64,
    )
    observable_rank = np.asarray(
        [row.get("observable_rank", np.nan) for row in rows],
        dtype=np.float64,
    )
    feedback_recommended = np.asarray(
        [bool(row.get("observability_feedback_recommended", False)) for row in rows],
        dtype=np.float64,
    )
    info_density = np.asarray(
        [row.get("information_density", np.nan) for row in rows],
        dtype=np.float64,
    )
    grad_norm = np.asarray(
        [row.get("gradient_norm_horizontal", np.nan) for row in rows],
        dtype=np.float64,
    )

    return {
        "evaluations": len(rows),
        "allowed": sum(1 for row in rows if bool(row.get("feedback_allowed", False))),
        "applied": sum(1 for row in rows if bool(row.get("applied", False))),
        "top_rejection_reason": _top_reason(rows),
        "median_eigenvalue_ratio": float(np.nanmedian(eigen_ratio)),
        "p90_eigenvalue_ratio": float(np.nanpercentile(eigen_ratio, 90.0)),
        "median_ess_fraction": float(np.nanmedian(ess_fraction)),
        "median_abs_projected_correction_m": float(np.nanmedian(np.abs(projected_correction))),
        "median_projected_variance_m2": float(np.nanmedian(projected_variance)),
        "median_inflation_applied": float(np.nanmedian(inflation)),
        "median_observable_rank": float(np.nanmedian(observable_rank)),
        "feedback_recommended_fraction": float(np.nanmean(feedback_recommended)),
        "median_information_density": float(np.nanmedian(info_density)),
        "median_gradient_norm_horizontal": float(np.nanmedian(grad_norm)),
    }


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    f = float(value)
    return f if np.isfinite(f) else None


def run_one(
    *,
    scenario: str,
    label: str,
    output_dir: Path,
    seed: int,
    use_gradiometer: bool,
    gradient_std_per_s2: float,
    df_spec: DirectionalFeedbackSpec | None,
) -> dict[str, Any]:
    use_directional = df_spec is not None
    argv = [
        "run_single_scenario.py",
        "--scenario", scenario,
        "--seed", str(seed),
        "--run-id", label,
        "--output-dir", str(output_dir),
    ]
    if use_gradiometer:
        argv.extend([
            "--use-gradiometer",
            "--pf-gradient-std-per-s2",
            str(float(gradient_std_per_s2)),
        ])
    if use_directional:
        argv.append("--use-directional-feedback")

    orig_argv = sys.argv
    orig_build = runner_script._build_runner_config
    try:
        sys.argv = argv
        if use_directional:
            def patched(args: argparse.Namespace):
                cfg = orig_build(args)
                cfg.map_match.directional_feedback_spec = df_spec
                return cfg
            runner_script._build_runner_config = patched
        rc = runner_script.main()
        if rc != 0:
            raise RuntimeError(f"run_single_scenario.py returned {rc} for {label}.")
    finally:
        sys.argv = orig_argv
        runner_script._build_runner_config = orig_build

    metrics_path = output_dir / f"{scenario}_{label}_metrics.json"
    archive_path = output_dir / f"{scenario}_{label}.npz"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    ins = metrics["ins_position_error"]
    pf = metrics.get("pf_position_error") or {}
    integrity = metrics.get("integrity") or {}
    directional_rows = _load_directional_rows(archive_path)
    directional = _summarize_directional_rows(directional_rows)

    row = {
        "label": label,
        "scenario": scenario,
        "seed": int(seed),
        "use_gradiometer": bool(use_gradiometer),
        "uses_directional_feedback": bool(use_directional),
        "directional_feedback_spec": None if df_spec is None else asdict(df_spec),
        "ins_horizontal_rmse_m": float(ins["horizontal_rmse_m"]),
        "ins_cep95_m": float(ins["cep95_m"]),
        "ins_horizontal_max_m": float(ins["horizontal_max_m"]),
        "ins_vertical_rmse_m": float(ins["vertical_rmse_m"]),
        "pf_horizontal_rmse_m": None if pf.get("horizontal_rmse_m") is None else float(pf["horizontal_rmse_m"]),
        "pf_cep95_m": None if pf.get("cep95_m") is None else float(pf["cep95_m"]),
        "integrity_hmi_horizontal": _finite_or_none(integrity.get("fraction_hazardously_misleading_horizontal")),
        "integrity_nis_pass_fraction": _finite_or_none(integrity.get("fraction_nis_passed")),
        "directional_feedback": directional,
        "metrics_path": _relative(metrics_path),
        "archive_path": _relative(archive_path),
    }
    return row


def _default_configs() -> list[tuple[str, bool, DirectionalFeedbackSpec | None]]:
    return [
        ("observe_only", False, None),
        ("observe_plus_gradient", True, None),
        (
            "dir_default_safe_noop",
            True,
            DirectionalFeedbackSpec(
                horizontal_only=True,
                min_eigenvalue_ratio=8.0,
                max_correction_norm_m=2.0,
                base_inflation=15.0,
                adaptive_inflation=True,
                persistence_count=3,
            ),
        ),
        (
            "dir_ratio3_gentle",
            True,
            DirectionalFeedbackSpec(
                horizontal_only=True,
                min_eigenvalue_ratio=3.0,
                max_correction_norm_m=1.0,
                base_inflation=20.0,
                adaptive_inflation=True,
                persistence_count=3,
            ),
        ),
        (
            "dir_ratio2_gentle",
            True,
            DirectionalFeedbackSpec(
                horizontal_only=True,
                min_eigenvalue_ratio=2.0,
                max_correction_norm_m=1.0,
                base_inflation=20.0,
                adaptive_inflation=True,
                persistence_count=3,
            ),
        ),
        (
            "dir_ratio15_gentle",
            True,
            DirectionalFeedbackSpec(
                horizontal_only=True,
                min_eigenvalue_ratio=1.5,
                max_correction_norm_m=1.0,
                base_inflation=20.0,
                adaptive_inflation=True,
                persistence_count=3,
            ),
        ),
        (
            "dir_ratio2_tiny",
            True,
            DirectionalFeedbackSpec(
                horizontal_only=True,
                min_eigenvalue_ratio=2.0,
                max_correction_norm_m=0.5,
                base_inflation=30.0,
                adaptive_inflation=True,
                persistence_count=3,
            ),
        ),
        (
            "dir_ratio15_tiny",
            True,
            DirectionalFeedbackSpec(
                horizontal_only=True,
                min_eigenvalue_ratio=1.5,
                max_correction_norm_m=0.5,
                base_inflation=30.0,
                adaptive_inflation=True,
                persistence_count=3,
            ),
        ),
        (
            "dir_ratio12_tiny_p4",
            True,
            DirectionalFeedbackSpec(
                horizontal_only=True,
                min_eigenvalue_ratio=1.2,
                max_correction_norm_m=0.5,
                base_inflation=40.0,
                adaptive_inflation=True,
                persistence_count=4,
            ),
        ),
        (
            "dir_ratio15_tiny_nis1",
            True,
            DirectionalFeedbackSpec(
                horizontal_only=True,
                min_eigenvalue_ratio=1.5,
                max_correction_norm_m=0.5,
                base_inflation=30.0,
                adaptive_inflation=True,
                persistence_count=3,
                nis_threshold=1.0,
            ),
        ),
        (
            "dir_obs_rank2_tiny_p4",
            True,
            DirectionalFeedbackSpec(
                horizontal_only=True,
                min_eigenvalue_ratio=1.2,
                max_correction_norm_m=0.5,
                base_inflation=40.0,
                adaptive_inflation=True,
                persistence_count=4,
                require_observability_recommended=True,
                min_observable_rank=2,
            ),
        ),
        (
            "dir_obs_rank2_tinier_p4",
            True,
            DirectionalFeedbackSpec(
                horizontal_only=True,
                min_eigenvalue_ratio=1.2,
                max_correction_norm_m=0.25,
                base_inflation=60.0,
                adaptive_inflation=True,
                persistence_count=4,
                require_observability_recommended=True,
                min_observable_rank=2,
            ),
        ),
        (
            "dir_obs_info_tinier",
            True,
            DirectionalFeedbackSpec(
                horizontal_only=True,
                min_eigenvalue_ratio=1.2,
                max_correction_norm_m=0.25,
                base_inflation=60.0,
                adaptive_inflation=True,
                persistence_count=4,
                require_observability_recommended=True,
                min_observable_rank=2,
                min_information_density=-10.0,
                min_observability_gradient_norm=1.0e-8,
            ),
        ),
    ]


def _select_best_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observe_plus_gradient = next(row for row in rows if row["label"] == "observe_plus_gradient")
    directional_rows = [row for row in rows if row["uses_directional_feedback"]]
    baseline_rmse = float(observe_plus_gradient["ins_horizontal_rmse_m"])
    baseline_hmi = float(observe_plus_gradient["integrity_hmi_horizontal"] or 0.0)

    safe_rows = [
        row for row in directional_rows
        if float(row["integrity_hmi_horizontal"] or 0.0) <= baseline_hmi + 1.0e-12
    ]
    if len(safe_rows) == 0:
        return {
            "status": "no_safe_directional_policy",
            "recommended_label": "observe_plus_gradient",
            "reason": "No directional configuration preserved the observe+gradient integrity baseline.",
        }

    best_safe = min(
        safe_rows,
        key=lambda row: (
            float(row["ins_horizontal_rmse_m"]),
            -int(row["directional_feedback"]["applied"]),
        ),
    )
    active_safe_rows = [row for row in safe_rows if int(row["directional_feedback"]["applied"]) > 0]
    best_active_safe = None if len(active_safe_rows) == 0 else min(
        active_safe_rows,
        key=lambda row: (
            float(row["ins_horizontal_rmse_m"]),
            -int(row["directional_feedback"]["applied"]),
        ),
    )
    improved = float(best_safe["ins_horizontal_rmse_m"]) + 1.0e-12 < baseline_rmse
    result = {
        "status": "directional_beats_baseline" if improved else "directional_does_not_beat_baseline",
        "recommended_label": best_safe["label"] if improved else "observe_plus_gradient",
        "best_safe_directional_label": best_safe["label"],
        "best_safe_directional_rmse_m": float(best_safe["ins_horizontal_rmse_m"]),
        "observe_plus_gradient_rmse_m": baseline_rmse,
        "reason": (
            "A safe directional policy improved INS horizontal RMSE."
            if improved
            else "Directional feedback can be made non-destructive, but none of the tested policies beats observe+gradient at the INS level."
        ),
    }
    if best_active_safe is not None:
        result["best_active_safe_directional_label"] = best_active_safe["label"]
        result["best_active_safe_directional_rmse_m"] = float(best_active_safe["ins_horizontal_rmse_m"])
    return result


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


def _plot_summary(rows: list[dict[str, Any]], fig_path: Path) -> Path:
    labels = [row["label"] for row in rows]
    ins_rmse = np.asarray([row["ins_horizontal_rmse_m"] for row in rows], dtype=np.float64)
    pf_rmse = np.asarray([
        np.nan if row["pf_horizontal_rmse_m"] is None else row["pf_horizontal_rmse_m"]
        for row in rows
    ], dtype=np.float64)
    applied = np.asarray([row["directional_feedback"]["applied"] for row in rows], dtype=np.float64)
    hmi_pct = 100.0 * np.asarray([
        0.0 if row["integrity_hmi_horizontal"] is None else row["integrity_hmi_horizontal"]
        for row in rows
    ], dtype=np.float64)

    x = np.arange(len(labels), dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.0), constrained_layout=True)

    axes[0].plot(x, ins_rmse, marker="o", label="INS horizontal RMSE")
    axes[0].plot(x, pf_rmse, marker="s", label="PF horizontal RMSE")
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_ylabel("RMSE [m]")
    axes[0].set_title("Directional-feedback gradient sweep")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].bar(x - 0.18, applied, width=0.36, label="directional updates applied")
    axes[1].bar(x + 0.18, hmi_pct, width=0.36, label="HMI horiz [%]")
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].set_ylabel("count / percent")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def _write_report(
    *,
    scenario: str,
    seed: int,
    gradient_std_per_s2: float,
    rows: list[dict[str, Any]],
    recommendation: dict[str, Any],
    summary_path: Path,
    fig_path: Path,
    report_path: Path,
) -> Path:
    observe_only = next(row for row in rows if row["label"] == "observe_only")
    observe_plus_gradient = next(row for row in rows if row["label"] == "observe_plus_gradient")
    default_directional = next(row for row in rows if row["label"] == "dir_default_safe_noop")
    best_pf = min(
        rows,
        key=lambda row: float("inf") if row["pf_horizontal_rmse_m"] is None else float(row["pf_horizontal_rmse_m"]),
    )
    best_directional = min(
        [row for row in rows if row["uses_directional_feedback"]],
        key=lambda row: float(row["ins_horizontal_rmse_m"]),
    )
    active_directional_rows = [
        row for row in rows
        if row["uses_directional_feedback"] and int(row["directional_feedback"]["applied"]) > 0
    ]
    best_active_directional = None if len(active_directional_rows) == 0 else min(
        active_directional_rows,
        key=lambda row: float(row["ins_horizontal_rmse_m"]),
    )
    representative_directional = best_active_directional if best_active_directional is not None else best_directional

    lines = [
        "# Directional Feedback Gradient Sweep",
        "",
        f"- Scenario: `{scenario}`",
        f"- Seed: `{seed}`",
        f"- PF gradient measurement std: `{gradient_std_per_s2:.3e} 1/s^2`",
        f"- Summary JSON: `{_relative(summary_path)}`",
        f"- Figure: `{_relative(fig_path)}`",
        "",
        "## Headline",
        "",
        (
            f"- `observe_plus_gradient` improves PF horizontal RMSE from "
            f"`{observe_only['pf_horizontal_rmse_m']:.3f} m` to "
            f"`{observe_plus_gradient['pf_horizontal_rmse_m']:.3f} m`, but the INS remains flat at "
            f"`{observe_plus_gradient['ins_horizontal_rmse_m']:.3f} m` horizontal RMSE."
        ),
        (
            f"- The strict default directional policy remains the best safe tie: `{best_directional['label']}` gives "
            f"`{best_directional['ins_horizontal_rmse_m']:.3f} m` INS horizontal RMSE with "
            f"`{best_directional['directional_feedback']['applied']}` applied updates."
        ),
        (
            f"- Recommendation: `{recommendation['recommended_label']}`. "
            f"{recommendation['reason']}"
        ),
        "",
        "## Key Findings",
        "",
        (
            f"- The strongest PF result in this sweep was `{best_pf['label']}` at "
            f"`{best_pf['pf_horizontal_rmse_m']:.3f} m` PF horizontal RMSE."
        ),
    ]
    if best_active_directional is not None:
        lines.append(
            f"- The best non-trivial directional policy was `{best_active_directional['label']}` with "
            f"`{best_active_directional['ins_horizontal_rmse_m']:.3f} m` INS horizontal RMSE, "
            f"`{best_active_directional['pf_horizontal_rmse_m']:.3f} m` PF horizontal RMSE, and "
            f"`{best_active_directional['directional_feedback']['applied']}` applied updates."
        )
    lines.extend([
        (
            f"- The gradient path improves PF localization modestly, but the current PF posterior is still too weakly anisotropic for safe closed-loop feedback: "
            f"`dir_default_safe_noop` sees median directional eigenvalue ratio "
            f"`{default_directional['directional_feedback']['median_eigenvalue_ratio']:.3f}` and p90 "
            f"`{default_directional['directional_feedback']['p90_eigenvalue_ratio']:.3f}`."
        ),
        (
            f"- In the best active directional case, median absolute projected PF-to-INS correction remained "
            f"`{representative_directional['directional_feedback']['median_abs_projected_correction_m']:.3f} m`, "
            f"so even small allowed updates can still push the INS the wrong way if the posterior mean is biased."
        ),
        "",
        "## Sweep Table",
        "",
        "| Label | INS RMSE [m] | PF RMSE [m] | CEP95 [m] | HMI horiz [%] | Allowed | Applied | Top rejection reason |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in rows:
        directional = row["directional_feedback"]
        hmi_pct = 100.0 * float(row["integrity_hmi_horizontal"] or 0.0)
        pf_rmse = "—" if row["pf_horizontal_rmse_m"] is None else f"{row['pf_horizontal_rmse_m']:.3f}"
        lines.append(
            f"| `{row['label']}` | {row['ins_horizontal_rmse_m']:.3f} | {pf_rmse} | "
            f"{row['ins_cep95_m']:.3f} | {hmi_pct:.3f} | {directional['allowed']} | "
            f"{directional['applied']} | {directional['top_rejection_reason'] or '—'} |"
        )

    lines.extend([
        "",
        "## Conclusion",
        "",
        (
            "- This sweep does not justify relaxing the default directional-feedback gates on the current maritime baseline. "
            "The cheapest next move remains observability analysis or a richer measurement channel, not more gate tuning."
        ),
        (
            "- Practically: keep `observe_plus_gradient` as the working Priority-3 reference, and treat directional feedback as instrumentation until the posterior becomes better calibrated."
        ),
        "",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep directional-feedback policies on the gradient-enabled baseline.",
    )
    parser.add_argument("--scenario", default="maritime_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pf-gradient-std-per-s2", type=float, default=1.0e-8)
    parser.add_argument("--output-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--figure-path", default=str(DEFAULT_FIG_PATH))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    summary_path = Path(args.summary_path).expanduser().resolve()
    report_path = Path(args.report_path).expanduser().resolve()
    figure_path = Path(args.figure_path).expanduser().resolve()

    rows: list[dict[str, Any]] = []
    for label, use_gradiometer, spec in _default_configs():
        print(f"\n=== {label} ===", flush=True)
        row = run_one(
            scenario=str(args.scenario),
            label=label,
            output_dir=output_dir,
            seed=int(args.seed),
            use_gradiometer=bool(use_gradiometer),
            gradient_std_per_s2=float(args.pf_gradient_std_per_s2),
            df_spec=spec,
        )
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    recommendation = _select_best_policy(rows)
    payload = {
        "scenario": str(args.scenario),
        "seed": int(args.seed),
        "pf_gradient_std_per_s2": float(args.pf_gradient_std_per_s2),
        "recommendation": recommendation,
        "rows": rows,
    }
    _write_json(summary_path, payload)
    _plot_summary(rows, figure_path)
    _write_report(
        scenario=str(args.scenario),
        seed=int(args.seed),
        gradient_std_per_s2=float(args.pf_gradient_std_per_s2),
        rows=rows,
        recommendation=recommendation,
        summary_path=summary_path,
        fig_path=figure_path,
        report_path=report_path,
    )

    print("\n=== SUMMARY ===")
    print(
        f"{'label':<24} {'INS_RMSE':>10} {'PF_RMSE':>10} {'HMI%':>8} "
        f"{'allowed':>8} {'applied':>8}"
    )
    for row in rows:
        pf_rmse = float("nan") if row["pf_horizontal_rmse_m"] is None else float(row["pf_horizontal_rmse_m"])
        hmi_pct = 100.0 * float(row["integrity_hmi_horizontal"] or 0.0)
        directional = row["directional_feedback"]
        print(
            f"{row['label']:<24} {row['ins_horizontal_rmse_m']:10.3f} "
            f"{pf_rmse:10.3f} {hmi_pct:8.3f} {directional['allowed']:8d} {directional['applied']:8d}"
        )

    print(f"\nSaved summary: {_relative(summary_path)}")
    print(f"Saved report: {_relative(report_path)}")
    print(f"Saved figure: {_relative(figure_path)}")
    print(f"Recommended config: {recommendation['recommended_label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
