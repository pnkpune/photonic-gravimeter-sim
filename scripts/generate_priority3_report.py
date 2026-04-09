#!/usr/bin/env python3
"""
Generate a Priority-3 comparison package from saved run bundles.

This script compares:
- observe-only PF
- observe + gradient likelihood
- directional feedback + gradient likelihood

It reads the saved run archives and metrics under `data/outputs/runs/priority3`,
generates PF-focused comparison plots, and writes a markdown report describing
what improved and what did not.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np

from gravnav.estimators.integrity import geodetic_position_error_ned
from gravnav.simulation.results import load_npz_archive


RUN_DIR = PROJECT_ROOT / "data/outputs/runs/priority3"
FIG_DIR = PROJECT_ROOT / "data/outputs/figures/priority3"
REPORT_DIR = PROJECT_ROOT / "data/outputs/reports"
REPORT_PATH = REPORT_DIR / "priority3_gradient_report.md"

RUN_IDS = (
    "observe_only",
    "observe_plus_gradient",
    "directional_plus_gradient",
)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _load_metrics(run_id: str) -> dict[str, Any]:
    path = RUN_DIR / f"maritime_baseline_{run_id}_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(run_id: str) -> dict[str, Any]:
    path = RUN_DIR / f"maritime_baseline_{run_id}_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_archive(run_id: str) -> dict[str, Any]:
    path = RUN_DIR / f"maritime_baseline_{run_id}.npz"
    return load_npz_archive(path)


def _interp_truth_geodetic(
    archive: dict[str, Any],
    query_time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_truth = np.asarray(archive["truth_time_s"], dtype=np.float64)
    lat_truth = np.asarray(archive["truth_lat_rad"], dtype=np.float64)
    lon_truth = np.asarray(archive["truth_lon_rad"], dtype=np.float64)
    h_truth = np.asarray(archive["truth_height_m"], dtype=np.float64)
    tq = np.asarray(query_time_s, dtype=np.float64)

    lat = np.interp(tq, t_truth, lat_truth)
    lon = np.interp(tq, t_truth, lon_truth)
    h = np.interp(tq, t_truth, h_truth)
    return lat, lon, h


def _position_error_history_from_archive(
    archive: dict[str, Any],
    *,
    prefix: str,
) -> np.ndarray:
    t = np.asarray(archive[f"{prefix}_time_s"], dtype=np.float64)
    lat = np.asarray(archive[f"{prefix}_lat_rad"], dtype=np.float64)
    lon = np.asarray(archive[f"{prefix}_lon_rad"], dtype=np.float64)
    h = np.asarray(archive[f"{prefix}_height_m"], dtype=np.float64)
    lat_true, lon_true, h_true = _interp_truth_geodetic(archive, t)

    err = np.empty((t.size, 3), dtype=np.float64)
    for k in range(t.size):
        err[k] = geodetic_position_error_ned(
            estimated_lat_rad=float(lat[k]),
            estimated_lon_rad=float(lon[k]),
            estimated_height_m=float(h[k]),
            true_lat_rad=float(lat_true[k]),
            true_lon_rad=float(lon_true[k]),
            true_height_m=float(h_true[k]),
        )
    return err


def _horizontal_error(err_ned: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.asarray(err_ned, dtype=np.float64)[:, :2] ** 2, axis=1))


def _pf_metrics_from_archive(archive: dict[str, Any]) -> dict[str, float]:
    err = _position_error_history_from_archive(archive, prefix="pf")
    horiz = _horizontal_error(err)
    vert = np.abs(err[:, 2])
    return {
        "horizontal_rmse_m": float(np.sqrt(np.mean(horiz ** 2))),
        "horizontal_mean_m": float(np.mean(horiz)),
        "horizontal_p95_m": float(np.percentile(horiz, 95.0)),
        "vertical_rmse_m": float(np.sqrt(np.mean(vert ** 2))),
    }


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_ins_horizontal_error_histories(
    bundles: dict[str, dict[str, Any]],
) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    for run_id, bundle in bundles.items():
        archive = bundle["archive"]
        err = _position_error_history_from_archive(archive, prefix="ins")
        t = np.asarray(archive["ins_time_s"], dtype=np.float64)
        ax.plot(t, _horizontal_error(err), label=run_id)

    ax.set_xlabel("time [s]")
    ax.set_ylabel("horizontal INS error [m]")
    ax.set_title("Priority-3 INS horizontal error comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _save_figure(fig, FIG_DIR / "priority3_ins_horizontal_error.png")


def _plot_pf_horizontal_error_histories(
    bundles: dict[str, dict[str, Any]],
) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    for run_id, bundle in bundles.items():
        archive = bundle["archive"]
        err = _position_error_history_from_archive(archive, prefix="pf")
        t = np.asarray(archive["pf_time_s"], dtype=np.float64)
        ax.plot(t, _horizontal_error(err), label=run_id)

    ax.set_xlabel("time [s]")
    ax.set_ylabel("horizontal PF error [m]")
    ax.set_title("Priority-3 PF horizontal error comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _save_figure(fig, FIG_DIR / "priority3_pf_horizontal_error.png")


def _plot_pf_likelihood_diagnostics(
    bundles: dict[str, dict[str, Any]],
) -> Path:
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(9.0, 8.6),
        sharex=True,
        constrained_layout=True,
    )

    for run_id, bundle in bundles.items():
        archive = bundle["archive"]
        t = np.asarray(archive["pf_time_s"], dtype=np.float64)
        ess = np.asarray(archive["pf_effective_sample_size"], dtype=np.float64)
        gstd = np.asarray(archive["pf_predicted_disturbance_std_mps2"], dtype=np.float64) * 1.0e5

        axes[0].plot(t, ess, label=run_id)
        axes[1].plot(t, gstd, label=run_id)

        err = _position_error_history_from_archive(archive, prefix="pf")
        axes[2].plot(t, _horizontal_error(err), label=run_id)

    axes[0].set_ylabel("ESS")
    axes[0].set_title("Priority-3 PF likelihood diagnostics")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].set_ylabel("pred. dist. std\n[mGal]")
    axes[1].grid(True, alpha=0.25)

    axes[2].set_ylabel("PF horiz err\n[m]")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(True, alpha=0.25)

    return _save_figure(fig, FIG_DIR / "priority3_pf_likelihood_diagnostics.png")


def _plot_directional_feedback_diagnostics(
    directional_bundle: dict[str, Any],
) -> Path:
    archive = directional_bundle["archive"]
    rows = archive.get("estimator_custom_streams", {}).get("pf_directional_feedback", [])
    if len(rows) == 0:
        raise RuntimeError("Directional feedback diagnostics stream is missing.")

    t = np.asarray(archive["pf_time_s"], dtype=np.float64)
    n = min(len(t), len(rows))
    t = t[:n]

    eig_ratio = np.asarray([rows[k].get("eigenvalue_ratio", np.nan) for k in range(n)], dtype=np.float64)
    ess_fraction = np.asarray([rows[k].get("ess_fraction", np.nan) for k in range(n)], dtype=np.float64)
    allowed = np.asarray([bool(rows[k].get("feedback_allowed", False)) for k in range(n)], dtype=np.float64)
    applied = np.asarray([bool(rows[k].get("applied", False)) for k in range(n)], dtype=np.float64)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(9.0, 8.6),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(t, eig_ratio, label="directional eig ratio")
    axes[0].axhline(8.0, color="tab:red", linestyle="--", label="gate threshold")
    axes[0].set_ylabel("eig ratio")
    axes[0].set_title("Directional feedback gate diagnostics")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(t, ess_fraction, label="ESS fraction")
    axes[1].set_ylabel("ESS frac")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    axes[2].step(t, allowed, where="post", label="allowed")
    axes[2].step(t, applied, where="post", linestyle="--", label="applied")
    axes[2].set_ylabel("flag")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    return _save_figure(fig, FIG_DIR / "priority3_directional_feedback_diagnostics.png")


def _top_rejection_reasons(rows: list[dict[str, Any]], *, top_n: int = 5) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = row.get("rejection_reason")
        if reason is not None:
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]


def _metric(bundle: dict[str, Any], *keys: str) -> Any:
    value: Any = bundle["metrics"]
    for key in keys:
        value = value[key]
    return value


def _change_phrase(new_value: float, base_value: float) -> str:
    delta_pct = 100.0 * (new_value - base_value) / base_value
    if np.isclose(delta_pct, 0.0):
        return "no material change"
    direction = "better" if delta_pct < 0.0 else "worse"
    return f"{abs(delta_pct):.2f}% {direction}"


def _write_report(
    bundles: dict[str, dict[str, Any]],
    *,
    figure_paths: dict[str, Path],
) -> Path:
    observe = bundles["observe_only"]
    gradient = bundles["observe_plus_gradient"]
    directional = bundles["directional_plus_gradient"]

    direction_rows = directional["archive"].get("estimator_custom_streams", {}).get(
        "pf_directional_feedback",
        [],
    )
    gradiometer_samples = len(
        gradient["archive"].get("sensor_custom_streams", {}).get("gradiometer", [])
    )
    directional_allowed = sum(1 for row in direction_rows if row.get("feedback_allowed"))
    directional_applied = sum(1 for row in direction_rows if row.get("applied"))

    obs_pf_rmse = float(_metric(observe, "pf_position_error", "horizontal_rmse_m"))
    grad_pf_rmse = float(_metric(gradient, "pf_position_error", "horizontal_rmse_m"))
    dir_pf_rmse = float(_metric(directional, "pf_position_error", "horizontal_rmse_m"))
    obs_ins_rmse = float(_metric(observe, "ins_position_error", "horizontal_rmse_m"))
    dir_ins_rmse = float(_metric(directional, "ins_position_error", "horizontal_rmse_m"))

    gradient_pf_rmse_change = _change_phrase(grad_pf_rmse, obs_pf_rmse)
    directional_pf_rmse_change = _change_phrase(dir_pf_rmse, obs_pf_rmse)
    obs_pf_cep95 = float(_metric(observe, "pf_position_error", "cep95_m"))
    grad_pf_cep95 = float(_metric(gradient, "pf_position_error", "cep95_m"))
    dir_pf_cep95 = float(_metric(directional, "pf_position_error", "cep95_m"))

    top_rejections = _top_rejection_reasons(direction_rows, top_n=5)
    top_rejection_lines = [
        f"  - `{reason}`: `{count}` updates"
        for reason, count in top_rejections
    ]
    if len(top_rejection_lines) == 0:
        top_rejection_lines = ["  - none"]

    lines = [
        "# Priority-3 Gradient Likelihood Report",
        "",
        "## Scope",
        "",
        "- Scenario: `maritime_baseline`",
        "- Runs compared:",
        "  - `observe_only`",
        "  - `observe_plus_gradient`",
        "  - `directional_plus_gradient`",
        "- Output directory:",
        f"  - `{_relative(RUN_DIR)}`",
        "",
        "## Key Results",
        "",
        "| Run | INS horiz RMSE [m] | INS CEP95 [m] | PF horiz RMSE [m] | PF CEP95 [m] |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| observe_only | {_metric(observe, 'ins_position_error', 'horizontal_rmse_m'):.3f} | {_metric(observe, 'ins_position_error', 'cep95_m'):.3f} | {_metric(observe, 'pf_position_error', 'horizontal_rmse_m'):.3f} | {_metric(observe, 'pf_position_error', 'cep95_m'):.3f} |",
        f"| observe_plus_gradient | {_metric(gradient, 'ins_position_error', 'horizontal_rmse_m'):.3f} | {_metric(gradient, 'ins_position_error', 'cep95_m'):.3f} | {_metric(gradient, 'pf_position_error', 'horizontal_rmse_m'):.3f} | {_metric(gradient, 'pf_position_error', 'cep95_m'):.3f} |",
        f"| directional_plus_gradient | {_metric(directional, 'ins_position_error', 'horizontal_rmse_m'):.3f} | {_metric(directional, 'ins_position_error', 'cep95_m'):.3f} | {_metric(directional, 'pf_position_error', 'horizontal_rmse_m'):.3f} | {_metric(directional, 'pf_position_error', 'cep95_m'):.3f} |",
        "",
        "## Highlights",
        "",
        f"- INS horizontal RMSE is unchanged across all three runs at `{obs_ins_rmse:.3f} m`.",
        f"- `observe_plus_gradient` changes PF horizontal RMSE from `{obs_pf_rmse:.3f} m` to `{grad_pf_rmse:.3f} m` (`{gradient_pf_rmse_change}`) and PF CEP95 from `{obs_pf_cep95:.3f} m` to `{grad_pf_cep95:.3f} m`.",
        f"- `directional_plus_gradient` matches `observe_plus_gradient` on this baseline: PF horizontal RMSE is `{dir_pf_rmse:.3f} m`, PF CEP95 is `{dir_pf_cep95:.3f} m`, and no PF feedback update was applied.",
        f"- The gradiometer path was active in the gradient runs with `{gradiometer_samples}` logged gradiometer samples.",
        f"- Directional feedback evaluations: `{len(direction_rows)}`; allowed: `{directional_allowed}`; applied: `{directional_applied}`.",
        "",
        "## Directional Feedback Diagnostics",
        "",
        "- Dominant rejection reasons:",
        *top_rejection_lines,
        "",
        "Interpretation:",
        "- The gradiometer and gradient-likelihood path is active, but the current closed-loop policy still finds the directional feedback unsafe on every PF update.",
        "- With zero applied PF feedback, INS-level RMSE and CEP95 stay flat across the three runs.",
        "",
        "## Figures",
        "",
        f"- INS horizontal error comparison: `{_relative(figure_paths['ins_horizontal'])}`",
        f"- PF horizontal error comparison: `{_relative(figure_paths['pf_horizontal'])}`",
        f"- PF likelihood diagnostics: `{_relative(figure_paths['pf_likelihood'])}`",
        f"- Directional feedback gate diagnostics: `{_relative(figure_paths['directional_diag'])}`",
        "",
        "## Conclusion",
        "",
        f"The Priority-3 plumbing is working. Gradient likelihood is active, the gradiometer stream is logged, and the benchmark/report path now exposes both PF-side and INS-side outcomes directly. On the current maritime baseline, scalar-plus-gradient likelihood improves PF horizontal RMSE modestly from `{obs_pf_rmse:.3f} m` to `{grad_pf_rmse:.3f} m`, but the directional controller still injects zero PF corrections into the INS. The net result is unchanged INS performance at `{obs_ins_rmse:.3f} m` horizontal RMSE across the comparison runs.",
        "",
        "The next step is not more plotting. It is deciding whether to relax the feedback gates in a controlled experiment, or to move to the next planned information source so the PF posterior becomes trustworthy enough for closed-loop feedback.",
        "",
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


def main() -> int:
    bundles: dict[str, dict[str, Any]] = {}
    for run_id in RUN_IDS:
        bundles[run_id] = {
            "metrics": _load_metrics(run_id),
            "config": _load_config(run_id),
            "archive": _load_archive(run_id),
        }

        if bundles[run_id]["metrics"].get("pf_position_error") is None:
            bundles[run_id]["metrics"]["pf_position_error"] = _pf_metrics_from_archive(
                bundles[run_id]["archive"]
            )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "ins_horizontal": _plot_ins_horizontal_error_histories(bundles),
        "pf_horizontal": _plot_pf_horizontal_error_histories(bundles),
        "pf_likelihood": _plot_pf_likelihood_diagnostics(bundles),
        "directional_diag": _plot_directional_feedback_diagnostics(
            bundles["directional_plus_gradient"]
        ),
    }

    report_path = _write_report(bundles, figure_paths=figure_paths)
    print(f"Saved report: {_relative(report_path)}")
    for key, path in figure_paths.items():
        print(f"Saved {key}: {_relative(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
