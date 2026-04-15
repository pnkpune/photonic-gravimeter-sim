#!/usr/bin/env python3
"""
Generate high-signal summary plots for the main results collected so far.

The goal of this script is not to reproduce every experiment. It extracts the
strongest tracked checkpoints plus the latest local three-region public-result
artifacts and turns them into a compact visual status report:

1. milestone progression across the main phases
2. what worked versus what failed in the synthetic estimator experiments
3. current three-region multimodal ablations
4. ranked impact summary of the main approaches
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gravnav_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gravnav_xdg_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/outputs/reports/project_summary_plots"

VALIDATED_IMU_METRICS = PROJECT_ROOT / "data/outputs/runs/validated_maritime_baseline/maritime_baseline_imu_only_metrics.json"
VALIDATED_AIDED_METRICS = PROJECT_ROOT / "data/outputs/runs/validated_maritime_baseline/maritime_baseline_aided_metrics.json"
PRIORITY3_SUMMARY = PROJECT_ROOT / "data/outputs/reports/priority3_benchmark/maritime_baseline_summary.json"
REGIONAL_SUMMARY = PROJECT_ROOT / "data/outputs/reports/norwegian_margin_benchmark/norwegian_margin_maritime_summary.json"
TWO_REGION_SUMMARY = PROJECT_ROOT / "data/outputs/reports/second_region_validation/two_region_validation_summary.json"

DEFAULT_CURRENT_REGION_SUMMARIES = {
    "Norwegian margin": Path("/private/tmp/gravnav_priority9_emodnet_nm_hybrid3/hardware_tied_maritime_demo_summary.json"),
    "Helgeland offshore": Path("/private/tmp/gravnav_priority9_emodnet_hel_hybrid3/hardware_tied_maritime_demo_summary.json"),
    "Nordland offshore": Path("/private/tmp/gravnav_priority9_emodnet_nord_hybrid3/hardware_tied_maritime_demo_summary.json"),
}

ABLATION_LABEL_ORDER = [
    "live_ins",
    "photonic_gravity_baseline",
    "photonic_gravity_tide",
    "photonic_gravity_tide_acoustic",
    "photonic_gravity_tide_acoustic_magnetic",
    "photonic_gravity_tide_acoustic_magnetic_current",
]

ABLATION_LABEL_DISPLAY = {
    "live_ins": "INS",
    "photonic_gravity_baseline": "Gravity",
    "photonic_gravity_tide": "+ Tide",
    "photonic_gravity_tide_acoustic": "+ Acoustic",
    "photonic_gravity_tide_acoustic_magnetic": "+ Magnetic",
    "photonic_gravity_tide_acoustic_magnetic_current": "+ Current",
}

GOOD_GREEN = "#1f7a4c"
GOOD_TEAL = "#1a7f8e"
WARN_AMBER = "#c48a00"
BAD_RED = "#b9372b"
MID_BLUE = "#235789"
NEUTRAL_GRAY = "#667085"
LIGHT_GRAY = "#d0d5dd"
BACKGROUND = "#f7f5ef"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _median_key(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return _median(values)


def _majority_str(rows: list[dict[str, Any]], key: str) -> str | None:
    vals = [str(row[key]) for row in rows if row.get(key) is not None]
    if not vals:
        return None
    return statistics.mode(vals)


def _fraction_dict_sum(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        payload = row.get(key)
        if not isinstance(payload, dict):
            continue
        for name, value in payload.items():
            out[str(name)] = out.get(str(name), 0.0) + float(value)
    total = float(sum(out.values()))
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in out.items()}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": BACKGROUND,
            "axes.edgecolor": "#344054",
            "axes.labelcolor": "#101828",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.color": "#344054",
            "ytick.color": "#344054",
            "grid.color": "#d5d9e0",
            "grid.alpha": 0.6,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 220,
            "font.size": 10,
        }
    )


def _validated_baseline_payload() -> dict[str, Any]:
    imu = _load_json(VALIDATED_IMU_METRICS)
    aided = _load_json(VALIDATED_AIDED_METRICS)
    return {
        "imu_only_rmse_m": float(imu["ins_position_error"]["horizontal_rmse_m"]),
        "imu_only_cep95_m": float(imu["ins_position_error"]["cep95_m"]),
        "aided_rmse_m": float(aided["ins_position_error"]["horizontal_rmse_m"]),
        "aided_cep95_m": float(aided["ins_position_error"]["cep95_m"]),
        "duration_s": float(aided["duration_s"]),
    }


def _priority3_payload() -> list[dict[str, Any]]:
    payload = _load_json(PRIORITY3_SUMMARY)
    if not isinstance(payload, list):
        raise TypeError("Expected list payload for priority3 summary.")
    return [dict(row) for row in payload]


def _regional_payload() -> dict[str, Any]:
    payload = _load_json(REGIONAL_SUMMARY)
    if not isinstance(payload, dict):
        raise TypeError("Expected dict payload for regional summary.")
    return dict(payload)


def _two_region_payload() -> dict[str, Any]:
    payload = _load_json(TWO_REGION_SUMMARY)
    if not isinstance(payload, dict):
        raise TypeError("Expected dict payload for two-region summary.")
    return dict(payload)


def _current_region_payload(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise TypeError(f"Expected list payload in {path}.")
    return [dict(row) for row in payload]


def _current_region_summaries(
    region_paths: dict[str, Path],
) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for region_name, path in region_paths.items():
        rows = _current_region_payload(path)
        by_label: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_label.setdefault(str(row["label"]), []).append(row)
        region_summary: dict[str, dict[str, Any]] = {}
        for label, label_rows in by_label.items():
            region_summary[label] = {
                "ins_rmse_m": _median_key(label_rows, "ins_horizontal_rmse_m"),
                "ins_cep95_m": _median_key(label_rows, "ins_cep95_m"),
                "sequence_rmse_m": _median_key(label_rows, "sequence_horizontal_rmse_m"),
                "sequence_cep95_m": _median_key(label_rows, "sequence_cep95_m"),
                "lag_rmse_m": _median_key(label_rows, "lag_horizontal_rmse_m"),
                "lag_cep95_m": _median_key(label_rows, "lag_cep95_m"),
                "reported_rmse_m": _median_key(label_rows, "earth_signature_horizontal_rmse_m"),
                "reported_cep95_m": _median_key(label_rows, "earth_signature_cep95_m"),
                "reported_hmi": _median_key(label_rows, "earth_signature_hmi_horizontal"),
                "reported_mode": _majority_str(label_rows, "reported_output_mode"),
                "reported_reason": _majority_str(label_rows, "reported_output_reason"),
                "failure_mode_mix": _fraction_dict_sum(label_rows, "ambiguity_failure_mode_counts"),
                "informative_fraction": _median_key(label_rows, "ambiguity_informative_fraction"),
                "edge_clipped_fraction": _median_key(label_rows, "ambiguity_edge_clipped_fraction"),
                "prior_dominated_fraction": _median_key(label_rows, "ambiguity_prior_dominated_fraction"),
                "flat_signature_fraction": _median_key(label_rows, "ambiguity_flat_signature_fraction"),
                "bathymetry_info_ratio": _median_key(label_rows, "ambiguity_median_bathymetry_information_ratio"),
                "magnetic_info_ratio": _median_key(label_rows, "ambiguity_median_magnetic_information_ratio"),
            }
        out[region_name] = region_summary
    return out


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_milestone_progression(
    *,
    baseline: dict[str, Any],
    priority3: list[dict[str, Any]],
    regional: dict[str, Any],
    two_region: dict[str, Any],
    current_regions: dict[str, dict[str, dict[str, Any]]],
    output_path: Path,
) -> None:
    priority3_by_label = {str(row["label"]): dict(row) for row in priority3}
    regional_by_label = {str(row["label"]): dict(row) for row in regional["runs"]}
    two_regions_by_name = {str(region["region_name"]): dict(region) for region in two_region["regions"]}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle("Project Progression: What Improved Accuracy and What Did Not", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    bars = [
        ("IMU-only", baseline["imu_only_rmse_m"], BAD_RED),
        ("Validated\naided", baseline["aided_rmse_m"], GOOD_GREEN),
    ]
    ax.bar([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars], width=0.6)
    ax.set_yscale("log")
    ax.set_title("Baseline Stabilization")
    ax.set_ylabel("Horizontal RMSE [m]")
    ax.grid(True, axis="y")
    ax.text(
        0.02,
        0.98,
        f"34 min mission\n{baseline['imu_only_rmse_m'] / baseline['aided_rmse_m']:.1f}x improvement",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": LIGHT_GRAY, "boxstyle": "round,pad=0.3"},
    )

    ax = axes[0, 1]
    synthetic_labels = [
        ("Live INS", priority3_by_label["observe_only"]["horizontal_rmse_m"], NEUTRAL_GRAY),
        ("PF + grad", priority3_by_label["observe_plus_gradient"]["pf_horizontal_rmse_m"], MID_BLUE),
        ("Sequence + grad", priority3_by_label["sequence_plus_gradient"]["sequence_horizontal_rmse_m"], GOOD_GREEN),
        ("Lag smoothed", priority3_by_label["sequence_plus_gradient_lag_smoothed"]["lag_smoothed_horizontal_rmse_m"], GOOD_TEAL),
        ("Replay\nrelaxed", priority3_by_label["sequence_plus_gradient_replay_feedback_relaxed"]["horizontal_rmse_m"], BAD_RED),
    ]
    ax.bar([b[0] for b in synthetic_labels], [b[1] for b in synthetic_labels], color=[b[2] for b in synthetic_labels], width=0.6)
    ax.set_title("Synthetic Estimator Benchmark")
    ax.set_ylabel("Horizontal RMSE [m]")
    ax.grid(True, axis="y")
    ax.text(
        4,
        synthetic_labels[-1][1] + 2.5,
        f"HMI {priority3_by_label['sequence_plus_gradient_replay_feedback_relaxed']['hmi_horizontal']:.3f}",
        ha="center",
        va="bottom",
        color=BAD_RED,
        fontweight="bold",
    )

    ax = axes[1, 0]
    regional_labels = [
        ("Live INS", regional_by_label["ins_only"]["horizontal_rmse_m"], NEUTRAL_GRAY),
        ("PF + grad", regional_by_label["observe_plus_gradient"]["pf_horizontal_rmse_m"], MID_BLUE),
        ("Sequence + grad", regional_by_label["sequence_plus_gradient"]["sequence_horizontal_rmse_m"], GOOD_GREEN),
        ("Lag smoothed", regional_by_label["sequence_plus_gradient_lag_smoothed"]["lag_smoothed_horizontal_rmse_m"], GOOD_TEAL),
    ]
    ax.bar([b[0] for b in regional_labels], [b[1] for b in regional_labels], color=[b[2] for b in regional_labels], width=0.6)
    ax.set_title("Regional Fixture Reality Check")
    ax.set_ylabel("Horizontal RMSE [m]")
    ax.grid(True, axis="y")
    ax.text(
        0.02,
        0.98,
        "Gains survive, but shrink sharply\nrelative to synthetic maps",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": LIGHT_GRAY, "boxstyle": "round,pad=0.3"},
    )

    ax = axes[1, 1]
    region_order = ["Norwegian margin", "Helgeland offshore", "Nordland offshore"]
    x = np.arange(len(region_order))
    live_vals = []
    report_vals = []
    labels = []
    for region_name in region_order:
        if region_name == "Norwegian margin":
            live_vals.append(float(two_regions_by_name["norwegian_margin"]["live_ins_horizontal_rmse_m_median"]))
            report_vals.append(float(current_regions[region_name]["photonic_gravity_tide_acoustic_magnetic"]["reported_rmse_m"]))
            labels.append("sequence")
        elif region_name == "Helgeland offshore":
            live_vals.append(float(two_regions_by_name["helgeland_offshore"]["live_ins_horizontal_rmse_m_median"]))
            report_vals.append(float(current_regions[region_name]["photonic_gravity_tide_acoustic_magnetic"]["reported_rmse_m"]))
            labels.append(str(current_regions[region_name]["photonic_gravity_tide_acoustic_magnetic"]["reported_mode"]))
        else:
            live_vals.append(float(current_regions[region_name]["live_ins"]["reported_rmse_m"]))
            report_vals.append(float(current_regions[region_name]["photonic_gravity_tide_acoustic_magnetic"]["reported_rmse_m"]))
            labels.append(str(current_regions[region_name]["photonic_gravity_tide_acoustic_magnetic"]["reported_mode"]))
    width = 0.36
    ax.bar(x - width / 2, live_vals, width=width, color=NEUTRAL_GRAY, label="Live INS")
    ax.bar(x + width / 2, report_vals, width=width, color=GOOD_GREEN, label="Best current reported output")
    for xi, y, mode in zip(x, report_vals, labels, strict=True):
        ax.text(xi + width / 2, y + 3.0, mode, ha="center", va="bottom", fontsize=9, color="#101828")
    ax.set_xticks(x, region_order)
    ax.set_title("Current Public Three-Region Status")
    ax.set_ylabel("Horizontal RMSE [m]")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left")

    _save_figure(fig, output_path)


def _plot_feedback_failure(
    *,
    priority3: list[dict[str, Any]],
    output_path: Path,
) -> None:
    rows = {str(row["label"]): dict(row) for row in priority3}
    chosen = [
        ("Live INS", rows["observe_only"]["horizontal_rmse_m"], rows["observe_only"]["cep95_m"], 0.0, NEUTRAL_GRAY),
        ("Sequence + grad", rows["sequence_plus_gradient"]["sequence_horizontal_rmse_m"], rows["sequence_plus_gradient"]["sequence_cep95_m"], 0.0, GOOD_GREEN),
        ("Lag smoothed", rows["sequence_plus_gradient_lag_smoothed"]["lag_smoothed_horizontal_rmse_m"], rows["sequence_plus_gradient_lag_smoothed"]["lag_smoothed_cep95_m"], rows["sequence_plus_gradient_lag_smoothed"]["lag_hmi_horizontal"] or 0.0, GOOD_TEAL),
        ("Replay\nrelaxed", rows["sequence_plus_gradient_replay_feedback_relaxed"]["horizontal_rmse_m"], rows["sequence_plus_gradient_replay_feedback_relaxed"]["cep95_m"], rows["sequence_plus_gradient_replay_feedback_relaxed"]["hmi_horizontal"], BAD_RED),
        ("Directional\nsafe", rows["sequence_plus_gradient_replay_directional_safe"]["horizontal_rmse_m"], rows["sequence_plus_gradient_replay_directional_safe"]["cep95_m"], rows["sequence_plus_gradient_replay_directional_safe"]["hmi_horizontal"], WARN_AMBER),
    ]

    x = np.arange(len(chosen))
    fig, ax1 = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax2 = ax1.twinx()
    ax1.bar(x, [c[1] for c in chosen], color=[c[4] for c in chosen], width=0.62)
    ax2.plot(x, [c[3] for c in chosen], color=BAD_RED, marker="o", linewidth=2.2, label="Horizontal HMI")
    ax1.set_xticks(x, [c[0] for c in chosen])
    ax1.set_ylabel("Horizontal RMSE [m]")
    ax2.set_ylabel("HMI fraction")
    ax1.set_title("Synthetic Feedback Experiments: Observe-Only Works, Direct Feedback Does Not")
    ax1.grid(True, axis="y")
    ax2.set_ylim(-0.02, max(0.42, max(c[3] for c in chosen) + 0.04))
    ax2.axhline(0.0, color=LIGHT_GRAY, linewidth=1.0)
    ax2.legend(loc="upper right")

    for xi, (_, rmse, cep95, hmi, _) in enumerate(chosen):
        ax1.text(xi, rmse + 2.0, f"CEP95 {cep95:.1f}", ha="center", va="bottom", fontsize=8)
        if hmi > 0.0:
            ax2.text(xi, hmi + 0.015, f"HMI {hmi:.3f}", ha="center", va="bottom", fontsize=9, color=BAD_RED)

    _save_figure(fig, output_path)


def _plot_three_region_ablation(
    *,
    current_regions: dict[str, dict[str, dict[str, Any]]],
    output_path: Path,
) -> None:
    region_order = ["Norwegian margin", "Helgeland offshore", "Nordland offshore"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True, sharey=True)
    fig.suptitle("Current Three-Region Multimodal Ablation", fontsize=15, fontweight="bold")

    x = np.arange(len(ABLATION_LABEL_ORDER))
    xticklabels = [ABLATION_LABEL_DISPLAY[label] for label in ABLATION_LABEL_ORDER]
    for ax, region_name in zip(axes, region_order, strict=True):
        region = current_regions[region_name]
        ins_line = float(region["live_ins"]["reported_rmse_m"])
        sequence = [region[label]["sequence_rmse_m"] for label in ABLATION_LABEL_ORDER]
        lag = [region[label]["lag_rmse_m"] for label in ABLATION_LABEL_ORDER]
        reported = [region[label]["reported_rmse_m"] for label in ABLATION_LABEL_ORDER]
        modes = [region[label]["reported_mode"] or "-" for label in ABLATION_LABEL_ORDER]
        hmi = [region[label]["reported_hmi"] for label in ABLATION_LABEL_ORDER]

        ax.axhline(ins_line, color=NEUTRAL_GRAY, linestyle="--", linewidth=1.8, label="Live INS" if region_name == region_order[0] else None)
        ax.plot(x, sequence, color=GOOD_GREEN, marker="o", linewidth=2.2, label="Raw sequence" if region_name == region_order[0] else None)
        ax.plot(x, lag, color=GOOD_TEAL, marker="s", linewidth=2.2, label="Raw lag" if region_name == region_order[0] else None)
        ax.plot(x, reported, color=MID_BLUE, marker="D", linewidth=2.5, label="Published output" if region_name == region_order[0] else None)
        ax.set_title(region_name)
        ax.set_xticks(x, xticklabels, rotation=25, ha="right")
        ax.set_ylabel("Horizontal RMSE [m]")
        ax.grid(True, axis="y")
        for xi, y, mode, hmi_val in zip(x, reported, modes, hmi, strict=True):
            mode_text = "INS" if mode == "live_ins" else ("SEQ" if mode == "sequence" else ("LAG" if mode == "lag_smoothed" else mode))
            text = mode_text
            if hmi_val is not None and hmi_val > 0.0:
                text += f"\nHMI {hmi_val:.3f}"
            ax.text(xi, y + 4.0, text, ha="center", va="bottom", fontsize=8, color="#101828")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    _save_figure(fig, output_path)


def _plot_impact_ranking(
    *,
    baseline: dict[str, Any],
    priority3: list[dict[str, Any]],
    regional: dict[str, Any],
    current_regions: dict[str, dict[str, dict[str, Any]]],
    output_path: Path,
) -> list[dict[str, Any]]:
    priority3_by_label = {str(row["label"]): dict(row) for row in priority3}
    regional_by_label = {str(row["label"]): dict(row) for row in regional["runs"]}

    def improvement_pct(baseline_rmse: float, candidate_rmse: float) -> float:
        return 100.0 * (baseline_rmse - candidate_rmse) / baseline_rmse

    items = [
        {
            "name": "Validated local aiding vs IMU-only",
            "group": "Baseline",
            "improvement_pct": improvement_pct(baseline["imu_only_rmse_m"], baseline["aided_rmse_m"]),
        },
        {
            "name": "PF + gradient on synthetic benchmark",
            "group": "Synthetic",
            "improvement_pct": improvement_pct(
                priority3_by_label["observe_plus_gradient"]["horizontal_rmse_m"],
                priority3_by_label["observe_plus_gradient"]["pf_horizontal_rmse_m"],
            ),
        },
        {
            "name": "Sequence + gradient on synthetic benchmark",
            "group": "Synthetic",
            "improvement_pct": improvement_pct(
                priority3_by_label["sequence_plus_gradient"]["horizontal_rmse_m"],
                priority3_by_label["sequence_plus_gradient"]["sequence_horizontal_rmse_m"],
            ),
        },
        {
            "name": "Lag-smoothed output on synthetic benchmark",
            "group": "Synthetic",
            "improvement_pct": improvement_pct(
                priority3_by_label["sequence_plus_gradient_lag_smoothed"]["horizontal_rmse_m"],
                priority3_by_label["sequence_plus_gradient_lag_smoothed"]["lag_smoothed_horizontal_rmse_m"],
            ),
        },
        {
            "name": "Sequence + gradient on regional fixture",
            "group": "Regional",
            "improvement_pct": improvement_pct(
                regional_by_label["sequence_plus_gradient"]["horizontal_rmse_m"],
                regional_by_label["sequence_plus_gradient"]["sequence_horizontal_rmse_m"],
            ),
        },
        {
            "name": "Lag-smoothed output on regional fixture",
            "group": "Regional",
            "improvement_pct": improvement_pct(
                regional_by_label["sequence_plus_gradient_lag_smoothed"]["horizontal_rmse_m"],
                regional_by_label["sequence_plus_gradient_lag_smoothed"]["lag_smoothed_horizontal_rmse_m"],
            ),
        },
        {
            "name": "Bathymetry lag at frozen Norwegian demo",
            "group": "Milestone",
            "improvement_pct": 100.0 * (241.52639714417754 - 197.37123449518643) / 241.52639714417754,
        },
        {
            "name": "Current Norwegian branch: acoustic + magnetic reported",
            "group": "Current",
            "improvement_pct": improvement_pct(
                current_regions["Norwegian margin"]["live_ins"]["reported_rmse_m"],
                current_regions["Norwegian margin"]["photonic_gravity_tide_acoustic_magnetic"]["reported_rmse_m"],
            ),
        },
        {
            "name": "Current-aware prior on Norwegian margin",
            "group": "Current",
            "improvement_pct": improvement_pct(
                current_regions["Norwegian margin"]["live_ins"]["reported_rmse_m"],
                current_regions["Norwegian margin"]["photonic_gravity_tide_acoustic_magnetic_current"]["reported_rmse_m"],
            ),
        },
        {
            "name": "Replay feedback relaxed",
            "group": "Dead end",
            "improvement_pct": improvement_pct(
                priority3_by_label["sequence_plus_gradient_replay_feedback_relaxed"]["horizontal_rmse_m"],
                priority3_by_label["sequence_plus_gradient_replay_feedback_relaxed"]["horizontal_rmse_m"],
            ),
        },
    ]

    items[-1]["improvement_pct"] = 100.0 * (
        priority3_by_label["sequence_plus_gradient"]["sequence_horizontal_rmse_m"]
        - priority3_by_label["sequence_plus_gradient_replay_feedback_relaxed"]["horizontal_rmse_m"]
    ) / priority3_by_label["sequence_plus_gradient"]["sequence_horizontal_rmse_m"]

    items = sorted(items, key=lambda item: item["improvement_pct"])

    fig, ax = plt.subplots(figsize=(11.5, 7.5), constrained_layout=True)
    y = np.arange(len(items))
    colors = [
        GOOD_GREEN if item["improvement_pct"] > 5.0 else GOOD_TEAL if item["improvement_pct"] > 0.0 else BAD_RED
        for item in items
    ]
    ax.barh(y, [item["improvement_pct"] for item in items], color=colors)
    ax.axvline(0.0, color="#101828", linewidth=1.0)
    ax.set_yticks(y, [item["name"] for item in items])
    ax.set_xlabel("Improvement vs relevant baseline [%]")
    ax.set_title("Ranked Impact of the Main Approaches")
    ax.grid(True, axis="x")
    for yi, item in enumerate(items):
        ax.text(
            item["improvement_pct"] + (0.35 if item["improvement_pct"] >= 0.0 else -0.35),
            yi,
            f"{item['improvement_pct']:.2f}%",
            va="center",
            ha="left" if item["improvement_pct"] >= 0.0 else "right",
            fontsize=9,
        )
    _save_figure(fig, output_path)
    return items


def _write_summary_markdown(
    path: Path,
    *,
    baseline: dict[str, Any],
    impact_items: list[dict[str, Any]],
    current_regions: dict[str, dict[str, dict[str, Any]]],
    region_paths: dict[str, Path],
) -> None:
    nm = current_regions["Norwegian margin"]["photonic_gravity_tide_acoustic_magnetic"]
    hel = current_regions["Helgeland offshore"]["photonic_gravity_tide_acoustic_magnetic"]
    nord = current_regions["Nordland offshore"]["photonic_gravity_tide_acoustic_magnetic"]

    top_positive = sorted(
        [item for item in impact_items if item["improvement_pct"] > 0.0],
        key=lambda item: item["improvement_pct"],
        reverse=True,
    )[:5]
    bottom = sorted(impact_items, key=lambda item: item["improvement_pct"])[:3]

    lines = [
        "# Project Summary Plots",
        "",
        "## Scope",
        "",
        "This report condenses the strongest tracked checkpoints plus the latest local three-region public-summary artifacts into a small set of high-signal figures.",
        "",
        "Data sources used:",
        "",
        f"- validated baseline metrics: `{VALIDATED_IMU_METRICS}` and `{VALIDATED_AIDED_METRICS}`",
        f"- synthetic benchmark summary: `{PRIORITY3_SUMMARY}`",
        f"- regional benchmark summary: `{REGIONAL_SUMMARY}`",
        f"- second-region validation summary: `{TWO_REGION_SUMMARY}`",
    ]
    for region_name, src in region_paths.items():
        lines.append(f"- latest local public-summary artifact for `{region_name}`: `{src}`")

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `milestone_progression.png`: baseline stabilization, synthetic estimator ranking, regional reality check, and current three-region public status.",
            "- `feedback_failure_modes.png`: shows why observe-only sequence won and why direct replay feedback was not promotable.",
            "- `three_region_multimodal_ablation.png`: compares raw sequence, raw lag, and published output as tide, acoustic terrain, magnetic, and current layers are added.",
            "- `approach_impact_ranking.png`: ranks the main approaches by measured improvement against their relevant baseline.",
            "",
            "## What The Plots Show",
            "",
            f"- The simulator stabilization step was foundational: IMU-only RMSE `{baseline['imu_only_rmse_m']:.3f} m` versus validated aided baseline `{baseline['aided_rmse_m']:.3f} m`.",
            "- The biggest estimator-level gain came from sequence-based gravity matching, not PF feedback.",
            "- The strongest additive cue after gravity has been bathymetry / acoustic terrain.",
            "- Scalar magnetic anomaly is a small positive nudge, not a transformative change yet.",
            "- Current-aware prior is still a regression in the current branch state.",
            "",
            "Current public-region interpretation:",
            "",
            f"- Norwegian margin is still promotable: reported RMSE `{nm['reported_rmse_m']:.3f} m` with mode `{nm['reported_mode']}`.",
            f"- Helgeland raw sequence is useful (`{hel['sequence_rmse_m']:.3f} m`) but the published output reverts to `{hel['reported_mode']}` because the region remains `{hel['reported_reason']}`.",
            f"- Nordland raw sequence is also useful (`{nord['sequence_rmse_m']:.3f} m`) but the published output still reverts to `{nord['reported_mode']}` because the region remains `{nord['reported_reason']}`.",
            "",
            "## Best Positive Effects",
            "",
        ]
    )
    for item in top_positive:
        lines.append(f"- `{item['name']}`: `{item['improvement_pct']:.2f}%`")

    lines.extend(["", "## Weakest Or Negative Effects", ""])
    for item in bottom:
        lines.append(f"- `{item['name']}`: `{item['improvement_pct']:.2f}%`")

    lines.extend(
        [
            "",
            "## Practical Reading Of The Current State",
            "",
            "- What is working: sequence matching, bathymetry / acoustic terrain, bounded-lag output in strong regions, and a small magnetic assist.",
            "- What is not working: direct feedback heuristics, current-aware prior, and weak-region publication beyond INS fallback.",
            "- What this means: the next likely gain is not another modality first. It is a stronger delayed-output estimator that can convert weak-region raw wins into a safe published output.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate high-level project summary plots.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the report and figures will be written.",
    )
    parser.add_argument(
        "--nm-summary",
        type=Path,
        default=DEFAULT_CURRENT_REGION_SUMMARIES["Norwegian margin"],
        help="Latest local summary JSON for Norwegian margin.",
    )
    parser.add_argument(
        "--hel-summary",
        type=Path,
        default=DEFAULT_CURRENT_REGION_SUMMARIES["Helgeland offshore"],
        help="Latest local summary JSON for Helgeland offshore.",
    )
    parser.add_argument(
        "--nord-summary",
        type=Path,
        default=DEFAULT_CURRENT_REGION_SUMMARIES["Nordland offshore"],
        help="Latest local summary JSON for Nordland offshore.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    _apply_style()

    region_paths = {
        "Norwegian margin": args.nm_summary.expanduser().resolve(),
        "Helgeland offshore": args.hel_summary.expanduser().resolve(),
        "Nordland offshore": args.nord_summary.expanduser().resolve(),
    }
    missing = [str(path) for path in region_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing latest local public-summary artifacts:\n- " + "\n- ".join(missing)
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _validated_baseline_payload()
    priority3 = _priority3_payload()
    regional = _regional_payload()
    two_region = _two_region_payload()
    current_regions = _current_region_summaries(region_paths)

    _plot_milestone_progression(
        baseline=baseline,
        priority3=priority3,
        regional=regional,
        two_region=two_region,
        current_regions=current_regions,
        output_path=output_dir / "milestone_progression.png",
    )
    _plot_feedback_failure(
        priority3=priority3,
        output_path=output_dir / "feedback_failure_modes.png",
    )
    _plot_three_region_ablation(
        current_regions=current_regions,
        output_path=output_dir / "three_region_multimodal_ablation.png",
    )
    impact_items = _plot_impact_ranking(
        baseline=baseline,
        priority3=priority3,
        regional=regional,
        current_regions=current_regions,
        output_path=output_dir / "approach_impact_ranking.png",
    )

    summary_payload = {
        "baseline": baseline,
        "impact_ranking": impact_items,
        "current_regions": current_regions,
        "region_paths": {name: str(path) for name, path in region_paths.items()},
    }
    (output_dir / "project_summary_metrics.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_summary_markdown(
        output_dir / "project_summary_plots.md",
        baseline=baseline,
        impact_items=impact_items,
        current_regions=current_regions,
        region_paths=region_paths,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
