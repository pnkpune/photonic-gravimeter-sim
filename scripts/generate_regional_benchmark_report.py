#!/usr/bin/env python3
"""
Generate the first Norwegian-margin regional benchmark report.

This compares the regional benchmark against the existing synthetic maritime
benchmark and answers three questions:
- does sequence matching still beat PF?
- does the bounded-lag smoother still beat the live INS?
- do the gains shrink materially versus the synthetic map benchmark?
"""

from __future__ import annotations

import argparse
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

from gravnav.datasets.gravity_loader import load_regional_manifest


DEFAULT_REGIONAL_SUMMARY = (
    PROJECT_ROOT / "data/outputs/reports/norwegian_margin_benchmark/norwegian_margin_maritime_summary.json"
)
DEFAULT_SYNTHETIC_SUMMARY = (
    PROJECT_ROOT / "data/outputs/reports/priority3_benchmark/maritime_baseline_summary.json"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data/gravity_maps/processed/norwegian_margin_gravity_map_manifest.json"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "data/outputs/reports/norwegian_margin_regional_benchmark_report.md"
)
DEFAULT_FIG_DIR = PROJECT_ROOT / "data/outputs/figures/norwegian_margin_regional_benchmark"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _load_summary_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        meta = {
            "scenario": path.stem.replace("_summary", ""),
            "profile": "legacy_list",
            "source_path": _relative(path),
        }
        return meta, payload
    if not isinstance(payload, dict) or "runs" not in payload:
        raise ValueError(f"Unsupported benchmark summary format in {path}.")
    meta = {
        "scenario": payload.get("scenario"),
        "profile": payload.get("profile"),
        "regional_map": payload.get("regional_map"),
        "source_path": _relative(path),
    }
    return meta, list(payload["runs"])


def _rows_by_label(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["label"]): row for row in rows}


def _best_row(rows: list[dict[str, Any]], metric_key: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get(metric_key) is not None]
    if len(candidates) == 0:
        return None
    return min(candidates, key=lambda row: float(row[metric_key]))


def _pct_improvement(baseline: float, candidate: float) -> float:
    return 100.0 * (baseline - candidate) / baseline


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_rmse_comparison(
    *,
    synthetic_live_rmse: float,
    synthetic_best_pf_rmse: float | None,
    synthetic_best_seq_rmse: float | None,
    synthetic_best_lag_rmse: float | None,
    regional_live_rmse: float,
    regional_best_pf_rmse: float | None,
    regional_best_seq_rmse: float | None,
    regional_best_lag_rmse: float | None,
    figure_dir: Path,
) -> Path:
    labels = ["live_ins", "best_pf", "best_sequence", "best_lag_smoothed"]
    synth_values = [
        synthetic_live_rmse,
        np.nan if synthetic_best_pf_rmse is None else synthetic_best_pf_rmse,
        np.nan if synthetic_best_seq_rmse is None else synthetic_best_seq_rmse,
        np.nan if synthetic_best_lag_rmse is None else synthetic_best_lag_rmse,
    ]
    regional_values = [
        regional_live_rmse,
        np.nan if regional_best_pf_rmse is None else regional_best_pf_rmse,
        np.nan if regional_best_seq_rmse is None else regional_best_seq_rmse,
        np.nan if regional_best_lag_rmse is None else regional_best_lag_rmse,
    ]

    x = np.arange(len(labels), dtype=np.float64)
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    ax.bar(x - 0.5 * width, synth_values, width=width, label="synthetic")
    ax.bar(x + 0.5 * width, regional_values, width=width, label="norwegian fixture")
    ax.set_xticks(x, labels)
    ax.set_ylabel("horizontal RMSE [m]")
    ax.set_title("Synthetic vs Norwegian regional benchmark")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    return _save_figure(fig, figure_dir / "norwegian_margin_rmse_comparison.png")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the Norwegian regional benchmark report.")
    parser.add_argument("--regional-summary", default=str(DEFAULT_REGIONAL_SUMMARY))
    parser.add_argument("--synthetic-summary", default=str(DEFAULT_SYNTHETIC_SUMMARY))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIG_DIR))
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    regional_summary_path = Path(args.regional_summary).expanduser().resolve()
    synthetic_summary_path = Path(args.synthetic_summary).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    report_path = Path(args.report_path).expanduser().resolve()
    figure_dir = Path(args.figure_dir).expanduser().resolve()

    regional_meta, regional_rows = _load_summary_rows(regional_summary_path)
    synthetic_meta, synthetic_rows = _load_summary_rows(synthetic_summary_path)
    regional_by_label = _rows_by_label(regional_rows)
    synthetic_by_label = _rows_by_label(synthetic_rows)
    manifest = load_regional_manifest(manifest_path)

    regional_live = regional_by_label.get("ins_only", regional_rows[0])
    synthetic_live = synthetic_by_label.get("ins_only", synthetic_rows[0])
    regional_best_pf = _best_row(regional_rows, "pf_horizontal_rmse_m")
    synthetic_best_pf = _best_row(synthetic_rows, "pf_horizontal_rmse_m")
    regional_best_seq = _best_row(regional_rows, "sequence_horizontal_rmse_m")
    synthetic_best_seq = _best_row(synthetic_rows, "sequence_horizontal_rmse_m")
    regional_best_lag = _best_row(regional_rows, "lag_smoothed_horizontal_rmse_m")
    synthetic_best_lag = _best_row(synthetic_rows, "lag_smoothed_horizontal_rmse_m")

    regional_live_rmse = float(regional_live["horizontal_rmse_m"])
    synthetic_live_rmse = float(synthetic_live["horizontal_rmse_m"])
    regional_best_pf_rmse = None if regional_best_pf is None else float(regional_best_pf["pf_horizontal_rmse_m"])
    synthetic_best_pf_rmse = None if synthetic_best_pf is None else float(synthetic_best_pf["pf_horizontal_rmse_m"])
    regional_best_seq_rmse = None if regional_best_seq is None else float(regional_best_seq["sequence_horizontal_rmse_m"])
    synthetic_best_seq_rmse = None if synthetic_best_seq is None else float(synthetic_best_seq["sequence_horizontal_rmse_m"])
    regional_best_lag_rmse = None if regional_best_lag is None else float(regional_best_lag["lag_smoothed_horizontal_rmse_m"])
    synthetic_best_lag_rmse = None if synthetic_best_lag is None else float(synthetic_best_lag["lag_smoothed_horizontal_rmse_m"])

    regional_sequence_beats_pf = (
        regional_best_pf_rmse is not None
        and regional_best_seq_rmse is not None
        and regional_best_seq_rmse < regional_best_pf_rmse
    )
    regional_lag_beats_live = (
        regional_best_lag_rmse is not None
        and regional_best_lag_rmse < regional_live_rmse
        and float(regional_best_lag.get("lag_hmi_horizontal") or 0.0) == 0.0
    )

    synthetic_seq_gain_pct = None
    regional_seq_gain_pct = None
    synthetic_lag_gain_pct = None
    regional_lag_gain_pct = None
    if synthetic_best_seq_rmse is not None:
        synthetic_seq_gain_pct = _pct_improvement(synthetic_live_rmse, synthetic_best_seq_rmse)
    if regional_best_seq_rmse is not None:
        regional_seq_gain_pct = _pct_improvement(regional_live_rmse, regional_best_seq_rmse)
    if synthetic_best_lag_rmse is not None:
        synthetic_lag_gain_pct = _pct_improvement(synthetic_live_rmse, synthetic_best_lag_rmse)
    if regional_best_lag_rmse is not None:
        regional_lag_gain_pct = _pct_improvement(regional_live_rmse, regional_best_lag_rmse)

    gains_shrank_materially = (
        synthetic_seq_gain_pct is not None
        and regional_seq_gain_pct is not None
        and regional_seq_gain_pct < 0.5 * synthetic_seq_gain_pct
    )

    synthetic_seq_gain_text = "n/a" if synthetic_seq_gain_pct is None else f"{synthetic_seq_gain_pct:.2f}%"
    regional_seq_gain_text = "n/a" if regional_seq_gain_pct is None else f"{regional_seq_gain_pct:.2f}%"
    synthetic_lag_gain_text = "n/a" if synthetic_lag_gain_pct is None else f"{synthetic_lag_gain_pct:.2f}%"
    regional_lag_gain_text = "n/a" if regional_lag_gain_pct is None else f"{regional_lag_gain_pct:.2f}%"

    figure_path = _plot_rmse_comparison(
        synthetic_live_rmse=synthetic_live_rmse,
        synthetic_best_pf_rmse=synthetic_best_pf_rmse,
        synthetic_best_seq_rmse=synthetic_best_seq_rmse,
        synthetic_best_lag_rmse=synthetic_best_lag_rmse,
        regional_live_rmse=regional_live_rmse,
        regional_best_pf_rmse=regional_best_pf_rmse,
        regional_best_seq_rmse=regional_best_seq_rmse,
        regional_best_lag_rmse=regional_best_lag_rmse,
        figure_dir=figure_dir,
    )

    lines = [
        "# Norwegian Margin Regional Benchmark Report",
        "",
        "## Scope",
        "",
        "This report compares the existing synthetic maritime benchmark against the new Norwegian-margin regional gravity benchmark path.",
        "",
        f"- regional benchmark summary: `{_relative(regional_summary_path)}`",
        f"- synthetic benchmark summary: `{_relative(synthetic_summary_path)}`",
        f"- regional manifest: `{_relative(manifest_path)}`",
        f"- comparison figure: `{_relative(figure_path)}`",
        "",
        "## Regional Map Source",
        "",
        f"- region: `{manifest.region_name}`",
        f"- source name: `{manifest.source_name}`",
        f"- source kind: `{manifest.source_kind}`",
        f"- raw data path: `{manifest.raw_data_path}`",
        f"- processed map path: `{manifest.processed_map_path}`",
        f"- latitude bounds [deg]: `{manifest.lat_bounds_deg[0]:.3f}` to `{manifest.lat_bounds_deg[1]:.3f}`",
        f"- longitude bounds [deg]: `{manifest.lon_bounds_deg[0]:.3f}` to `{manifest.lon_bounds_deg[1]:.3f}`",
        f"- grid shape: `{manifest.shape[0]} x {manifest.shape[1]}`",
        f"- spacing [deg]: `{manifest.spacing_deg[0]:.3f}` lat, `{manifest.spacing_deg[1]:.3f}` lon",
        "",
        "Important note: the current in-repo Norwegian-margin input is a small tracked fixture that exercises the real regional ingest/cache path. It is not a full public survey grid.",
        "",
        "## Regional Benchmark",
        "",
        "| Configuration | INS RMSE [m] | INS CEP95 [m] | PF RMSE [m] | Sequence RMSE [m] | Lag-smoothed RMSE [m] | HMI horiz [%] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in regional_rows:
        pf_rmse = row["pf_horizontal_rmse_m"]
        seq_rmse = row["sequence_horizontal_rmse_m"]
        lag_rmse = row["lag_smoothed_horizontal_rmse_m"]
        lines.append(
            f"| `{row['label']}` | "
            f"{float(row['horizontal_rmse_m']):.3f} | "
            f"{float(row['cep95_m']):.3f} | "
            f"{'n/a' if pf_rmse is None else f'{float(pf_rmse):.3f}'} | "
            f"{'n/a' if seq_rmse is None else f'{float(seq_rmse):.3f}'} | "
            f"{'n/a' if lag_rmse is None else f'{float(lag_rmse):.3f}'} | "
            f"{100.0 * float(row['hmi_horizontal'] or 0.0):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Synthetic vs Regional Comparison",
            "",
            "| Estimator path | Synthetic RMSE [m] | Regional RMSE [m] |",
            "| --- | ---: | ---: |",
            f"| live INS | {synthetic_live_rmse:.3f} | {regional_live_rmse:.3f} |",
            f"| best PF observe-only | {'n/a' if synthetic_best_pf_rmse is None else f'{synthetic_best_pf_rmse:.3f}'} | {'n/a' if regional_best_pf_rmse is None else f'{regional_best_pf_rmse:.3f}'} |",
            f"| best sequence observe-only | {'n/a' if synthetic_best_seq_rmse is None else f'{synthetic_best_seq_rmse:.3f}'} | {'n/a' if regional_best_seq_rmse is None else f'{regional_best_seq_rmse:.3f}'} |",
            f"| best lag-smoothed output | {'n/a' if synthetic_best_lag_rmse is None else f'{synthetic_best_lag_rmse:.3f}'} | {'n/a' if regional_best_lag_rmse is None else f'{regional_best_lag_rmse:.3f}'} |",
            "",
            "## Findings",
            "",
            f"- Sequence still beats PF on the Norwegian regional benchmark: `{regional_sequence_beats_pf}`.",
            f"- The bounded-lag smoother still beats the live INS on the Norwegian regional benchmark with zero lag-output HMI: `{regional_lag_beats_live}`.",
            f"- Synthetic best sequence improvement vs live INS: `{synthetic_seq_gain_text}`.",
            f"- Regional best sequence improvement vs live INS: `{regional_seq_gain_text}`.",
            f"- Synthetic best lag-smoothed improvement vs live INS: `{synthetic_lag_gain_text}`.",
            f"- Regional best lag-smoothed improvement vs live INS: `{regional_lag_gain_text}`.",
            f"- Gains shrink materially relative to the synthetic benchmark: `{gains_shrank_materially}`.",
            "",
            "## Conclusion",
            "",
        ]
    )

    if regional_sequence_beats_pf and regional_lag_beats_live:
        lines.append(
            "The current estimator ranking survives on the Norwegian-margin fixture path: sequence-based matching remains better than PF, and the bounded-lag smoother still beats the live INS. The improvement is much smaller than on the synthetic maritime benchmark, so the main lesson is not that the algorithm failed, but that the synthetic map materially overstated how much gravity distinctiveness was available."
        )
    else:
        lines.append(
            "The Norwegian regional benchmark weakens the synthetic story enough that the next phase should prioritize stronger regional map realism before any new feedback architecture is promoted."
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report: {_relative(report_path)}")
    print(f"Saved figure: {_relative(figure_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
