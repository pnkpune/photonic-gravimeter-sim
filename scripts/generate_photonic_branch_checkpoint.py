#!/usr/bin/env python3
"""
Summarize calibrated digital-twin telemetry against the two frozen demo packs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


LIVE_LABEL = "live_ins"
PROMOTED_LABEL = "photonic_gravity_bathymetry_lag"
SENSOR_HEALTH_THRESHOLDS = {
    "valid_sample_fraction": 0.85,
    "median_fringe_contrast": 0.45,
    "tilt_exceedance_fraction": 0.15,
}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"Expected list payload in {path}.")
    return [dict(row) for row in payload]


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(statistics.median(values))


def _group_by_label(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row["label"]), []).append(row)
    return out


def _nav_summary(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = _group_by_label(rows)
    if LIVE_LABEL not in by_label or PROMOTED_LABEL not in by_label:
        raise KeyError(
            f"Expected labels {LIVE_LABEL!r} and {PROMOTED_LABEL!r} in region {name!r}."
        )

    live_rows = by_label[LIVE_LABEL]
    promoted_rows = by_label[PROMOTED_LABEL]
    zero_hmi_all = all(float(row["earth_signature_hmi_horizontal"]) == 0.0 for row in promoted_rows)

    return {
        "region_name": name,
        "live_ins_horizontal_rmse_m_median": _median(live_rows, "ins_horizontal_rmse_m"),
        "live_ins_cep95_m_median": _median(live_rows, "ins_cep95_m"),
        "promoted_horizontal_rmse_m_median": _median(
            promoted_rows,
            "earth_signature_horizontal_rmse_m",
        ),
        "promoted_cep95_m_median": _median(promoted_rows, "earth_signature_cep95_m"),
        "promoted_hmi_horizontal_median": _median(
            promoted_rows,
            "earth_signature_hmi_horizontal",
        ),
        "promoted_hmi_zero_all_seeds": zero_hmi_all,
        "accepted": bool(
            _median(promoted_rows, "earth_signature_horizontal_rmse_m")
            < _median(live_rows, "ins_horizontal_rmse_m")
            and _median(promoted_rows, "earth_signature_cep95_m")
            < _median(live_rows, "ins_cep95_m")
            and zero_hmi_all
        ),
    }


def _dominant_rejection_reason(rows: list[dict[str, Any]]) -> str | None:
    counts: dict[str, int] = {}
    for row in rows:
        for reason, count in row.get("rejection_reason_counts", {}).items():
            counts[str(reason)] = counts.get(str(reason), 0) + int(count)
    if not counts:
        return None
    return max(sorted(counts), key=lambda key: counts[key])


def _sensor_summary(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = _group_by_label(rows)
    sensor_rows = by_label.get(PROMOTED_LABEL)
    if not sensor_rows:
        raise KeyError(f"Expected photonic rows for label {PROMOTED_LABEL!r} in region {name!r}.")

    summary = {
        "region_name": name,
        "valid_sample_fraction_median": _median(sensor_rows, "valid_sample_fraction"),
        "median_fringe_contrast_median": _median(sensor_rows, "median_fringe_contrast"),
        "p95_fringe_contrast_median": _median(sensor_rows, "p95_fringe_contrast"),
        "rms_vibration_residual_phase_rad_median": _median(
            sensor_rows,
            "rms_vibration_residual_phase_rad",
        ),
        "rms_disturbance_residual_mps2_median": _median(
            sensor_rows,
            "rms_disturbance_residual_mps2",
        ),
        "tilt_exceedance_fraction_median": _median(
            sensor_rows,
            "tilt_exceedance_fraction",
        ),
        "dominant_rejection_reason": _dominant_rejection_reason(sensor_rows),
    }
    summary["sensor_health_acceptable"] = bool(
        summary["valid_sample_fraction_median"] is not None
        and summary["median_fringe_contrast_median"] is not None
        and summary["tilt_exceedance_fraction_median"] is not None
        and summary["valid_sample_fraction_median"]
        >= SENSOR_HEALTH_THRESHOLDS["valid_sample_fraction"]
        and summary["median_fringe_contrast_median"]
        >= SENSOR_HEALTH_THRESHOLDS["median_fringe_contrast"]
        and summary["tilt_exceedance_fraction_median"]
        <= SENSOR_HEALTH_THRESHOLDS["tilt_exceedance_fraction"]
    )
    return summary


def _diagnose_region(
    nav_summary: dict[str, Any],
    sensor_summary: dict[str, Any],
) -> dict[str, Any]:
    if bool(nav_summary["accepted"]):
        diagnosis = "accepted"
    elif bool(sensor_summary["sensor_health_acceptable"]):
        diagnosis = "region_limited"
    else:
        diagnosis = "sensor_limited"
    return {
        "region_name": nav_summary["region_name"],
        "accepted": bool(nav_summary["accepted"]),
        "sensor_health_acceptable": bool(sensor_summary["sensor_health_acceptable"]),
        "sensor_matches_accepted_reference": False,
        "diagnosis": diagnosis,
        "navigation": nav_summary,
        "sensor": sensor_summary,
    }


def _sensor_matches_reference(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> bool:
    def _close(a_key: str, b_key: str | None = None, *, atol: float) -> bool:
        ref_key = a_key if b_key is None else b_key
        a_val = candidate.get(a_key)
        b_val = reference.get(ref_key)
        if a_val is None or b_val is None:
            return False
        return abs(float(a_val) - float(b_val)) <= atol

    return bool(
        _close("valid_sample_fraction_median", atol=0.03)
        and _close("median_fringe_contrast_median", atol=0.03)
        and _close("tilt_exceedance_fraction_median", atol=0.03)
        and _close("rms_disturbance_residual_mps2_median", atol=5.0e-4)
    )


def _next_branch(second_region: dict[str, Any]) -> str:
    if bool(second_region["accepted"]):
        return "undecided_after_pass"
    if second_region["diagnosis"] == "region_limited":
        return "feature/regional-telemetry-hardening"
    return "feature/photonic-calibration-hardening"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_markdown(
    path: Path,
    *,
    first_region: dict[str, Any],
    second_region: dict[str, Any],
    calibration_summary: dict[str, Any],
    next_branch: str,
) -> None:
    lines = [
        "# Photonic Digital Twin Branch Checkpoint",
        "",
        "## Calibration",
        "",
        f"- lab_static accepted: `{all(calibration_summary['lab_static']['acceptance'].values())}`",
        f"- maritime_benign accepted: `{all(calibration_summary['maritime_benign']['acceptance'].values())}`",
        f"- maritime_rough accepted: `{all(calibration_summary['maritime_rough']['acceptance'].values())}`",
        "",
        "## Region Comparison",
        "",
        "| Region | Accepted | Sensor health acceptable | Matches accepted reference | Diagnosis | Live INS RMSE med [m] | Promoted RMSE med [m] | Live INS CEP95 med [m] | Promoted CEP95 med [m] | HMI zero all seeds | Valid fraction med | Contrast med | Tilt exceedance med |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for region in (first_region, second_region):
        nav = region["navigation"]
        sensor = region["sensor"]
        lines.append(
            f"| `{region['region_name']}` | "
            f"`{region['accepted']}` | "
            f"`{region['sensor_health_acceptable']}` | "
            f"`{region['sensor_matches_accepted_reference']}` | "
            f"`{region['diagnosis']}` | "
            f"{nav['live_ins_horizontal_rmse_m_median']:.3f} | "
            f"{nav['promoted_horizontal_rmse_m_median']:.3f} | "
            f"{nav['live_ins_cep95_m_median']:.3f} | "
            f"{nav['promoted_cep95_m_median']:.3f} | "
            f"`{nav['promoted_hmi_zero_all_seeds']}` | "
            f"{sensor['valid_sample_fraction_median']:.3f} | "
            f"{sensor['median_fringe_contrast_median']:.3f} | "
            f"{sensor['tilt_exceedance_fraction_median']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- next branch: `{next_branch}`",
            f"- second-region diagnosis: `{second_region['diagnosis']}`",
            f"- second-region sensor matches accepted first-region envelope: `{second_region['sensor_matches_accepted_reference']}`",
            f"- second-region dominant rejection reason: `{second_region['sensor']['dominant_rejection_reason']}`",
            "",
            "## Interpretation",
            "",
            "- If the second region fails while sensor health remains acceptable, the remaining bottleneck is map/route distinctiveness and telemetry-aware regional hardening.",
            "- If the second region fails with degraded sensor health, the next branch stays sensor-first and continues photonic calibration hardening.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a compact photonic digital-twin branch checkpoint.")
    parser.add_argument("--region-one-name", required=True)
    parser.add_argument("--region-one-summary", required=True)
    parser.add_argument("--region-one-photonic-summary", required=True)
    parser.add_argument("--region-two-name", required=True)
    parser.add_argument("--region-two-summary", required=True)
    parser.add_argument("--region-two-photonic-summary", required=True)
    parser.add_argument("--calibration-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    region_one = _diagnose_region(
        _nav_summary(
            args.region_one_name,
            _load_rows(Path(args.region_one_summary)),
        ),
        _sensor_summary(
            args.region_one_name,
            _load_rows(Path(args.region_one_photonic_summary)),
        ),
    )
    region_two = _diagnose_region(
        _nav_summary(
            args.region_two_name,
            _load_rows(Path(args.region_two_summary)),
        ),
        _sensor_summary(
            args.region_two_name,
            _load_rows(Path(args.region_two_photonic_summary)),
        ),
    )
    if (not region_two["accepted"]) and bool(region_one["accepted"]) and _sensor_matches_reference(
        region_two["sensor"],
        region_one["sensor"],
    ):
        region_two["sensor_matches_accepted_reference"] = True
        region_two["diagnosis"] = "region_limited"
    calibration_summary = json.loads(Path(args.calibration_summary).read_text(encoding="utf-8"))
    next_branch = _next_branch(region_two)
    payload = {
        "first_region": region_one,
        "second_region": region_two,
        "calibration_summary": calibration_summary,
        "next_branch": next_branch,
    }

    _write_json(output_dir / "photonic_branch_checkpoint_summary.json", payload)
    _write_markdown(
        output_dir / "photonic_branch_checkpoint_report.md",
        first_region=region_one,
        second_region=region_two,
        calibration_summary=calibration_summary,
        next_branch=next_branch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
