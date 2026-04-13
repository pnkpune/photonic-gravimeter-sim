from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def _nav_rows(*, ins_rmse: float, ins_cep95: float, promoted: list[tuple[int, float, float, float]]):
    rows: list[dict[str, float | int | str]] = []
    for seed in (42, 123, 777):
        rows.append(
            {
                "label": "live_ins",
                "seed": seed,
                "ins_horizontal_rmse_m": ins_rmse,
                "ins_cep95_m": ins_cep95,
                "earth_signature_horizontal_rmse_m": ins_rmse,
                "earth_signature_cep95_m": ins_cep95,
                "earth_signature_hmi_horizontal": 0.0,
                "earth_signature_mode": "live_ins",
            }
        )
    for seed, rmse, cep95, hmi in promoted:
        rows.append(
            {
                "label": "photonic_gravity_bathymetry_lag",
                "seed": seed,
                "ins_horizontal_rmse_m": ins_rmse,
                "ins_cep95_m": ins_cep95,
                "earth_signature_horizontal_rmse_m": rmse,
                "earth_signature_cep95_m": cep95,
                "earth_signature_hmi_horizontal": hmi,
                "earth_signature_mode": "lag_smoothed",
            }
        )
    return rows


def _photonic_rows(
    *,
    valid_fraction: float,
    contrast: float,
    tilt_fraction: float,
    dominant_reason: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in (42, 123, 777):
        rows.append(
            {
                "label": "photonic_gravity_bathymetry_lag",
                "seed": seed,
                "valid_sample_fraction": valid_fraction,
                "median_fringe_contrast": contrast,
                "p95_fringe_contrast": contrast,
                "rms_vibration_residual_phase_rad": 0.02,
                "rms_disturbance_residual_mps2": 1.0e-5,
                "tilt_exceedance_fraction": tilt_fraction,
                "median_estimated_measurement_variance_mps4": 1.0e-10,
                "rejection_reason_counts": {dominant_reason: 3},
            }
        )
    return rows


def test_generate_photonic_branch_checkpoint_selects_regional_hardening(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "out"
    region_one_summary = tmp_path / "region_one_summary.json"
    region_one_photonic = tmp_path / "region_one_photonic.json"
    region_two_summary = tmp_path / "region_two_summary.json"
    region_two_photonic = tmp_path / "region_two_photonic.json"
    calibration_summary = tmp_path / "calibration.json"

    region_one_summary.write_text(
        json.dumps(
            _nav_rows(
                ins_rmse=240.0,
                ins_cep95=420.0,
                promoted=[
                    (42, 190.0, 360.0, 0.0),
                    (123, 180.0, 340.0, 0.0),
                    (777, 200.0, 380.0, 0.0),
                ],
            )
        ),
        encoding="utf-8",
    )
    region_one_photonic.write_text(
        json.dumps(
            _photonic_rows(
                valid_fraction=0.96,
                contrast=0.58,
                tilt_fraction=0.03,
                dominant_reason="low_contrast",
            )
        ),
        encoding="utf-8",
    )
    region_two_summary.write_text(
        json.dumps(
            _nav_rows(
                ins_rmse=160.0,
                ins_cep95=280.0,
                promoted=[
                    (42, 195.0, 330.0, 0.7),
                    (123, 185.0, 320.0, 0.8),
                    (777, 175.0, 310.0, 0.8),
                ],
            )
        ),
        encoding="utf-8",
    )
    region_two_photonic.write_text(
        json.dumps(
            _photonic_rows(
                valid_fraction=0.92,
                contrast=0.54,
                tilt_fraction=0.05,
                dominant_reason="low_contrast",
            )
        ),
        encoding="utf-8",
    )
    calibration_summary.write_text(
        json.dumps(
            {
                "lab_static": {"acceptance": {"a": True}},
                "maritime_benign": {"acceptance": {"a": True}},
                "maritime_rough": {"acceptance": {"a": True}},
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/generate_photonic_branch_checkpoint.py",
            "--region-one-name",
            "norwegian_margin",
            "--region-one-summary",
            str(region_one_summary),
            "--region-one-photonic-summary",
            str(region_one_photonic),
            "--region-two-name",
            "helgeland_offshore",
            "--region-two-summary",
            str(region_two_summary),
            "--region-two-photonic-summary",
            str(region_two_photonic),
            "--calibration-summary",
            str(calibration_summary),
            "--output-dir",
            str(output_dir),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr

    payload = json.loads(
        (output_dir / "photonic_branch_checkpoint_summary.json").read_text(encoding="utf-8")
    )
    report_text = (
        output_dir / "photonic_branch_checkpoint_report.md"
    ).read_text(encoding="utf-8")

    assert payload["second_region"]["diagnosis"] == "region_limited"
    assert payload["next_branch"] == "feature/regional-telemetry-hardening"
    assert "helgeland_offshore" in report_text
    assert "feature/regional-telemetry-hardening" in report_text
