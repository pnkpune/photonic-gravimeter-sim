from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def _rows(
    *,
    ins_rmse: float,
    ins_cep95: float,
    promoted: list[tuple[int, float, float, float]],
) -> list[dict[str, float | int | str]]:
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


def test_two_region_validation_report_selects_hardening_branch(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    region_one_summary = tmp_path / "region_one.json"
    region_two_summary = tmp_path / "region_two.json"
    output_dir = tmp_path / "out"

    region_one_summary.write_text(
        json.dumps(
            _rows(
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
    region_two_summary.write_text(
        json.dumps(
            _rows(
                ins_rmse=160.0,
                ins_cep95=280.0,
                promoted=[
                    (42, 200.0, 330.0, 0.8),
                    (123, 150.0, 300.0, 0.7),
                    (777, 170.0, 310.0, 0.8),
                ],
            )
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/generate_two_region_validation_report.py",
            "--region-one-name",
            "norwegian_margin",
            "--region-one-summary",
            str(region_one_summary),
            "--region-two-name",
            "helgeland_offshore",
            "--region-two-summary",
            str(region_two_summary),
            "--output-dir",
            str(output_dir),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr

    payload = json.loads((output_dir / "two_region_validation_summary.json").read_text())
    second_report = (output_dir / "second_region_validation_report.md").read_text(
        encoding="utf-8"
    )
    combined_report = (output_dir / "two_region_validation_report.md").read_text(
        encoding="utf-8"
    )

    assert payload["overall_pass"] is False
    assert payload["next_branch"] == "feature/regional-demo-pack-hardening"
    assert payload["regions"][0]["accepted"] is True
    assert payload["regions"][1]["accepted"] is False
    assert "helgeland_offshore" in second_report
    assert "FAIL" in second_report
    assert "feature/regional-demo-pack-hardening" in combined_report
