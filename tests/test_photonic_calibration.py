from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from gravnav.sensors.photonic_gravimeter import (
    PhotonicGravimeterSpec,
    PhotonicOperatingMode,
)


def _load_script_module(script_name: str, module_name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_photonic_calibration_presets_load() -> None:
    module = _load_script_module(
        "run_photonic_calibration.py",
        "run_photonic_calibration_test_module",
    )

    lab = module._load_spec(module.LAB_STATIC_CONFIG)
    benign = module._load_spec(module.MARITIME_BENIGN_CONFIG)
    rough = module._load_spec(module.MARITIME_ROUGH_CONFIG)

    assert isinstance(lab, PhotonicGravimeterSpec)
    assert isinstance(benign, PhotonicGravimeterSpec)
    assert isinstance(rough, PhotonicGravimeterSpec)
    assert lab.operating_mode == PhotonicOperatingMode.IDEALIZED.value
    assert benign.operating_mode == PhotonicOperatingMode.MISSION.value
    assert rough.operating_mode == PhotonicOperatingMode.DEGRADED.value
    assert benign.vibration_compensation.residual_correction_fraction > rough.vibration_compensation.residual_correction_fraction


def test_run_photonic_calibration_writes_summary_and_report(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "photonic_calibration"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_photonic_calibration.py",
            "--output-dir",
            str(output_dir),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr

    summary_path = output_dir / "photonic_calibration_summary.json"
    report_path = output_dir / "photonic_calibration_report.md"
    assert summary_path.exists()
    assert report_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(payload) == {"lab_static", "maritime_benign", "maritime_rough"}
    assert "acceptance" in payload["lab_static"]
    assert "vibration_improvement_ratio" in payload["maritime_benign"]
    assert "compensated_summary" in payload["maritime_rough"]


def test_run_maritime_demo_smoke_for_frozen_demo_packs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    packs = [
        ("norwegian_margin", root / "data/bathymetry/processed/norwegian_margin_maritime_demo_pack.json"),
        ("helgeland_offshore", root / "data/bathymetry/processed/helgeland_offshore_demo_pack.json"),
    ]

    for region_name, demo_pack in packs:
        output_dir = tmp_path / region_name
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/run_maritime_demo.py",
                "--demo-pack-manifest",
                str(demo_pack),
                "--output-dir",
                str(output_dir),
                "--seeds",
                "42",
                "--dt-s",
                "10.0",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert proc.returncode == 0, proc.stderr

        summary_path = output_dir / "hardware_tied_maritime_demo_summary.json"
        photonic_path = output_dir / "hardware_tied_maritime_demo_photonic_summary.json"
        report_path = output_dir / "hardware_tied_maritime_demo_report.md"
        assert summary_path.exists()
        assert photonic_path.exists()
        assert report_path.exists()

        rows = json.loads(summary_path.read_text(encoding="utf-8"))
        photonic_rows = json.loads(photonic_path.read_text(encoding="utf-8"))
        report_text = report_path.read_text(encoding="utf-8")

        assert {row["label"] for row in rows} == {
            "live_ins",
            "surrogate_gravity",
            "photonic_gravity",
            "photonic_gravity_bathymetry",
            "photonic_gravity_bathymetry_lag",
        }
        assert all("reported_output_mode" in row for row in rows)
        assert all("reported_output_reason" in row for row in rows)
        assert len(photonic_rows) == 3
        assert region_name in report_text
