from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from gravnav.datasets.bathymetry_loader import RegionalDemoPackManifest


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_public_multimodal_region.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prepare_public_multimodal_region_test_module",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generic_multimodal_decimal_year_from_iso8601_is_stable() -> None:
    module = _load_module()
    value = module._decimal_year_from_iso8601("2026-04-14T03:00:00Z")
    assert 2026.28 < value < 2026.29


def test_generic_multimodal_augment_demo_pack_allows_no_tide_config(tmp_path: Path) -> None:
    module = _load_module()
    manifest = RegionalDemoPackManifest(
        region_name="fixture_region",
        gravity_manifest_path="gravity.json",
        bathymetry_manifest_path="bathy.json",
        scenario_path="scenario.json",
        sequence_profile_path="profile.json",
        notes=["base"],
        metadata={"seed": 42},
    )
    out = module._augment_demo_pack(
        manifest,
        magnetic_manifest_path=tmp_path / "magnetic.json",
        current_manifest_path=tmp_path / "current.json",
        tide_config_path=None,
        snapshot_time_iso="2026-04-14T03:00:00Z",
        decimal_year=2026.2866438356164,
    )

    assert out.magnetic_manifest_path == str((tmp_path / "magnetic.json").resolve())
    assert out.current_manifest_path == str((tmp_path / "current.json").resolve())
    assert out.tide_config_path is None
    assert out.metadata["multimodal_public_pack"] is True


def test_generic_multimodal_fill_invalid_current_rows_uses_nearest_valid_fill() -> None:
    module = _load_module()
    lat_axis = np.array([63.0, 63.1], dtype=np.float64)
    lon_axis = np.array([4.0, 4.1], dtype=np.float64)
    rows = [
        (63.0, 4.0, 0.10, -0.20),
        (63.0, 4.1, np.nan, np.nan),
        (63.1, 4.0, 0.30, -0.40),
        (63.1, 4.1, 0.50, -0.60),
    ]

    filled_rows, stats = module._fill_invalid_current_rows(
        rows,
        lat_axis_deg=lat_axis,
        lon_axis_deg=lon_axis,
    )

    filled_lookup = {(lat, lon): (north, east) for lat, lon, north, east in filled_rows}
    assert stats["num_invalid_points"] == 1
    assert stats["fill_strategy"] == "nearest_valid"
    assert filled_lookup[(63.0, 4.1)] == (0.10, -0.20)
