from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from gravnav.datasets.bathymetry_loader import RegionalDemoPackManifest


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_public_multimodal_norway.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prepare_public_multimodal_norway_test_module",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decimal_year_from_iso8601_is_stable() -> None:
    module = _load_module()
    value = module._decimal_year_from_iso8601("2026-04-14T03:00:00Z")
    assert 2026.28 < value < 2026.29


def test_parse_wmm_output_total_field(tmp_path: Path) -> None:
    module = _load_module()
    output_path = tmp_path / "sample_output.txt"
    output_path.write_text(
        "\n".join(
            [
                "Date Coord-System Altitude Latitude Longitude D_deg D_min I_deg I_min H_nT X_nT Y_nT Z_nT F_nT dD_min dI_min dH_nT dX_nT dY_nT dZ_nT dF_nT",
                "2026.287 M M0.000 63.508351 3.980000 16d 50m 79d 14m 9803.8 9384.0 2838.2 51581.4 52504.8 15.0 1.4 -11.4 -23.3 37.8 51.2 48.2",
                "2026.287 M M0.000 63.508351 4.010000 16d 42m 79d 14m 9809.5 9395.6 2819.3 51555.8 52480.7 15.0 1.4 -11.5 -23.3 37.8 51.2 48.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = module._parse_wmm_output_total_field(output_path)

    assert np.isclose(rows[(63.508351, 3.98)], 52504.8)
    assert np.isclose(rows[(63.508351, 4.01)], 52480.7)


def test_parse_hycom_point_csv_maps_east_and_north_correctly() -> None:
    module = _load_module()
    payload = (
        'time,latitude[unit="degrees_north"],longitude[unit="degrees_east"],'
        'vertCoord[unit="m"],water_u[unit="m/s"],water_v[unit="m/s"]\n'
        "2026-04-14T03:00:00Z,63.7,4.2,0.0,-0.07200000435113907,0.16100001335144043\n"
    )

    time_iso, north_mps, east_mps = module._parse_hycom_point_csv(payload)

    assert time_iso == "2026-04-14T03:00:00Z"
    assert np.isclose(north_mps, 0.16100001335144043)
    assert np.isclose(east_mps, -0.07200000435113907)


def test_augment_demo_pack_adds_priority9_paths(tmp_path: Path) -> None:
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
        tide_config_path=tmp_path / "tide.json",
        snapshot_time_iso="2026-04-14T03:00:00Z",
        decimal_year=2026.2866438356164,
    )

    assert out.magnetic_manifest_path == str((tmp_path / "magnetic.json").resolve())
    assert out.current_manifest_path == str((tmp_path / "current.json").resolve())
    assert out.tide_config_path == str((tmp_path / "tide.json").resolve())
    assert out.metadata["priority9_multimodal"] is True
    assert out.metadata["snapshot_time_iso"] == "2026-04-14T03:00:00Z"
    assert "Magnetic map prepared from official NOAA WMMHR2025 total field." in out.notes


def test_fill_invalid_current_rows_uses_nearest_valid_fill() -> None:
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
    assert stats["all_invalid_fallback_zero"] is False
    assert filled_lookup[(63.0, 4.1)] == (0.10, -0.20)


def test_fill_invalid_current_rows_zero_fallback_when_all_invalid() -> None:
    module = _load_module()
    lat_axis = np.array([63.0, 63.1], dtype=np.float64)
    lon_axis = np.array([4.0, 4.1], dtype=np.float64)
    rows = [
        (63.0, 4.0, np.nan, np.nan),
        (63.0, 4.1, np.nan, np.nan),
        (63.1, 4.0, np.nan, np.nan),
        (63.1, 4.1, np.nan, np.nan),
    ]

    filled_rows, stats = module._fill_invalid_current_rows(
        rows,
        lat_axis_deg=lat_axis,
        lon_axis_deg=lon_axis,
    )

    assert stats["num_invalid_points"] == 4
    assert stats["fill_strategy"] == "all_zero_fallback"
    assert stats["all_invalid_fallback_zero"] is True
    assert all(north == 0.0 and east == 0.0 for _, _, north, east in filled_rows)
