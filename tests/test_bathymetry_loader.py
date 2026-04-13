from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gravnav.datasets.bathymetry_loader import (
    BathymetryGrid,
    RegionalBathymetryManifest,
    RegionalDemoPackManifest,
    load_bathymetry_grid_from_manifest,
    resolve_regional_demo_pack,
)
from gravnav.datasets.gravity_loader import RegionalGravityMapManifest
from gravnav.physics.gravity_map import GravityGridMap


def test_bathymetry_grid_water_depth_and_bounds() -> None:
    grid = BathymetryGrid(
        lat_axis_deg=np.array([63.6, 63.7]),
        lon_axis_deg=np.array([4.1, 4.2]),
        elevation_grid_m=np.array(
            [
                [-1200.0, -1180.0],
                [-1210.0, -1190.0],
            ]
        ),
        reference_surface_height_m=0.0,
        fill_value_m=np.nan,
    )

    assert bool(grid.contains(63.65, 4.15))
    assert not bool(grid.contains(63.5, 4.15))

    depth_m = float(grid.evaluate_water_depth_m(63.65, 4.15))
    assert 1180.0 < depth_m < 1210.0


def test_bathymetry_manifests_round_trip() -> None:
    bathy = RegionalBathymetryManifest(
        region_name="norwegian_margin",
        source_name="gebco_fixture",
        source_kind="regular_grid_csv",
        raw_data_path="data/bathymetry/raw/example.csv",
        processed_grid_path="data/bathymetry/processed/example.npz",
        manifest_path="data/bathymetry/processed/example.json",
        elevation_units="m",
        reference_surface_height_m=0.0,
        lat_bounds_deg=(63.6, 63.8),
        lon_bounds_deg=(4.0, 4.4),
        spacing_deg=(0.02, 0.03),
        shape=(11, 15),
        interpolation="linear",
        notes=["fixture"],
        metadata={"source": "test"},
    )
    demo = RegionalDemoPackManifest(
        region_name="norwegian_margin_demo",
        gravity_manifest_path="data/gravity_maps/processed/example.json",
        bathymetry_manifest_path="data/bathymetry/processed/example.json",
        scenario_path="configs/scenarios/norwegian_margin_maritime.json",
        sequence_profile_path="configs/sequence_profiles/norwegian_margin_maritime_demo.json",
        notes=["demo"],
        metadata={"mode": "test"},
    )

    assert RegionalBathymetryManifest.from_mapping(bathy.to_mapping()) == bathy
    assert RegionalDemoPackManifest.from_mapping(demo.to_mapping()) == demo


def test_demo_pack_resolution_uses_manifest_relative_paths(tmp_path: Path) -> None:
    gravity_dir = tmp_path / "gravity"
    bathy_dir = tmp_path / "bathy"
    config_dir = tmp_path / "configs"
    gravity_dir.mkdir()
    bathy_dir.mkdir()
    config_dir.mkdir()

    gravity_npz = gravity_dir / "gravity_map.npz"
    gravity_manifest_path = gravity_dir / "gravity_manifest.json"
    bathy_npz = bathy_dir / "bathymetry_grid.npz"
    bathy_manifest_path = bathy_dir / "bathymetry_manifest.json"
    scenario_path = config_dir / "scenario.json"
    profile_path = config_dir / "profile.json"
    demo_pack_path = tmp_path / "demo_pack.json"

    gravity_map = GravityGridMap.from_mgal_grid(
        lat_axis_rad=np.deg2rad(np.array([63.6, 63.7])),
        lon_axis_rad=np.deg2rad(np.array([4.1, 4.2])),
        disturbance_grid_mgal=np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float64),
        reference_height_m=0.0,
        default_method="linear",
        bounds_error=False,
        fill_value_mgal=np.nan,
        name="fixture_gravity",
    )
    gravity_map.to_npz(gravity_npz)
    RegionalGravityMapManifest(
        region_name="fixture_gravity_region",
        source_name="fixture_gravity",
        source_kind="regular_grid_csv",
        raw_data_path="raw/gravity.csv",
        processed_map_path="gravity_map.npz",
        manifest_path="gravity_manifest.json",
        disturbance_units="mGal",
        reference_height_m=0.0,
        lat_bounds_deg=(63.6, 63.7),
        lon_bounds_deg=(4.1, 4.2),
        spacing_deg=(0.1, 0.1),
        shape=(2, 2),
        interpolation="linear",
    ).write_json(gravity_manifest_path)

    bathy_grid = BathymetryGrid(
        lat_axis_deg=np.array([63.6, 63.7]),
        lon_axis_deg=np.array([4.1, 4.2]),
        elevation_grid_m=np.array(
            [[-1200.0, -1180.0], [-1210.0, -1190.0]],
            dtype=np.float64,
        ),
        reference_surface_height_m=0.0,
        default_method="linear",
        bounds_error=False,
        fill_value_m=np.nan,
        name="fixture_bathy",
    )
    bathy_grid.to_npz(bathy_npz)
    RegionalBathymetryManifest(
        region_name="fixture_bathy_region",
        source_name="fixture_bathy",
        source_kind="regular_grid_csv",
        raw_data_path="raw/bathy.csv",
        processed_grid_path="bathymetry_grid.npz",
        manifest_path="bathymetry_manifest.json",
        elevation_units="m",
        reference_surface_height_m=0.0,
        lat_bounds_deg=(63.6, 63.7),
        lon_bounds_deg=(4.1, 4.2),
        spacing_deg=(0.1, 0.1),
        shape=(2, 2),
        interpolation="linear",
    ).write_json(bathy_manifest_path)

    scenario_path.write_text('{"name":"demo"}\n', encoding="utf-8")
    profile_path.write_text('{"window_size":11}\n', encoding="utf-8")

    RegionalDemoPackManifest(
        region_name="fixture_demo",
        gravity_manifest_path="gravity/gravity_manifest.json",
        bathymetry_manifest_path="bathy/bathymetry_manifest.json",
        scenario_path="configs/scenario.json",
        sequence_profile_path="configs/profile.json",
    ).write_json(demo_pack_path)

    bathy_loaded, bathy_manifest_loaded, bathy_grid_path, loaded_bathy_manifest_path = (
        load_bathymetry_grid_from_manifest(bathy_manifest_path)
    )
    assert bathy_loaded.shape == (2, 2)
    assert bathy_manifest_loaded.region_name == "fixture_bathy_region"
    assert bathy_grid_path == bathy_npz.resolve()
    assert loaded_bathy_manifest_path == bathy_manifest_path.resolve()

    resolved = resolve_regional_demo_pack(demo_pack_path)
    assert resolved.manifest.region_name == "fixture_demo"
    assert resolved.gravity_manifest.region_name == "fixture_gravity_region"
    assert resolved.bathymetry_manifest.region_name == "fixture_bathy_region"
    assert resolved.gravity_map_path == gravity_npz.resolve()
    assert resolved.bathymetry_grid_path == bathy_npz.resolve()
    assert resolved.scenario_path == scenario_path.resolve()
    assert resolved.sequence_profile_path == profile_path.resolve()
    assert resolved.gravity_map.shape == (2, 2)
    assert resolved.bathymetry_grid.shape == (2, 2)


def test_run_maritime_demo_report_records_loaded_asset_sources(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_maritime_demo.py"
    spec = importlib.util.spec_from_file_location("run_maritime_demo_test_module", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    report_path = tmp_path / "report.md"
    module._write_report(
        report_path,
        scenario=SimpleNamespace(name="fixture_scenario"),
        demo_pack_path=Path("/tmp/demo_pack.json"),
        demo_pack_region="fixture_demo_region",
        gravity_map_path=Path("/tmp/gravity_map.npz"),
        gravity_manifest_path=Path("/tmp/gravity_manifest.json"),
        bathymetry_grid_path=Path("/tmp/bathy_grid.npz"),
        bathymetry_manifest_path=Path("/tmp/bathy_manifest.json"),
        profile_path=Path("/tmp/profile.json"),
        profile={
            "window_size": 11,
            "grid_half_span_m": [100.0, 100.0],
            "grid_spacing_m": [20.0, 20.0],
            "transition_std_m": [15.0, 15.0],
            "center_prior_std_m": [50.0, 50.0],
            "bathymetry_meas_std_m": 2.0,
            "bathymetry_weight": 3.5,
            "map_match_every_steps": 1,
        },
        rows=[
            {
                "label": "live_ins",
                "ins_horizontal_rmse_m": 10.0,
                "earth_signature_horizontal_rmse_m": 10.0,
                "ins_cep95_m": 20.0,
                "earth_signature_cep95_m": 20.0,
                "earth_signature_hmi_horizontal": 0.0,
                "earth_signature_mode": "live_ins",
            },
            {
                "label": "photonic_gravity",
                "ins_horizontal_rmse_m": 10.0,
                "sequence_horizontal_rmse_m": 9.0,
                "earth_signature_horizontal_rmse_m": 9.0,
                "ins_cep95_m": 20.0,
                "sequence_cep95_m": 18.0,
                "earth_signature_cep95_m": 18.0,
                "earth_signature_hmi_horizontal": 0.0,
                "earth_signature_mode": "sequence",
            },
            {
                "label": "photonic_gravity_bathymetry",
                "ins_horizontal_rmse_m": 10.0,
                "sequence_horizontal_rmse_m": 8.0,
                "earth_signature_horizontal_rmse_m": 8.0,
                "ins_cep95_m": 20.0,
                "sequence_cep95_m": 16.0,
                "earth_signature_cep95_m": 16.0,
                "earth_signature_hmi_horizontal": 0.0,
                "earth_signature_mode": "sequence",
            },
            {
                "label": "photonic_gravity_bathymetry_lag",
                "ins_horizontal_rmse_m": 10.0,
                "lag_horizontal_rmse_m": 7.5,
                "earth_signature_horizontal_rmse_m": 7.5,
                "ins_cep95_m": 20.0,
                "lag_cep95_m": 15.0,
                "earth_signature_cep95_m": 15.0,
                "earth_signature_hmi_horizontal": 0.0,
                "earth_signature_mode": "lag_smoothed",
            },
        ],
        seeds=[42],
        lag_validated=True,
        dt_s=2.0,
        initial_position_offset_ned_m=(60.0, -30.0, 0.0),
    )
    report_text = report_path.read_text(encoding="utf-8")
    assert "demo pack manifest" in report_text
    assert "/tmp/gravity_map.npz" in report_text
    assert "/tmp/gravity_manifest.json" in report_text
    assert "/tmp/bathy_grid.npz" in report_text
    assert "/tmp/bathy_manifest.json" in report_text
    assert "fixture_demo_region" in report_text
