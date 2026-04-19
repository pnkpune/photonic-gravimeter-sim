from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gravnav.datasets.bathymetry_loader import (
    BathymetryGrid,
    RegionalBathymetryManifest,
    RegionalDemoPackManifest,
    _resolve_manifest_ref,
    load_bathymetry_grid_from_manifest,
    process_regular_array_bathymetry_grid,
    resolve_regional_demo_pack,
)
from gravnav.datasets.current_loader import process_regular_csv_current_field
from gravnav.datasets.gravity_loader import RegionalGravityMapManifest
from gravnav.datasets.magnetic_loader import process_regular_csv_magnetic_grid
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


def test_process_regular_array_bathymetry_grid_round_trips(tmp_path: Path) -> None:
    raw_tiff = tmp_path / "fixture.tif"
    raw_tiff.write_bytes(b"fixture")
    processed_npz = tmp_path / "fixture_bathymetry.npz"
    manifest_path = tmp_path / "fixture_bathymetry.json"

    grid, manifest = process_regular_array_bathymetry_grid(
        lat_axis_deg=np.array([66.0, 66.1], dtype=np.float64),
        lon_axis_deg=np.array([14.0, 14.1, 14.2], dtype=np.float64),
        elevation_grid_m=np.array(
            [
                [-500.0, -510.0, -520.0],
                [-530.0, -540.0, -550.0],
            ],
            dtype=np.float64,
        ),
        raw_data_path=raw_tiff,
        processed_npz_path=processed_npz,
        manifest_path=manifest_path,
        region_name="fixture_region",
        source_name="fixture_array",
        source_kind="fixture_array_kind",
        project_root=tmp_path,
        metadata_extra={"fixture": True},
    )

    loaded, loaded_manifest, loaded_npz, loaded_manifest_path = load_bathymetry_grid_from_manifest(
        manifest_path
    )

    assert grid.shape == (2, 3)
    assert manifest.source_kind == "fixture_array_kind"
    assert manifest.metadata["fixture"] is True
    assert loaded.shape == (2, 3)
    assert np.allclose(loaded.elevation_grid_m, grid.elevation_grid_m)
    assert loaded_manifest.region_name == "fixture_region"
    assert loaded_npz == processed_npz.resolve()
    assert loaded_manifest_path == manifest_path.resolve()


def test_demo_pack_resolution_uses_manifest_relative_paths(tmp_path: Path) -> None:
    gravity_dir = tmp_path / "gravity"
    bathy_dir = tmp_path / "bathy"
    magnetic_dir = tmp_path / "magnetic"
    current_dir = tmp_path / "current"
    config_dir = tmp_path / "configs"
    gravity_dir.mkdir()
    bathy_dir.mkdir()
    magnetic_dir.mkdir()
    current_dir.mkdir()
    config_dir.mkdir()

    gravity_npz = gravity_dir / "gravity_map.npz"
    gravity_manifest_path = gravity_dir / "gravity_manifest.json"
    bathy_npz = bathy_dir / "bathymetry_grid.npz"
    bathy_manifest_path = bathy_dir / "bathymetry_manifest.json"
    magnetic_csv = magnetic_dir / "magnetic.csv"
    magnetic_npz = magnetic_dir / "magnetic_grid.npz"
    magnetic_manifest_path = magnetic_dir / "magnetic_manifest.json"
    current_csv = current_dir / "current.csv"
    current_npz = current_dir / "current_grid.npz"
    current_manifest_path = current_dir / "current_manifest.json"
    tide_config_path = config_dir / "tide.json"
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
    magnetic_csv.write_text(
        "\n".join(
            [
                "lat_deg,lon_deg,total_field_nt,anomaly_nt",
                "63.6,4.1,50100.0,20.0",
                "63.6,4.2,50110.0,30.0",
                "63.7,4.1,50120.0,40.0",
                "63.7,4.2,50130.0,50.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    process_regular_csv_magnetic_grid(
        magnetic_csv,
        region_name="fixture_magnetic_region",
        source_name="fixture_magnetic",
        processed_map_path=magnetic_npz,
        manifest_path=magnetic_manifest_path,
        project_root=magnetic_dir,
    )
    current_csv.write_text(
        "\n".join(
            [
                "lat_deg,lon_deg,depth_m,north_current_mps,east_current_mps",
                "63.6,4.1,10.0,0.10,0.20",
                "63.6,4.2,10.0,0.15,0.25",
                "63.7,4.1,10.0,0.20,0.30",
                "63.7,4.2,10.0,0.25,0.35",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    process_regular_csv_current_field(
        current_csv,
        region_name="fixture_current_region",
        source_name="fixture_current",
        processed_grid_path=current_npz,
        manifest_path=current_manifest_path,
        project_root=current_dir,
    )
    tide_config_path.write_text(
        json.dumps(
            {
                "name": "fixture_tide",
                "sea_surface_constituents": [
                    {
                        "name": "M2",
                        "angular_frequency_rad_per_s": 0.0001405189,
                        "amplitude_at_equator": 0.5,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    RegionalDemoPackManifest(
        region_name="fixture_demo",
        gravity_manifest_path="gravity/gravity_manifest.json",
        bathymetry_manifest_path="bathy/bathymetry_manifest.json",
        scenario_path="configs/scenario.json",
        sequence_profile_path="configs/profile.json",
        magnetic_manifest_path="magnetic/magnetic_manifest.json",
        current_manifest_path="current/current_manifest.json",
        tide_config_path="configs/tide.json",
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
    assert resolved.magnetic_grid_path == magnetic_npz.resolve()
    assert resolved.current_field_grid_path == current_npz.resolve()
    assert resolved.tide_config_path == tide_config_path.resolve()
    assert resolved.scenario_path == scenario_path.resolve()
    assert resolved.sequence_profile_path == profile_path.resolve()
    assert resolved.gravity_map.shape == (2, 2)
    assert resolved.bathymetry_grid.shape == (2, 2)
    assert resolved.magnetic_grid is not None
    assert resolved.current_field is not None


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
        magnetic_grid_path=Path("/tmp/magnetic_grid.npz"),
        magnetic_manifest_path=Path("/tmp/magnetic_manifest.json"),
        current_field_grid_path=Path("/tmp/current_grid.npz"),
        current_manifest_path=Path("/tmp/current_manifest.json"),
        tide_config_path=Path("/tmp/tide.json"),
        profile_path=Path("/tmp/profile.json"),
        profile={
            "window_size": 11,
            "grid_half_span_m": [100.0, 100.0],
            "grid_spacing_m": [20.0, 20.0],
            "transition_std_m": [15.0, 15.0],
            "center_prior_std_m": [50.0, 50.0],
            "bathymetry_meas_std_m": 2.0,
            "bathymetry_weight": 3.5,
            "bathymetry_gradient_meas_std_m_per_m": 0.01,
            "bathymetry_rugosity_meas_std_m": 2.0,
            "magnetic_meas_std_nt": 8.0,
            "magnetic_gradient_meas_std_nt_per_m": 0.02,
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
                "label": "photonic_gravity_baseline",
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
                "label": "photonic_gravity_tide_acoustic",
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
                "label": "photonic_gravity_tide_acoustic_magnetic_current",
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
    assert "/tmp/magnetic_grid.npz" in report_text
    assert "/tmp/current_grid.npz" in report_text
    assert "/tmp/tide.json" in report_text
    assert "fixture_demo_region" in report_text


def test_run_maritime_demo_hybrid_lag_ins_uses_only_runtime_safe_lag_samples() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_maritime_demo.py"
    spec = importlib.util.spec_from_file_location("run_maritime_demo_test_module_hybrid", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    diagnostics = [
        SimpleNamespace(
            gravity_information_ratio=0.20,
            bathymetry_information_ratio=0.0,
            magnetic_information_ratio=0.0,
            grid_saturated_any=False,
            dominant_failure_mode="informative",
        ),
        SimpleNamespace(
            gravity_information_ratio=0.20,
            bathymetry_information_ratio=0.0,
            magnetic_information_ratio=0.0,
            grid_saturated_any=False,
            dominant_failure_mode="prior_dominated",
        ),
    ]
    hybrid = module._hybrid_lag_ins_from_runtime_signals(
        ins_times_s=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        ins_error_ned_m=np.array(
            [[10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        ins_hmi_horizontal=np.array([False, False, False], dtype=bool),
        lag_times_s=np.array([1.0, 2.0], dtype=np.float64),
        lag_error_ned_m=np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64),
        lag_hmi_horizontal=np.array([False, False], dtype=bool),
        lag_alert_ok=np.array([True, True], dtype=bool),
        sequence_times_s=np.array([1.0, 2.0], dtype=np.float64),
        sequence_diagnostics=diagnostics,
    )

    assert hybrid is not None
    assert hybrid["lag_selected_count"] == 1
    assert np.isclose(hybrid["lag_selected_fraction"], 1.0 / 3.0)
    assert np.isclose(
        hybrid["position_error"].horizontal_rmse_m,
        np.sqrt((10.0 ** 2 + 2.0 ** 2 + 10.0 ** 2) / 3.0),
    )
    assert hybrid["hmi_horizontal"] == 0.0


def test_run_maritime_demo_hybrid_runtime_selector_uses_sequence_when_lag_unavailable() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_maritime_demo.py"
    spec = importlib.util.spec_from_file_location("run_maritime_demo_test_module_sequence", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    diagnostics = [
        SimpleNamespace(
            gravity_information_ratio=0.20,
            bathymetry_information_ratio=0.30,
            magnetic_information_ratio=0.0,
            grid_saturated_any=False,
            dominant_failure_mode="informative",
            edge_mass_fraction=0.02,
            posterior_candidate_ess_fraction=0.10,
            support_radius_n_m=10.0,
            support_radius_e_m=10.0,
            grid_half_span_m=np.array([100.0, 100.0], dtype=np.float64),
        ),
        SimpleNamespace(
            gravity_information_ratio=0.20,
            bathymetry_information_ratio=0.25,
            magnetic_information_ratio=0.0,
            grid_saturated_any=False,
            dominant_failure_mode="informative",
            edge_mass_fraction=0.03,
            posterior_candidate_ess_fraction=0.08,
            support_radius_n_m=12.0,
            support_radius_e_m=12.0,
            grid_half_span_m=np.array([100.0, 100.0], dtype=np.float64),
        ),
    ]
    hybrid = module._hybrid_earth_signature_from_runtime_signals(
        ins_times_s=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        ins_error_ned_m=np.array(
            [[10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        ins_hmi_horizontal=np.array([False, False, False], dtype=bool),
        lag_times_s=np.array([], dtype=np.float64),
        lag_error_ned_m=np.empty((0, 3), dtype=np.float64),
        lag_hmi_horizontal=np.array([], dtype=bool),
        lag_alert_ok=np.array([], dtype=bool),
        lag_protection_level_m=np.array([], dtype=np.float64),
        sequence_times_s=np.array([1.0, 2.0], dtype=np.float64),
        sequence_error_ned_m=np.array([[4.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64),
        sequence_hmi_horizontal=np.array([False, False], dtype=bool),
        sequence_alert_ok=np.array([True, True], dtype=bool),
        sequence_protection_level_m=np.array([40.0, 42.0], dtype=np.float64),
        sequence_diagnostics=diagnostics,
        horizontal_alert_limit_m=100.0,
        enter_consecutive_steps=2,
    )

    assert hybrid is not None
    assert hybrid["lag_selected_count"] == 0
    assert hybrid["sequence_selected_count"] == 1
    assert hybrid["ins_selected_count"] == 2
    assert np.isclose(hybrid["sequence_selected_fraction"], 1.0 / 3.0)
    assert np.isclose(
        hybrid["position_error"].horizontal_rmse_m,
        np.sqrt((10.0 ** 2 + 10.0 ** 2 + 5.0 ** 2) / 3.0),
    )
    assert hybrid["hmi_horizontal"] == 0.0


def test_run_maritime_demo_hybrid_runtime_selector_stays_in_sequence_segment() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_maritime_demo.py"
    spec = importlib.util.spec_from_file_location("run_maritime_demo_test_module_sequence_segment", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    diagnostics = [
        SimpleNamespace(
            gravity_information_ratio=0.30,
            bathymetry_information_ratio=0.40,
            magnetic_information_ratio=0.0,
            grid_saturated_any=False,
            dominant_failure_mode="informative",
            edge_mass_fraction=0.02,
            posterior_candidate_ess_fraction=0.10,
            support_radius_n_m=10.0,
            support_radius_e_m=10.0,
            grid_half_span_m=np.array([100.0, 100.0], dtype=np.float64),
        ),
        SimpleNamespace(
            gravity_information_ratio=0.28,
            bathymetry_information_ratio=0.35,
            magnetic_information_ratio=0.0,
            grid_saturated_any=False,
            dominant_failure_mode="informative",
            edge_mass_fraction=0.03,
            posterior_candidate_ess_fraction=0.09,
            support_radius_n_m=12.0,
            support_radius_e_m=12.0,
            grid_half_span_m=np.array([100.0, 100.0], dtype=np.float64),
        ),
        SimpleNamespace(
            gravity_information_ratio=0.18,
            bathymetry_information_ratio=0.20,
            magnetic_information_ratio=0.0,
            grid_saturated_any=False,
            dominant_failure_mode="informative",
            edge_mass_fraction=0.05,
            posterior_candidate_ess_fraction=0.06,
            support_radius_n_m=16.0,
            support_radius_e_m=16.0,
            grid_half_span_m=np.array([100.0, 100.0], dtype=np.float64),
        ),
    ]
    hybrid = module._hybrid_earth_signature_from_runtime_signals(
        ins_times_s=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        ins_error_ned_m=np.array(
            [[10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        ins_hmi_horizontal=np.array([False, False, False, False], dtype=bool),
        lag_times_s=np.array([], dtype=np.float64),
        lag_error_ned_m=np.empty((0, 3), dtype=np.float64),
        lag_hmi_horizontal=np.array([], dtype=bool),
        lag_alert_ok=np.array([], dtype=bool),
        lag_protection_level_m=np.array([], dtype=np.float64),
        sequence_times_s=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        sequence_error_ned_m=np.array(
            [[4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        sequence_hmi_horizontal=np.array([False, False, False], dtype=bool),
        sequence_alert_ok=np.array([True, True, True], dtype=bool),
        sequence_protection_level_m=np.array([40.0, 42.0, 55.0], dtype=np.float64),
        sequence_diagnostics=diagnostics,
        horizontal_alert_limit_m=100.0,
        enter_consecutive_steps=2,
    )

    assert hybrid is not None
    assert hybrid["lag_selected_count"] == 0
    assert hybrid["sequence_selected_count"] == 2
    assert hybrid["ins_selected_count"] == 2
    assert np.isclose(hybrid["sequence_selected_fraction"], 0.5)
    assert np.isclose(
        hybrid["position_error"].horizontal_rmse_m,
        np.sqrt((10.0 ** 2 + 10.0 ** 2 + 5.0 ** 2 + 6.0 ** 2) / 4.0),
    )


def test_resolve_manifest_ref_remaps_foreign_absolute_prefix(
    tmp_path: Path,
) -> None:
    """Demo packs authored on another checkout (e.g. a Mac-absolute path) must
    still resolve under the current project root when the suffix matches a
    known top-level directory.
    """
    project_root = Path(__file__).resolve().parents[1]
    real_asset = (
        project_root
        / "data"
        / "gravity_maps"
        / "processed"
        / "norwegian_margin_gravity_map_manifest.json"
    )
    assert real_asset.exists(), (
        "Test fixture relies on the tracked Norwegian gravity manifest existing "
        "in the project; reseed it before running this test."
    )
    foreign_absolute = Path(
        "/Users/someone/Downloads/photonic-gravimeter-sim"
        "/data/gravity_maps/processed/norwegian_margin_gravity_map_manifest.json"
    )
    fake_manifest_path = tmp_path / "fake_demo_pack.json"
    fake_manifest_path.write_text("{}", encoding="utf-8")

    resolved = _resolve_manifest_ref(
        foreign_absolute,
        manifest_path=fake_manifest_path,
    )
    assert resolved == real_asset.resolve()


def test_resolve_manifest_ref_returns_original_absolute_when_no_match(
    tmp_path: Path,
) -> None:
    """If the absolute path cannot be remapped, the caller should still see the
    original resolved absolute path so the error it raises is informative.
    """
    fake_manifest_path = tmp_path / "fake_demo_pack.json"
    fake_manifest_path.write_text("{}", encoding="utf-8")
    unresolvable = Path("/definitely/not/a/real/path/nowhere.json")
    resolved = _resolve_manifest_ref(
        unresolvable,
        manifest_path=fake_manifest_path,
    )
    assert resolved == unresolvable.resolve()
