from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

import pytest


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_offline_multimodal_shim.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prepare_offline_multimodal_shim_test_module",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fixture_gravity_manifest(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    from gravnav.datasets.gravity_loader import RegionalGravityMapManifest
    from gravnav.physics.gravity_map import GravityGridMap

    lat_axis_deg = np.linspace(34.0, 36.0, 41)
    lon_axis_deg = np.linspace(-32.0, -30.0, 41)
    lat_grid, lon_grid = np.meshgrid(
        np.deg2rad(lat_axis_deg),
        np.deg2rad(lon_axis_deg),
        indexing="ij",
    )
    disturbance_mgal = (
        40.0 * np.sin(55.0 * (lat_grid - np.deg2rad(35.0)))
        + 20.0 * np.cos(60.0 * (lon_grid - np.deg2rad(-31.0)))
    )
    map_model = GravityGridMap.from_mgal_grid(
        lat_axis_rad=np.deg2rad(lat_axis_deg),
        lon_axis_rad=np.deg2rad(lon_axis_deg),
        disturbance_grid_mgal=disturbance_mgal,
        reference_height_m=0.0,
        default_method="linear",
        bounds_error=False,
        fill_value_mgal=np.nan,
        name="fixture_gravity",
        metadata={"region_name": "fixture_region"},
    )
    grid_path = tmp_path / "fixture_gravity.npz"
    map_model.to_npz(grid_path)
    manifest = RegionalGravityMapManifest(
        region_name="fixture_region",
        source_name="fixture_source",
        source_kind="regular_grid_csv",
        raw_data_path=str(grid_path),
        processed_map_path=str(grid_path),
        manifest_path=str(tmp_path / "fixture_gravity_manifest.json"),
        disturbance_units="mGal",
        reference_height_m=0.0,
        lat_bounds_deg=(float(lat_axis_deg[0]), float(lat_axis_deg[-1])),
        lon_bounds_deg=(float(lon_axis_deg[0]), float(lon_axis_deg[-1])),
        spacing_deg=(
            float(np.mean(np.diff(lat_axis_deg))),
            float(np.mean(np.diff(lon_axis_deg))),
        ),
        shape=(int(lat_axis_deg.size), int(lon_axis_deg.size)),
        interpolation="linear",
    )
    manifest_path = tmp_path / "fixture_gravity_manifest.json"
    manifest.write_json(manifest_path)

    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(
        json.dumps(
            {
                "name": "fixture_scenario",
                "initial_lat_deg": 35.0,
                "initial_lon_deg": -31.0,
                "initial_height_m": 0.0,
                "initial_heading_deg": 0.0,
                "default_dt_s": 1.0,
                "description": "fixture",
                "metadata": {},
                "segments": [
                    {
                        "type": "straight",
                        "duration_s": 20.0,
                        "speed_mps": 1.0,
                        "flight_path_angle_deg": 0.0,
                        "roll_deg": 0.0,
                        "label": "leg_1",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sequence_path = tmp_path / "sequence_profile.json"
    sequence_path.write_text(
        json.dumps(
            {
                "window_size": 4,
                "grid_half_span_m": [200.0, 200.0],
                "grid_spacing_m": [50.0, 50.0],
                "transition_std_m": [30.0, 30.0],
                "center_prior_std_m": [100.0, 100.0],
                "gravity_meas_std_mps2": 1.0e-7,
                "gradient_meas_std_per_s2": 1.0e-10,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path, scenario_path, sequence_path


def test_prepare_offline_multimodal_shim_writes_expected_manifests(
    tmp_path: Path,
) -> None:
    module = _load_module()
    manifest_path, scenario_path, sequence_path = _write_fixture_gravity_manifest(tmp_path)

    processed_bathy = tmp_path / "bathy_grid.npz"
    bathy_manifest = tmp_path / "bathy_manifest.json"
    demo_pack_manifest = tmp_path / "demo_pack_manifest.json"

    rc = module.main(
        [
            "--region-name",
            "fixture_region",
            "--gravity-manifest",
            str(manifest_path),
            "--scenario",
            str(scenario_path),
            "--sequence-profile",
            str(sequence_path),
            "--processed-bathymetry",
            str(processed_bathy),
            "--bathymetry-manifest",
            str(bathy_manifest),
            "--demo-pack-manifest",
            str(demo_pack_manifest),
            "--mean-depth-m",
            "-3500.0",
            "--admittance-m-per-mgal",
            "22.0",
        ]
    )
    assert rc == 0
    assert processed_bathy.exists()
    assert bathy_manifest.exists()
    assert demo_pack_manifest.exists()

    bathy_payload = json.loads(bathy_manifest.read_text(encoding="utf-8"))
    assert bathy_payload["region_name"] == "fixture_region"
    assert bathy_payload["source_kind"] == "gravity_admittance_proxy"
    assert bathy_payload["shape"] == [41, 41]

    demo_payload = json.loads(demo_pack_manifest.read_text(encoding="utf-8"))
    assert demo_payload["region_name"] == "fixture_region"
    assert demo_payload["gravity_manifest_path"] == str(manifest_path)
    assert demo_payload["bathymetry_manifest_path"] == str(bathy_manifest)
    assert demo_payload["magnetic_manifest_path"] is None
    assert demo_payload["current_manifest_path"] is None
    assert demo_payload["metadata"]["offline_build"] is True


def test_prepare_offline_multimodal_shim_proxy_structure_tracks_gravity(
    tmp_path: Path,
) -> None:
    """The proxy must preserve spatial structure of the source gravity field."""
    module = _load_module()
    manifest_path, scenario_path, sequence_path = _write_fixture_gravity_manifest(tmp_path)

    processed_bathy = tmp_path / "bathy_grid.npz"
    bathy_manifest = tmp_path / "bathy_manifest.json"
    demo_pack_manifest = tmp_path / "demo_pack_manifest.json"

    module.main(
        [
            "--region-name",
            "fixture_region",
            "--gravity-manifest",
            str(manifest_path),
            "--scenario",
            str(scenario_path),
            "--sequence-profile",
            str(sequence_path),
            "--processed-bathymetry",
            str(processed_bathy),
            "--bathymetry-manifest",
            str(bathy_manifest),
            "--demo-pack-manifest",
            str(demo_pack_manifest),
        ]
    )

    from gravnav.datasets.bathymetry_loader import BathymetryGrid
    from gravnav.datasets.gravity_loader import load_regional_map_from_manifest

    bathy = BathymetryGrid.from_npz(processed_bathy)
    gmap, _, _, _ = load_regional_map_from_manifest(manifest_path)
    gravity_mgal = np.asarray(gmap.disturbance_grid_mgal, dtype=np.float64)
    elevation = np.asarray(bathy.elevation_grid_m, dtype=np.float64)

    # Elevations must be in the expected ocean range.
    assert np.all(elevation >= module.MIN_PROXY_ELEVATION_M)
    assert np.all(elevation <= module.MAX_PROXY_ELEVATION_M)

    # The proxy should have a strong linear correlation with the gravity field
    # it was derived from.
    flat_grav = gravity_mgal.reshape(-1)
    flat_elev = elevation.reshape(-1)
    corr = float(np.corrcoef(flat_grav, flat_elev)[0, 1])
    assert corr > 0.95
