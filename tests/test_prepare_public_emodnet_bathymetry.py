from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from gravnav.datasets.bathymetry_loader import RegionalDemoPackManifest


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_public_emodnet_bathymetry.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prepare_public_emodnet_bathymetry_test_module",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_emodnet_tiff_flips_depth_to_elevation(tmp_path: Path) -> None:
    module = _load_module()
    tiff_path = tmp_path / "tile.tif"
    depth_grid = np.array(
        [
            [100.0, 110.0, 120.0],
            [200.0, 210.0, 220.0],
        ],
        dtype=np.float32,
    )
    Image.fromarray(depth_grid, mode="F").save(tiff_path)

    lat_axis, lon_axis, elevation_grid, metadata = module._parse_emodnet_tiff(
        tiff_path,
        lat_min_deg=66.0,
        lat_max_deg=66.0021,
        lon_min_deg=14.0,
        lon_max_deg=14.0032,
    )

    assert lat_axis.shape == (2,)
    assert lon_axis.shape == (3,)
    assert np.all(np.diff(lat_axis) > 0.0)
    assert np.all(np.diff(lon_axis) > 0.0)
    assert np.allclose(
        elevation_grid,
        np.array(
            [
                [-200.0, -210.0, -220.0],
                [-100.0, -110.0, -120.0],
            ],
            dtype=np.float64,
        ),
    )
    assert metadata["fill_strategy"] == "none"
    assert metadata["num_invalid_points"] == 0


def test_parse_emodnet_tiff_fills_invalid_cells(tmp_path: Path) -> None:
    module = _load_module()
    tiff_path = tmp_path / "tile_invalid.tif"
    depth_grid = np.array(
        [
            [100.0, np.float32(module.EMODNET_NODATA_ABS_THRESHOLD), 120.0],
            [200.0, 210.0, 220.0],
        ],
        dtype=np.float32,
    )
    Image.fromarray(depth_grid, mode="F").save(tiff_path)

    _, _, elevation_grid, metadata = module._parse_emodnet_tiff(
        tiff_path,
        lat_min_deg=66.0,
        lat_max_deg=66.0021,
        lon_min_deg=14.0,
        lon_max_deg=14.0032,
    )

    assert metadata["fill_strategy"] == "nearest_valid"
    assert metadata["num_invalid_points"] == 1
    assert np.isfinite(elevation_grid).all()


def test_augment_demo_pack_replaces_bathymetry_manifest(tmp_path: Path) -> None:
    module = _load_module()
    manifest = RegionalDemoPackManifest(
        region_name="fixture_region",
        gravity_manifest_path="gravity.json",
        bathymetry_manifest_path="bathy.json",
        scenario_path="scenario.json",
        sequence_profile_path="profile.json",
        magnetic_manifest_path="magnetic.json",
        current_manifest_path="current.json",
        tide_config_path="tide.json",
        notes=["base"],
        metadata={"seed": 42},
    )
    out = module._augment_demo_pack(
        manifest,
        bathymetry_manifest_path=tmp_path / "emodnet_bathy.json",
        coverage_id="emodnet__mean_2022",
    )

    assert out.bathymetry_manifest_path == str((tmp_path / "emodnet_bathy.json").resolve())
    assert out.magnetic_manifest_path == "magnetic.json"
    assert out.current_manifest_path == "current.json"
    assert out.tide_config_path == "tide.json"
    assert out.metadata["emodnet_bathymetry"] is True
    assert out.metadata["emodnet_coverage_id"] == "emodnet__mean_2022"
