from __future__ import annotations

import numpy as np

from gravnav.datasets.bathymetry_loader import (
    BathymetryGrid,
    RegionalBathymetryManifest,
    RegionalDemoPackManifest,
)


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
