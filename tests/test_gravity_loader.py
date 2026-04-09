from __future__ import annotations

import json
from pathlib import Path
import pytest

from gravnav.datasets.gravity_loader import (
    load_regional_manifest,
    load_regular_csv_gravity_map,
    load_regular_xyz_gravity_map,
    process_regular_csv_gravity_map,
)
from gravnav.truth.scenarios import ScenarioSpec, build_truth_trajectory_from_scenario
from gravnav.utils.config import load_config_mapping


def test_load_regular_csv_gravity_map_fixture() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data/gravity_maps/raw/norwegian_margin/norwegian_margin_fixture.csv"
    map_model = load_regular_csv_gravity_map(
        raw_path,
        name="norwegian_margin_fixture",
        region_name="norwegian_margin",
        source_name="fixture",
    )

    assert map_model.shape == (7, 9)
    assert float(map_model.lat_axis_deg[0]) == 63.58
    assert float(map_model.lat_axis_deg[-1]) == 63.82
    assert float(map_model.lon_axis_deg[0]) == 4.0
    assert float(map_model.lon_axis_deg[-1]) == 4.48
    assert map_model.metadata["region_name"] == "norwegian_margin"


def test_process_gravity_map_manifest_roundtrip(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data/gravity_maps/raw/norwegian_margin/norwegian_margin_fixture.csv"
    processed_path = tmp_path / "norwegian_margin_test_map.npz"
    manifest_path = tmp_path / "norwegian_margin_test_manifest.json"

    map_model, manifest = process_regular_csv_gravity_map(
        raw_path,
        processed_npz_path=processed_path,
        manifest_path=manifest_path,
        region_name="norwegian_margin",
        source_name="fixture",
        project_root=root,
    )

    loaded_manifest = load_regional_manifest(manifest_path)
    assert processed_path.exists()
    assert manifest_path.exists()
    assert loaded_manifest.region_name == "norwegian_margin"
    assert loaded_manifest.shape == map_model.shape
    assert loaded_manifest.lat_bounds_deg == manifest.lat_bounds_deg
    assert loaded_manifest.lon_bounds_deg == manifest.lon_bounds_deg
    assert json.loads(manifest_path.read_text())["processed_map_path"] == loaded_manifest.processed_map_path


def test_norwegian_margin_scenario_stays_inside_fixture_grid() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data/gravity_maps/raw/norwegian_margin/norwegian_margin_fixture.csv"
    map_model = load_regular_csv_gravity_map(
        raw_path,
        name="norwegian_margin_fixture",
        region_name="norwegian_margin",
        source_name="fixture",
    )
    scenario_mapping = load_config_mapping(
        root / "configs/scenarios/norwegian_margin_maritime.json"
    )
    scenario = ScenarioSpec.from_mapping(scenario_mapping)
    truth = build_truth_trajectory_from_scenario(scenario, dt_s=2.0)

    mask = map_model.contains(truth.lat_rad, truth.lon_rad)
    assert bool(mask.all())


def test_load_regular_xyz_gravity_map_small_grid(tmp_path: Path) -> None:
    xyz_path = tmp_path / "small.xyz"
    xyz_path.write_text(
        "\n".join(
            [
                "4.0 63.5 -10.0",
                "4.5 63.5 -8.0",
                "4.0 64.0 -6.0",
                "4.5 64.0 -4.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    map_model = load_regular_xyz_gravity_map(
        xyz_path,
        name="small_xyz",
        region_name="test_region",
        source_name="unit_test",
    )

    assert map_model.shape == (2, 2)
    assert float(map_model.lat_axis_deg[0]) == pytest.approx(63.5)
    assert float(map_model.lat_axis_deg[-1]) == pytest.approx(64.0)
    assert float(map_model.lon_axis_deg[0]) == pytest.approx(4.0)
    assert float(map_model.lon_axis_deg[-1]) == pytest.approx(4.5)
    assert map_model.metadata["source_kind"] == "regular_grid_xyz"
