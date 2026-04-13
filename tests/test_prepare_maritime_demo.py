from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gravnav.physics.gravity_map import GravityGridMap
from gravnav.truth.scenarios import (
    CoordinatedTurnSegmentSpec,
    ScenarioSpec,
    StraightSegmentSpec,
)


def _load_prepare_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_norwegian_maritime_demo.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prepare_norwegian_maritime_demo_test_module",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_scenario_uses_degree_keys(tmp_path: Path) -> None:
    module = _load_prepare_module()
    scenario = ScenarioSpec(
        name="helgeland_offshore_maritime",
        initial_lat_rad=float(np.deg2rad(66.12853366319116)),
        initial_lon_rad=float(np.deg2rad(12.910000000000029)),
        initial_height_m=0.0,
        initial_heading_rad=0.0,
        default_dt_s=1.0,
        description="fixture",
        metadata={"region_name": "helgeland_offshore"},
        segments=(
            StraightSegmentSpec(
                duration_s=300.0,
                speed_mps=4.0,
                flight_path_angle_rad=0.0,
                roll_rad=0.0,
                label="demo_leg_1",
            ),
            CoordinatedTurnSegmentSpec(
                duration_s=120.0,
                speed_mps=4.0,
                bank_angle_rad=float(np.deg2rad(8.0)),
                flight_path_angle_rad=0.0,
                roll_in_duration_s=20.0,
                roll_out_duration_s=20.0,
                smooth=True,
                label="demo_turn_1",
            ),
        ),
    )

    out_path = tmp_path / "scenario.json"
    module._write_scenario(out_path, scenario)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "initial_lat_deg" in payload
    assert "initial_lon_deg" in payload
    assert "initial_heading_deg" in payload
    assert "initial_lat_rad" not in payload
    assert "initial_lon_rad" not in payload
    assert "initial_heading_rad" not in payload
    assert payload["segments"][0]["flight_path_angle_deg"] == 0.0
    assert payload["segments"][0]["roll_deg"] == 0.0
    assert payload["segments"][1]["bank_angle_deg"] == 8.0
    assert "bank_angle_rad" not in payload["segments"][1]
    assert "gravity_mps2" not in payload["segments"][1]


def test_search_best_route_returns_in_bounds_candidate() -> None:
    module = _load_prepare_module()
    gravity_map = GravityGridMap.from_mgal_grid(
        lat_axis_rad=np.deg2rad(np.array([66.00, 66.01, 66.02, 66.03], dtype=np.float64)),
        lon_axis_rad=np.deg2rad(np.array([13.00, 13.01, 13.02, 13.03], dtype=np.float64)),
        disturbance_grid_mgal=np.array(
            [
                [0.0, 0.5, 1.0, 1.5],
                [0.7, 1.2, 1.7, 2.2],
                [1.4, 1.9, 2.4, 2.9],
                [2.1, 2.6, 3.1, 3.6],
            ],
            dtype=np.float64,
        ),
        reference_height_m=0.0,
        default_method="linear",
        bounds_error=False,
        fill_value_mgal=np.nan,
        name="fixture_search_map",
    )
    template = ScenarioSpec(
        name="template",
        initial_lat_rad=float(np.deg2rad(66.01)),
        initial_lon_rad=float(np.deg2rad(13.01)),
        initial_height_m=0.0,
        initial_heading_rad=0.0,
        default_dt_s=1.0,
        segments=(
            StraightSegmentSpec(
                duration_s=20.0,
                speed_mps=1.0,
                label="leg_1",
            ),
            CoordinatedTurnSegmentSpec(
                duration_s=10.0,
                speed_mps=1.0,
                bank_angle_rad=float(np.deg2rad(5.0)),
                roll_in_duration_s=2.0,
                roll_out_duration_s=2.0,
                smooth=True,
                label="turn_1",
            ),
            StraightSegmentSpec(
                duration_s=20.0,
                speed_mps=1.0,
                label="leg_2",
            ),
        ),
    )
    gravity_manifest = SimpleNamespace(
        lat_bounds_deg=(66.00, 66.03),
        lon_bounds_deg=(13.00, 13.03),
    )

    scenario, best = module._search_best_route(
        gravity_map=gravity_map,
        gravity_manifest=gravity_manifest,
        template=template,
        dt_s=1.0,
        search_lat_min=66.005,
        search_lat_max=66.015,
        search_lon_min=13.005,
        search_lon_max=13.015,
        search_lat_step_deg=0.005,
        search_lon_step_deg=0.005,
        headings_deg=[0.0, 90.0],
        scenario_name="helgeland_offshore_maritime",
        region_name="helgeland_offshore",
    )

    assert scenario.name == "helgeland_offshore_maritime"
    assert scenario.metadata["region_name"] == "helgeland_offshore"
    assert "route_selection" in scenario.metadata
    assert best["information_score"] > 0.0
    assert 66.005 <= best["initial_lat_deg"] <= 66.015
    assert 13.005 <= best["initial_lon_deg"] <= 13.015
