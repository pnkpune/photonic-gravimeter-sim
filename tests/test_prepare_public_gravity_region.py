from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_public_gravity_region.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prepare_public_gravity_region_test_module",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bin_scattered_to_map_fills_sparse_cells() -> None:
    module = _load_module()
    lat_deg = np.array([10.0, 10.0, 10.2, 10.2], dtype=np.float64)
    lon_deg = np.array([20.0, 20.2, 20.0, 20.2], dtype=np.float64)
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    grid = module._bin_scattered_to_map(
        lat_deg,
        lon_deg,
        values,
        lat_bounds_deg=(10.0, 10.2),
        lon_bounds_deg=(20.0, 20.2),
        density_scale=0.2,
        name="fixture_map",
        metadata={"region_name": "fixture_region"},
    )

    assert grid.shape[0] >= 20
    assert grid.shape[1] >= 20
    assert np.isfinite(grid.disturbance_grid_mps2).all()


def test_prepare_public_gravity_region_main_writes_outputs_for_regular_xyz(
    tmp_path: Path,
) -> None:
    module = _load_module()
    raw_path = tmp_path / "fixture.xyz"
    raw_path.write_text(
        "\n".join(
            [
                "13.000 66.000 0.0",
                "13.010 66.000 0.5",
                "13.020 66.000 1.0",
                "13.000 66.010 0.7",
                "13.010 66.010 1.2",
                "13.020 66.010 1.7",
                "13.000 66.020 1.4",
                "13.010 66.020 1.9",
                "13.020 66.020 2.4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    template_path = tmp_path / "template.json"
    template_path.write_text(
        json.dumps(
            {
                "name": "template",
                "initial_lat_deg": 66.01,
                "initial_lon_deg": 13.01,
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
                        "label": "leg_1"
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    processed_map = tmp_path / "map.npz"
    manifest = tmp_path / "manifest.json"
    scenario = tmp_path / "scenario.json"
    report = tmp_path / "report.json"

    old_argv = sys.argv[:]
    sys.argv = [
        "prepare_public_gravity_region.py",
        "--raw-format",
        "regular_xyz",
        "--raw-path",
        str(raw_path),
        "--region-name",
        "fixture_region",
        "--source-name",
        "fixture_xyz",
        "--processed-map-path",
        str(processed_map),
        "--manifest-path",
        str(manifest),
        "--scenario-out",
        str(scenario),
        "--report-out",
        str(report),
        "--template-scenario",
        str(template_path),
        "--search-lat-min",
        "66.0",
        "--search-lat-max",
        "66.01",
        "--search-lon-min",
        "13.0",
        "--search-lon-max",
        "13.01",
        "--lat-step-deg",
        "0.01",
        "--lon-step-deg",
        "0.01",
        "--headings-deg",
        "0",
        "90",
    ]
    try:
        rc = module.main()
    finally:
        sys.argv = old_argv

    assert rc == 0
    assert processed_map.exists()
    assert manifest.exists()
    assert scenario.exists()
    assert report.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["region_name"] == "fixture_region"
    assert payload["source_kind"] == "regular_grid_xyz"
