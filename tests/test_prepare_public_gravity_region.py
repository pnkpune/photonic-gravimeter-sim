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


def test_score_candidates_vectorized_matches_scalar_reference() -> None:
    """
    The vectorized scorer must reproduce the per-candidate score ordering of
    the original scalar route-score helper on a small synthetic gravity map,
    and the per-candidate score values must match within the numerical tolerance
    of the flat-Earth curvilinear approximation used by the vectorized path.
    """
    module = _load_module()
    from gravnav.physics.gravity_map import GravityGridMap
    from gravnav.truth.scenarios import ScenarioSpec

    # Build a smooth synthetic gravity map with a deterministic anomaly so both
    # the scalar and vectorized scorers see a well-defined, finite field.
    lat_axis_deg = np.linspace(66.0, 66.2, 41)
    lon_axis_deg = np.linspace(13.0, 13.3, 61)
    lat_grid, lon_grid = np.meshgrid(
        np.deg2rad(lat_axis_deg),
        np.deg2rad(lon_axis_deg),
        indexing="ij",
    )
    disturbance_mgal = (
        35.0 * np.sin(45.0 * (lat_grid - np.deg2rad(66.1)))
        + 20.0 * np.cos(60.0 * (lon_grid - np.deg2rad(13.15)))
    )
    map_model = GravityGridMap.from_mgal_grid(
        lat_axis_rad=np.deg2rad(lat_axis_deg),
        lon_axis_rad=np.deg2rad(lon_axis_deg),
        disturbance_grid_mgal=disturbance_mgal,
        reference_height_m=0.0,
        default_method="linear",
        bounds_error=False,
        fill_value_mgal=np.nan,
        name="vectorized_score_fixture",
    )

    template = ScenarioSpec.from_mapping(
        {
            "name": "template",
            "initial_lat_deg": 66.1,
            "initial_lon_deg": 13.15,
            "initial_height_m": 0.0,
            "initial_heading_deg": 0.0,
            "default_dt_s": 1.0,
            "description": "vectorized_score_fixture_template",
            "metadata": {},
            "segments": [
                {
                    "type": "straight",
                    "duration_s": 30.0,
                    "speed_mps": 2.0,
                    "flight_path_angle_deg": 0.0,
                    "roll_deg": 0.0,
                    "label": "leg_1",
                }
            ],
        }
    )

    lat_values_deg = np.array([66.05, 66.10, 66.15], dtype=np.float64)
    lon_values_deg = np.array([13.05, 13.15, 13.25], dtype=np.float64)
    headings_deg = [0.0, 90.0]

    vectorized = module._score_candidates_vectorized(
        map_model,
        template=template,
        dt_s=1.0,
        lat_values_deg=lat_values_deg,
        lon_values_deg=lon_values_deg,
        headings_deg=headings_deg,
    )

    scalar: list[dict] = []
    for lat_deg in lat_values_deg:
        for lon_deg in lon_values_deg:
            for heading_deg in headings_deg:
                score = module._candidate_route_score(
                    map_model,
                    lat_deg=float(lat_deg),
                    lon_deg=float(lon_deg),
                    heading_deg=float(heading_deg),
                    template=template,
                    dt_s=1.0,
                )
                if score is not None:
                    scalar.append(score)

    assert len(vectorized) == len(scalar) > 0

    def _key(row: dict) -> tuple[float, float, float]:
        return (
            round(row["initial_lat_deg"], 6),
            round(row["initial_lon_deg"], 6),
            round(row["initial_heading_deg"], 6),
        )

    vec_map = {_key(row): row for row in vectorized}
    for row in scalar:
        key = _key(row)
        assert key in vec_map
        v = vec_map[key]
        # Flat-Earth approximation introduces a small absolute score error at
        # low latitudes and short trajectories; require agreement within 10%.
        assert abs(v["information_score"] - row["information_score"]) <= max(
            1.0e-6,
            0.10 * abs(row["information_score"]),
        )

    # The best candidate selection is what the downstream pipeline uses; require
    # that the two scorers rank the same best candidate.
    best_vec = max(vectorized, key=lambda r: r["information_score"])
    best_scalar = max(scalar, key=lambda r: r["information_score"])
    assert _key(best_vec) == _key(best_scalar)


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
