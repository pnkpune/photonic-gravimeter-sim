#!/usr/bin/env python3
"""
Prepare a public-product Norwegian benchmark from the NAG-TEC Bouguer anomaly grid.

The public `bouguer_anomaly_geo.xyz` file is not a rectilinear latitude/longitude
grid. It is a geographic export of a projected grid, so this script:

1. loads scattered geographic XYZ anomaly points
2. crops a Norwegian-margin search region
3. bins those points onto a regular lat/lon grid
4. scores candidate routes using the existing maritime-style motion profile
5. writes one processed map cache, one manifest, and one generated scenario config
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from gravnav.datasets.gravity_loader import RegionalGravityMapManifest
from gravnav.physics.gravity_map import GravityGridMap
from gravnav.truth.scenarios import ScenarioSpec, build_truth_trajectory_from_scenario
from gravnav.utils.config import load_config_mapping


DEFAULT_RAW_PATH = PROJECT_ROOT / "data/gravity_maps/raw/public_nagtec/bouguer_anomaly_geo.xyz"
DEFAULT_PROCESSED_MAP = PROJECT_ROOT / "data/gravity_maps/processed/norwegian_margin_public_bouguer_map.npz"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/gravity_maps/processed/norwegian_margin_public_bouguer_manifest.json"
DEFAULT_SCENARIO_OUT = PROJECT_ROOT / "configs/scenarios/norwegian_margin_public_maritime.json"
DEFAULT_REPORT_OUT = PROJECT_ROOT / "data/outputs/reports/norwegian_margin_public_route_search.json"
DEFAULT_TEMPLATE_SCENARIO = PROJECT_ROOT / "configs/scenarios/norwegian_margin_maritime.json"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _load_template_scenario(path: Path) -> ScenarioSpec:
    return ScenarioSpec.from_mapping(load_config_mapping(path))


def _load_xyz_points(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_values: list[float] = []
    lat_values: list[float] = []
    disturbance_values: list[float] = []

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                lon_deg = float(parts[0])
                lat_deg = float(parts[1])
                disturbance_mgal = float(parts[2])
            except ValueError:
                continue
            lon_values.append(lon_deg)
            lat_values.append(lat_deg)
            disturbance_values.append(disturbance_mgal)

    if len(lat_values) == 0:
        raise ValueError(f"XYZ file {path} did not contain any valid numeric rows.")

    return (
        np.asarray(lat_values, dtype=np.float64),
        np.asarray(lon_values, dtype=np.float64),
        np.asarray(disturbance_values, dtype=np.float64),
    )


def _filter_points(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    values_mgal: np.ndarray,
    *,
    bounds_deg: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_min, lat_max, lon_min, lon_max = bounds_deg
    keep = (
        (lat_deg >= float(lat_min))
        & (lat_deg <= float(lat_max))
        & (lon_deg >= float(lon_min))
        & (lon_deg <= float(lon_max))
    )
    if not np.any(keep):
        raise ValueError("No public anomaly points fell inside the requested bounds.")
    return lat_deg[keep], lon_deg[keep], values_mgal[keep]


def _choose_grid_shape(
    *,
    lat_bounds_deg: tuple[float, float],
    lon_bounds_deg: tuple[float, float],
    n_points: int,
    density_scale: float,
) -> tuple[int, int]:
    lat_span_deg = max(1.0e-6, float(lat_bounds_deg[1] - lat_bounds_deg[0]))
    lon_span_deg = max(1.0e-6, float(lon_bounds_deg[1] - lon_bounds_deg[0]))
    lat_mid_deg = 0.5 * (float(lat_bounds_deg[0]) + float(lat_bounds_deg[1]))
    lat_span_m = lat_span_deg * 111_320.0
    lon_span_m = lon_span_deg * 111_320.0 * max(np.cos(np.deg2rad(lat_mid_deg)), 0.2)
    aspect = max(lon_span_m / lat_span_m, 1.0e-6)
    target_cells = int(np.clip(density_scale * float(n_points), 800.0, 20_000.0))
    n_lat = int(np.clip(np.sqrt(target_cells / aspect), 20.0, 200.0))
    n_lon = int(np.clip(target_cells / max(n_lat, 1), 20.0, 300.0))
    return n_lat, n_lon


def _fill_nan_nearest(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=np.float64).copy()
    known = np.argwhere(np.isfinite(out))
    missing = np.argwhere(~np.isfinite(out))
    if len(missing) == 0 or len(known) == 0:
        return out
    known_values = out[known[:, 0], known[:, 1]]
    for i, j in missing:
        diff = known - np.array([i, j], dtype=np.int64)
        dist2 = np.sum(diff * diff, axis=1)
        idx = int(np.argmin(dist2))
        out[i, j] = float(known_values[idx])
    return out


def _bin_scattered_to_map(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    values_mgal: np.ndarray,
    *,
    lat_bounds_deg: tuple[float, float],
    lon_bounds_deg: tuple[float, float],
    density_scale: float,
    name: str,
    metadata: dict[str, Any],
) -> GravityGridMap:
    n_lat, n_lon = _choose_grid_shape(
        lat_bounds_deg=lat_bounds_deg,
        lon_bounds_deg=lon_bounds_deg,
        n_points=int(lat_deg.size),
        density_scale=density_scale,
    )
    lat_axis_deg = np.linspace(lat_bounds_deg[0], lat_bounds_deg[1], n_lat)
    lon_axis_deg = np.linspace(lon_bounds_deg[0], lon_bounds_deg[1], n_lon)
    lat_step = float(lat_axis_deg[1] - lat_axis_deg[0])
    lon_step = float(lon_axis_deg[1] - lon_axis_deg[0])
    i = np.rint((lat_deg - float(lat_axis_deg[0])) / lat_step).astype(np.int64)
    j = np.rint((lon_deg - float(lon_axis_deg[0])) / lon_step).astype(np.int64)
    i = np.clip(i, 0, n_lat - 1)
    j = np.clip(j, 0, n_lon - 1)
    sums = np.zeros((n_lat, n_lon), dtype=np.float64)
    counts = np.zeros((n_lat, n_lon), dtype=np.int64)
    np.add.at(sums, (i, j), values_mgal)
    np.add.at(counts, (i, j), 1)
    grid_mgal = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    valid = counts > 0
    grid_mgal[valid] = sums[valid] / counts[valid]
    grid_mgal = _fill_nan_nearest(grid_mgal)
    return GravityGridMap.from_mgal_grid(
        lat_axis_rad=np.deg2rad(lat_axis_deg),
        lon_axis_rad=np.deg2rad(lon_axis_deg),
        disturbance_grid_mgal=grid_mgal,
        reference_height_m=0.0,
        default_method="linear",
        bounds_error=False,
        fill_value_mgal=np.nan,
        name=name,
        metadata=metadata,
    )


def _candidate_route_score(
    map_model: GravityGridMap,
    *,
    lat_deg: float,
    lon_deg: float,
    heading_deg: float,
    template: ScenarioSpec,
    dt_s: float,
) -> dict[str, Any] | None:
    scenario = ScenarioSpec(
        name="candidate",
        initial_lat_rad=float(np.deg2rad(lat_deg)),
        initial_lon_rad=float(np.deg2rad(lon_deg)),
        initial_height_m=float(template.initial_height_m),
        initial_heading_rad=float(np.deg2rad(heading_deg)),
        segments=template.segments,
        default_dt_s=float(template.default_dt_s),
        description=template.description,
        metadata=dict(template.metadata),
    )
    truth = build_truth_trajectory_from_scenario(scenario, dt_s=dt_s)
    inside = map_model.contains(truth.lat_rad, truth.lon_rad)
    if not bool(np.asarray(inside, dtype=bool).all()):
        return None

    values = np.asarray(
        map_model.sample_disturbance(truth.lat_rad, truth.lon_rad, truth.height_m),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        return None

    grad_ned = np.asarray(
        map_model.disturbance_gradient_ned(truth.lat_rad, truth.lon_rad, truth.height_m),
        dtype=np.float64,
    )
    horiz_grad = np.linalg.norm(grad_ned[:, :2], axis=1)
    north_vel = np.asarray(truth.v_ned_mps, dtype=np.float64)[:, 0]
    east_vel = np.asarray(truth.v_ned_mps, dtype=np.float64)[:, 1]
    speed = np.sqrt(north_vel**2 + east_vel**2)
    route_dir = np.stack(
        [
            np.divide(north_vel, speed, out=np.zeros_like(north_vel), where=speed > 0.0),
            np.divide(east_vel, speed, out=np.zeros_like(east_vel), where=speed > 0.0),
        ],
        axis=-1,
    )
    grad_dir = np.stack(
        [
            np.divide(grad_ned[:, 0], horiz_grad, out=np.zeros_like(horiz_grad), where=horiz_grad > 0.0),
            np.divide(grad_ned[:, 1], horiz_grad, out=np.zeros_like(horiz_grad), where=horiz_grad > 0.0),
        ],
        axis=-1,
    )
    cross_grad = np.abs(
        route_dir[:, 0] * grad_dir[:, 1] - route_dir[:, 1] * grad_dir[:, 0]
    )
    value_std_mgal = float(np.std(values) * 1.0e5)
    grad_mean_mgal_per_km = float(np.mean(horiz_grad) * 1.0e8)
    cross_grad_mean = float(np.mean(cross_grad))
    information_score = (
        1.0 * value_std_mgal
        + 0.35 * grad_mean_mgal_per_km
        + 8.0 * cross_grad_mean
    )
    return {
        "initial_lat_deg": float(lat_deg),
        "initial_lon_deg": float(lon_deg),
        "initial_heading_deg": float(heading_deg),
        "value_std_mgal": value_std_mgal,
        "mean_horizontal_gradient_mgal_per_km": grad_mean_mgal_per_km,
        "cross_gradient_alignment": cross_grad_mean,
        "information_score": float(information_score),
    }


def _build_best_scenario_mapping(template: ScenarioSpec, best: dict[str, Any]) -> dict[str, Any]:
    out = template.to_mapping()
    out["name"] = "norwegian_margin_public_maritime"
    out["initial_lat_deg"] = float(best["initial_lat_deg"])
    out.pop("initial_lat_rad", None)
    out["initial_lon_deg"] = float(best["initial_lon_deg"])
    out.pop("initial_lon_rad", None)
    out["initial_heading_deg"] = float(best["initial_heading_deg"])
    out.pop("initial_heading_rad", None)
    metadata = dict(out.get("metadata", {}))
    metadata.update(
        {
            "region_name": "norwegian_margin_public_bouguer",
            "route_selection": {
                "information_score": float(best["information_score"]),
                "value_std_mgal": float(best["value_std_mgal"]),
                "mean_horizontal_gradient_mgal_per_km": float(
                    best["mean_horizontal_gradient_mgal_per_km"]
                ),
                "cross_gradient_alignment": float(best["cross_gradient_alignment"]),
            },
        }
    )
    out["metadata"] = metadata
    out["description"] = (
        "Public-product Norwegian-margin maritime benchmark selected from the "
        "NAG-TEC Bouguer anomaly grid using a route-information score."
    )
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a public-product Norwegian benchmark map and scenario.")
    parser.add_argument("--raw-xyz-path", default=str(DEFAULT_RAW_PATH))
    parser.add_argument("--processed-map-path", default=str(DEFAULT_PROCESSED_MAP))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--scenario-out", default=str(DEFAULT_SCENARIO_OUT))
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT_OUT))
    parser.add_argument("--template-scenario", default=str(DEFAULT_TEMPLATE_SCENARIO))
    parser.add_argument("--dt-s", type=float, default=2.0)
    parser.add_argument("--search-lat-min", type=float, default=62.0)
    parser.add_argument("--search-lat-max", type=float, default=67.0)
    parser.add_argument("--search-lon-min", type=float, default=1.0)
    parser.add_argument("--search-lon-max", type=float, default=19.0)
    parser.add_argument("--coarse-lat-min", type=float, default=62.0)
    parser.add_argument("--coarse-lat-max", type=float, default=67.5)
    parser.add_argument("--coarse-lon-min", type=float, default=0.0)
    parser.add_argument("--coarse-lon-max", type=float, default=20.0)
    parser.add_argument("--lat-step-deg", type=float, default=0.20)
    parser.add_argument("--lon-step-deg", type=float, default=0.25)
    parser.add_argument("--headings-deg", type=float, nargs="+", default=(0.0, 30.0, 60.0, 90.0, 120.0, 150.0))
    parser.add_argument("--coarse-density-scale", type=float, default=0.25)
    parser.add_argument("--final-density-scale", type=float, default=0.55)
    parser.add_argument("--final-margin-deg", type=float, default=0.45)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    raw_xyz_path = Path(args.raw_xyz_path).expanduser().resolve()
    processed_map_path = Path(args.processed_map_path).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    scenario_out = Path(args.scenario_out).expanduser().resolve()
    report_out = Path(args.report_out).expanduser().resolve()
    template_scenario_path = Path(args.template_scenario).expanduser().resolve()

    lat_all_deg, lon_all_deg, value_all_mgal = _load_xyz_points(raw_xyz_path)
    coarse_bounds = (
        float(args.coarse_lat_min),
        float(args.coarse_lat_max),
        float(args.coarse_lon_min),
        float(args.coarse_lon_max),
    )
    lat_crop_deg, lon_crop_deg, value_crop_mgal = _filter_points(
        lat_all_deg,
        lon_all_deg,
        value_all_mgal,
        bounds_deg=coarse_bounds,
    )

    coarse_map = _bin_scattered_to_map(
        lat_crop_deg,
        lon_crop_deg,
        value_crop_mgal,
        lat_bounds_deg=(coarse_bounds[0], coarse_bounds[1]),
        lon_bounds_deg=(coarse_bounds[2], coarse_bounds[3]),
        density_scale=float(args.coarse_density_scale),
        name="norwegian_margin_public_bouguer_coarse",
        metadata={
            "region_name": "norwegian_margin_public_bouguer",
            "source_name": "nagtec_bouguer_anomaly_geo_xyz",
            "source_kind": "scattered_xyz_binned",
            "raw_data_path": _relative(raw_xyz_path),
            "coarse_bounds_deg": list(coarse_bounds),
        },
    )

    template = _load_template_scenario(template_scenario_path)
    lat_values = np.arange(args.search_lat_min, args.search_lat_max + 1e-12, args.lat_step_deg)
    lon_values = np.arange(args.search_lon_min, args.search_lon_max + 1e-12, args.lon_step_deg)

    candidates: list[dict[str, Any]] = []
    for lat_deg in lat_values:
        for lon_deg in lon_values:
            for heading_deg in args.headings_deg:
                score = _candidate_route_score(
                    coarse_map,
                    lat_deg=float(lat_deg),
                    lon_deg=float(lon_deg),
                    heading_deg=float(heading_deg),
                    template=template,
                    dt_s=float(args.dt_s),
                )
                if score is not None:
                    candidates.append(score)

    if len(candidates) == 0:
        raise RuntimeError("No in-bounds candidate routes were found in the requested search region.")

    candidates.sort(key=lambda row: float(row["information_score"]), reverse=True)
    best = dict(candidates[0])
    scenario_mapping = _build_best_scenario_mapping(template, best)
    truth_best = build_truth_trajectory_from_scenario(
        ScenarioSpec.from_mapping(scenario_mapping),
        dt_s=float(args.dt_s),
    )

    lat_margin = float(args.final_margin_deg)
    lon_margin = float(args.final_margin_deg)
    final_bounds = (
        float(np.min(truth_best.lat_rad) * 180.0 / np.pi) - lat_margin,
        float(np.max(truth_best.lat_rad) * 180.0 / np.pi) + lat_margin,
        float(np.min(truth_best.lon_rad) * 180.0 / np.pi) - lon_margin,
        float(np.max(truth_best.lon_rad) * 180.0 / np.pi) + lon_margin,
    )
    lat_final_deg, lon_final_deg, value_final_mgal = _filter_points(
        lat_all_deg,
        lon_all_deg,
        value_all_mgal,
        bounds_deg=final_bounds,
    )
    final_map = _bin_scattered_to_map(
        lat_final_deg,
        lon_final_deg,
        value_final_mgal,
        lat_bounds_deg=(final_bounds[0], final_bounds[1]),
        lon_bounds_deg=(final_bounds[2], final_bounds[3]),
        density_scale=float(args.final_density_scale),
        name="norwegian_margin_public_bouguer_map",
        metadata={
            "region_name": "norwegian_margin_public_bouguer",
            "source_name": "nagtec_bouguer_anomaly_geo_xyz",
            "source_kind": "scattered_xyz_binned",
            "raw_data_path": _relative(raw_xyz_path),
            "crop_bounds_deg": list(final_bounds),
            "dataset_name": "Gravity Anomaly Grids - NAG-TEC Atlas",
            "product_name": "bouguer_anomaly_geo.xyz",
            "doi": "10.22008/FK2/AQ38FS",
        },
    )
    processed_map_path.parent.mkdir(parents=True, exist_ok=True)
    final_map.to_npz(processed_map_path)

    manifest = RegionalGravityMapManifest(
        region_name="norwegian_margin_public_bouguer",
        source_name="nagtec_bouguer_anomaly_geo_xyz",
        source_kind="scattered_xyz_binned",
        raw_data_path=_relative(raw_xyz_path),
        processed_map_path=_relative(processed_map_path),
        manifest_path=_relative(manifest_path),
        disturbance_units="mGal",
        reference_height_m=0.0,
        lat_bounds_deg=(float(final_map.lat_axis_deg[0]), float(final_map.lat_axis_deg[-1])),
        lon_bounds_deg=(float(final_map.lon_axis_deg[0]), float(final_map.lon_axis_deg[-1])),
        spacing_deg=(
            float(np.mean(np.diff(final_map.lat_axis_deg))),
            float(np.mean(np.diff(final_map.lon_axis_deg))),
        ),
        shape=tuple(int(x) for x in final_map.shape),
        interpolation=str(final_map.default_method),
        notes=[
            "Built from the public NAG-TEC Bouguer anomaly XYZ product by binning scattered geographic points onto a regular lat/lon grid.",
            "This map is a regional scalar proxy, not a same-point gravity-disturbance product.",
        ],
        metadata={
            "map_name": final_map.name,
            "bounds_error": bool(final_map.bounds_error),
            "fill_value_mps2": None,
            "raw_loader": "prepare_public_norwegian_benchmark.py",
            "coarse_bounds_deg": list(coarse_bounds),
            "final_bounds_deg": list(final_bounds),
            "coarse_density_scale": float(args.coarse_density_scale),
            "final_density_scale": float(args.final_density_scale),
        },
    )
    manifest.write_json(manifest_path)

    scenario_out.parent.mkdir(parents=True, exist_ok=True)
    scenario_out.write_text(json.dumps(scenario_mapping, indent=2) + "\n", encoding="utf-8")

    report_payload = {
        "raw_xyz_path": _relative(raw_xyz_path),
        "processed_map_path": _relative(processed_map_path),
        "manifest_path": _relative(manifest_path),
        "scenario_path": _relative(scenario_out),
        "search_space": {
            "lat_min_deg": float(args.search_lat_min),
            "lat_max_deg": float(args.search_lat_max),
            "lon_min_deg": float(args.search_lon_min),
            "lon_max_deg": float(args.search_lon_max),
            "lat_step_deg": float(args.lat_step_deg),
            "lon_step_deg": float(args.lon_step_deg),
            "headings_deg": [float(x) for x in args.headings_deg],
        },
        "coarse_bounds_deg": list(coarse_bounds),
        "final_bounds_deg": list(final_bounds),
        "manifest": manifest.to_mapping(),
        "best_candidate": best,
        "top_candidates": candidates[:10],
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")

    print(f"Processed map: {_relative(processed_map_path)}")
    print(f"Manifest: {_relative(manifest_path)}")
    print(f"Scenario: {_relative(scenario_out)}")
    print(f"Search report: {_relative(report_out)}")
    print(
        "Best candidate: "
        f"lat={best['initial_lat_deg']:.3f} deg, "
        f"lon={best['initial_lon_deg']:.3f} deg, "
        f"heading={best['initial_heading_deg']:.1f} deg, "
        f"score={best['information_score']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
