#!/usr/bin/env python3
"""
Prepare a public regional gravity map and route-search scenario from supported raw formats.

Supported raw formats:
- regular-grid CSV with lat/lon/disturbance columns
- regular-grid XYZ text
- scattered XYZ text that is first binned onto a regular lat/lon grid
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

from gravnav.datasets.gravity_loader import (
    RegionalGravityMapManifest,
    load_regular_csv_gravity_map,
    load_regular_xyz_gravity_map,
)
from gravnav.physics.gravity_map import GravityGridMap
from gravnav.truth.scenarios import ScenarioSpec, build_truth_trajectory_from_scenario
from gravnav.utils.config import load_config_mapping


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
    bounds_deg: tuple[float, float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bounds_deg is None:
        return lat_deg, lon_deg, values_mgal
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
        1.0 * value_std_mgal + 0.35 * grad_mean_mgal_per_km + 8.0 * cross_grad_mean
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


def _build_best_scenario_mapping(
    template: ScenarioSpec,
    *,
    best: dict[str, Any],
    scenario_name: str,
    region_name: str,
    description: str,
) -> dict[str, Any]:
    out = template.to_mapping()
    out["name"] = str(scenario_name)
    out["initial_lat_deg"] = float(best["initial_lat_deg"])
    out.pop("initial_lat_rad", None)
    out["initial_lon_deg"] = float(best["initial_lon_deg"])
    out.pop("initial_lon_rad", None)
    out["initial_heading_deg"] = float(best["initial_heading_deg"])
    out.pop("initial_heading_rad", None)
    metadata = dict(out.get("metadata", {}))
    metadata.update(
        {
            "region_name": str(region_name),
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
    out["description"] = str(description)
    return out


def _manifest_from_map(
    map_model: GravityGridMap,
    *,
    raw_path: Path,
    processed_map_path: Path,
    manifest_path: Path,
    region_name: str,
    source_name: str,
    source_kind: str,
    metadata_extra: dict[str, Any],
    notes: list[str],
) -> RegionalGravityMapManifest:
    return RegionalGravityMapManifest(
        region_name=str(region_name),
        source_name=str(source_name),
        source_kind=str(source_kind),
        raw_data_path=_relative(raw_path),
        processed_map_path=_relative(processed_map_path),
        manifest_path=_relative(manifest_path),
        disturbance_units="mGal",
        reference_height_m=0.0,
        lat_bounds_deg=(float(map_model.lat_axis_deg[0]), float(map_model.lat_axis_deg[-1])),
        lon_bounds_deg=(float(map_model.lon_axis_deg[0]), float(map_model.lon_axis_deg[-1])),
        spacing_deg=(
            float(np.mean(np.diff(map_model.lat_axis_deg))),
            float(np.mean(np.diff(map_model.lon_axis_deg))),
        ),
        shape=tuple(int(x) for x in map_model.shape),
        interpolation=str(map_model.default_method),
        notes=notes,
        metadata={
            "map_name": map_model.name,
            "bounds_error": bool(map_model.bounds_error),
            "fill_value_mps2": (
                None
                if not np.isfinite(float(map_model.fill_value_mps2))
                else float(map_model.fill_value_mps2)
            ),
            **metadata_extra,
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a public regional gravity map and route-search scenario."
    )
    parser.add_argument(
        "--raw-format",
        choices=("regular_csv", "regular_xyz", "scattered_xyz"),
        required=True,
    )
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--region-name", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--processed-map-path", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--scenario-out", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--template-scenario", required=True)
    parser.add_argument("--scenario-name", default="")
    parser.add_argument("--dt-s", type=float, default=2.0)
    parser.add_argument("--search-lat-min", type=float, required=True)
    parser.add_argument("--search-lat-max", type=float, required=True)
    parser.add_argument("--search-lon-min", type=float, required=True)
    parser.add_argument("--search-lon-max", type=float, required=True)
    parser.add_argument("--lat-step-deg", type=float, default=0.20)
    parser.add_argument("--lon-step-deg", type=float, default=0.25)
    parser.add_argument(
        "--headings-deg",
        type=float,
        nargs="+",
        default=(0.0, 30.0, 60.0, 90.0, 120.0, 150.0),
    )
    parser.add_argument("--xyz-column-order", nargs=3, default=("lon_deg", "lat_deg", "disturbance_mgal"))
    parser.add_argument("--skiprows", type=int, default=0)
    parser.add_argument("--crop-lat-min", type=float, default=None)
    parser.add_argument("--crop-lat-max", type=float, default=None)
    parser.add_argument("--crop-lon-min", type=float, default=None)
    parser.add_argument("--crop-lon-max", type=float, default=None)
    parser.add_argument("--coarse-density-scale", type=float, default=0.25)
    parser.add_argument("--final-density-scale", type=float, default=0.55)
    parser.add_argument("--final-margin-deg", type=float, default=0.45)
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    raw_path = Path(args.raw_path).expanduser().resolve()
    processed_map_path = Path(args.processed_map_path).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    scenario_out = Path(args.scenario_out).expanduser().resolve()
    report_out = Path(args.report_out).expanduser().resolve()
    template_scenario_path = Path(args.template_scenario).expanduser().resolve()
    scenario_name = (
        str(args.scenario_name).strip()
        if str(args.scenario_name).strip()
        else scenario_out.stem
    )
    crop_bounds = None
    if None not in (
        args.crop_lat_min,
        args.crop_lat_max,
        args.crop_lon_min,
        args.crop_lon_max,
    ):
        crop_bounds = (
            float(args.crop_lat_min),
            float(args.crop_lat_max),
            float(args.crop_lon_min),
            float(args.crop_lon_max),
        )

    map_model: GravityGridMap
    coarse_metadata: dict[str, Any] = {}
    if args.raw_format == "regular_csv":
        map_model = load_regular_csv_gravity_map(
            raw_path,
            name=f"{args.region_name}_gravity_map",
            region_name=str(args.region_name),
            source_name=str(args.source_name),
            metadata_extra={"raw_loader": "prepare_public_gravity_region.py"},
        )
        source_kind = "regular_grid_csv"
        notes = [
            "Processed from a regular-grid CSV into GravityGridMap NPZ cache.",
        ]
    elif args.raw_format == "regular_xyz":
        map_model = load_regular_xyz_gravity_map(
            raw_path,
            name=f"{args.region_name}_gravity_map",
            region_name=str(args.region_name),
            source_name=str(args.source_name),
            column_order=tuple(str(x) for x in args.xyz_column_order),
            skiprows=int(args.skiprows),
            crop_bounds_deg=crop_bounds,
            metadata_extra={"raw_loader": "prepare_public_gravity_region.py"},
        )
        source_kind = "regular_grid_xyz"
        notes = [
            "Processed from a regular XYZ anomaly grid into GravityGridMap NPZ cache.",
        ]
    else:
        lat_all_deg, lon_all_deg, value_all_mgal = _load_xyz_points(raw_path)
        lat_crop_deg, lon_crop_deg, value_crop_mgal = _filter_points(
            lat_all_deg,
            lon_all_deg,
            value_all_mgal,
            bounds_deg=crop_bounds,
        )
        lat_bounds = (
            float(np.min(lat_crop_deg)),
            float(np.max(lat_crop_deg)),
        )
        lon_bounds = (
            float(np.min(lon_crop_deg)),
            float(np.max(lon_crop_deg)),
        )
        map_model = _bin_scattered_to_map(
            lat_crop_deg,
            lon_crop_deg,
            value_crop_mgal,
            lat_bounds_deg=lat_bounds,
            lon_bounds_deg=lon_bounds,
            density_scale=float(args.coarse_density_scale),
            name=f"{args.region_name}_gravity_map",
            metadata={
                "region_name": str(args.region_name),
                "source_name": str(args.source_name),
                "source_kind": "scattered_xyz_binned",
                "raw_data_path": _relative(raw_path),
                "raw_loader": "prepare_public_gravity_region.py",
            },
        )
        source_kind = "scattered_xyz_binned"
        notes = [
            "Built by binning scattered XYZ anomaly points onto a regular lat/lon grid.",
        ]
        coarse_metadata = {
            "coarse_density_scale": float(args.coarse_density_scale),
            "crop_bounds_deg": None if crop_bounds is None else list(crop_bounds),
        }

    template = _load_template_scenario(template_scenario_path)
    candidates: list[dict[str, Any]] = []
    lat_values = np.arange(args.search_lat_min, args.search_lat_max + 1.0e-12, args.lat_step_deg)
    lon_values = np.arange(args.search_lon_min, args.search_lon_max + 1.0e-12, args.lon_step_deg)
    for lat_deg in lat_values:
        for lon_deg in lon_values:
            for heading_deg in args.headings_deg:
                score = _candidate_route_score(
                    map_model,
                    lat_deg=float(lat_deg),
                    lon_deg=float(lon_deg),
                    heading_deg=float(heading_deg),
                    template=template,
                    dt_s=float(args.dt_s),
                )
                if score is not None:
                    candidates.append(score)
    if len(candidates) == 0:
        raise RuntimeError(
            "No in-bounds candidate routes were found in the requested search region."
        )
    candidates.sort(key=lambda row: float(row["information_score"]), reverse=True)
    best = dict(candidates[0])
    scenario_mapping = _build_best_scenario_mapping(
        template,
        best=best,
        scenario_name=scenario_name,
        region_name=str(args.region_name),
        description=(
            f"Public regional maritime benchmark selected inside {args.region_name} "
            "using the gravity-information score."
        ),
    )

    if args.raw_format == "scattered_xyz":
        truth_best = build_truth_trajectory_from_scenario(
            ScenarioSpec.from_mapping(scenario_mapping),
            dt_s=float(args.dt_s),
        )
        final_bounds = (
            float(np.min(truth_best.lat_rad) * 180.0 / np.pi) - float(args.final_margin_deg),
            float(np.max(truth_best.lat_rad) * 180.0 / np.pi) + float(args.final_margin_deg),
            float(np.min(truth_best.lon_rad) * 180.0 / np.pi) - float(args.final_margin_deg),
            float(np.max(truth_best.lon_rad) * 180.0 / np.pi) + float(args.final_margin_deg),
        )
        lat_final_deg, lon_final_deg, value_final_mgal = _filter_points(
            *_load_xyz_points(raw_path),
            bounds_deg=final_bounds,
        )
        map_model = _bin_scattered_to_map(
            lat_final_deg,
            lon_final_deg,
            value_final_mgal,
            lat_bounds_deg=(final_bounds[0], final_bounds[1]),
            lon_bounds_deg=(final_bounds[2], final_bounds[3]),
            density_scale=float(args.final_density_scale),
            name=f"{args.region_name}_gravity_map",
            metadata={
                "region_name": str(args.region_name),
                "source_name": str(args.source_name),
                "source_kind": "scattered_xyz_binned",
                "raw_data_path": _relative(raw_path),
                "raw_loader": "prepare_public_gravity_region.py",
                "final_bounds_deg": list(final_bounds),
            },
        )
        coarse_metadata["final_density_scale"] = float(args.final_density_scale)
        coarse_metadata["final_bounds_deg"] = list(final_bounds)

    processed_map_path.parent.mkdir(parents=True, exist_ok=True)
    map_model.to_npz(processed_map_path)
    manifest = _manifest_from_map(
        map_model,
        raw_path=raw_path,
        processed_map_path=processed_map_path,
        manifest_path=manifest_path,
        region_name=str(args.region_name),
        source_name=str(args.source_name),
        source_kind=source_kind,
        metadata_extra={
            "raw_loader": "prepare_public_gravity_region.py",
            "search_space": {
                "lat_min_deg": float(args.search_lat_min),
                "lat_max_deg": float(args.search_lat_max),
                "lon_min_deg": float(args.search_lon_min),
                "lon_max_deg": float(args.search_lon_max),
                "lat_step_deg": float(args.lat_step_deg),
                "lon_step_deg": float(args.lon_step_deg),
                "headings_deg": [float(x) for x in args.headings_deg],
            },
            **coarse_metadata,
        },
        notes=notes,
    )
    manifest.write_json(manifest_path)

    scenario_out.parent.mkdir(parents=True, exist_ok=True)
    scenario_out.write_text(json.dumps(scenario_mapping, indent=2) + "\n", encoding="utf-8")
    report_payload = {
        "raw_format": str(args.raw_format),
        "raw_path": _relative(raw_path),
        "processed_map_path": _relative(processed_map_path),
        "manifest_path": _relative(manifest_path),
        "scenario_path": _relative(scenario_out),
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
