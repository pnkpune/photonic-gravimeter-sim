#!/usr/bin/env python3
"""
Prepare a matched maritime demo pack from one regional gravity window.

This script supports two modes:

1. First-region compatibility mode
   Rebuild the promoted Norwegian-margin demo bathymetry/demo-pack around an
   existing tracked scenario.

2. Offshore search mode
   Use an existing processed gravity manifest, search over start pose and
   heading using the existing gravity-information score, write a scenario
   config, sample GEBCO bathymetry around that route, and emit a matched
   demo-pack manifest.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from gravnav.datasets.bathymetry_loader import (
    PROCESSED_GRID_NAME,
    PROCESSED_MANIFEST_NAME,
    RAW_FIXTURE_NAME,
    BathymetryGrid,
    RegionalDemoPackManifest,
    process_regular_csv_bathymetry_grid,
)
from gravnav.datasets.gravity_loader import (
    load_regional_manifest,
    load_regional_map_from_manifest,
)
from gravnav.simulation.runner import resolve_truth_trajectory
from gravnav.truth.scenarios import (
    ConstantRateTurnSegmentSpec,
    CoordinatedTurnSegmentSpec,
    ScenarioSpec,
    StraightSegmentSpec,
    build_truth_trajectory_from_scenario,
)
from gravnav.utils.config import load_config_mapping

DEFAULT_SCENARIO = PROJECT_ROOT / "configs/scenarios/norwegian_margin_maritime.json"
DEFAULT_TEMPLATE_SCENARIO = PROJECT_ROOT / "configs/scenarios/norwegian_margin_maritime.json"
DEFAULT_SEQUENCE_PROFILE = PROJECT_ROOT / "configs/sequence_profiles/norwegian_margin_maritime_demo.json"
DEFAULT_GRAVITY_MANIFEST = PROJECT_ROOT / "data/gravity_maps/processed/norwegian_margin_gravity_map_manifest.json"
DEFAULT_RAW_BATHY = PROJECT_ROOT / "data/bathymetry/raw/norwegian_margin_public" / RAW_FIXTURE_NAME
DEFAULT_PROCESSED_BATHY = PROJECT_ROOT / "data/bathymetry/processed" / PROCESSED_GRID_NAME
DEFAULT_BATHY_MANIFEST = PROJECT_ROOT / "data/bathymetry/processed" / PROCESSED_MANIFEST_NAME
DEFAULT_DEMO_PACK_MANIFEST = PROJECT_ROOT / "data/bathymetry/processed" / "norwegian_margin_maritime_demo_pack.json"
DEFAULT_REGION_NAME = "norwegian_margin_maritime_demo"


def _query_gebco_elevation_m(lat_deg: float, lon_deg: float) -> float:
    bbox = (
        f"{lat_deg - 5.0e-4:.6f},{lon_deg - 5.0e-4:.6f},"
        f"{lat_deg + 5.0e-4:.6f},{lon_deg + 5.0e-4:.6f}"
    )
    url = "https://wms.gebco.net/mapserv?" + urllib.parse.urlencode(
        {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetFeatureInfo",
            "LAYERS": "GEBCO_LATEST_2",
            "QUERY_LAYERS": "GEBCO_LATEST_2",
            "CRS": "EPSG:4326",
            "BBOX": bbox,
            "WIDTH": 2,
            "HEIGHT": 2,
            "I": 1,
            "J": 1,
            "INFO_FORMAT": "text/plain",
        }
    )
    with urllib.request.urlopen(url, timeout=20.0) as resp:
        text = resp.read().decode("utf-8")
    for line in text.splitlines():
        if "value_list" in line:
            value = line.split("=", 1)[1].strip().strip("'")
            return float(value)
    raise RuntimeError(
        f"GEBCO WMS response did not contain value_list for lat={lat_deg}, lon={lon_deg}."
    )


def _sample_regular_bathymetry_csv(
    lat_axis_deg: np.ndarray,
    lon_axis_deg: np.ndarray,
    *,
    out_path: Path,
    max_workers: int,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    points = [(float(lat), float(lon)) for lat in lat_axis_deg for lon in lon_axis_deg]

    def one(point: tuple[float, float]) -> tuple[float, float, float]:
        lat_deg, lon_deg = point
        elev = _query_gebco_elevation_m(lat_deg, lon_deg)
        return lat_deg, lon_deg, elev

    rows: list[tuple[float, float, float]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        for row in ex.map(one, points):
            rows.append(row)

    rows.sort(key=lambda x: (x[0], x[1]))
    lines = ["lat_deg,lon_deg,elevation_m"]
    for lat_deg, lon_deg, elev_m in rows:
        lines.append(f"{lat_deg:.6f},{lon_deg:.6f},{elev_m:.3f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _load_scenario(path: Path) -> ScenarioSpec:
    return ScenarioSpec.from_mapping(load_config_mapping(path))


def _truth_bounds_deg(
    scenario: ScenarioSpec,
    *,
    dt_s: float,
    margin_deg: float,
    lat_step_deg: float,
    lon_step_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    truth = resolve_truth_trajectory(scenario, dt_s=dt_s)
    lat_min = float(np.rad2deg(np.min(truth.lat_rad))) - margin_deg
    lat_max = float(np.rad2deg(np.max(truth.lat_rad))) + margin_deg
    lon_min = float(np.rad2deg(np.min(truth.lon_rad))) - margin_deg
    lon_max = float(np.rad2deg(np.max(truth.lon_rad))) + margin_deg
    return (
        np.arange(lat_min, lat_max + 1.0e-12, lat_step_deg, dtype=np.float64),
        np.arange(lon_min, lon_max + 1.0e-12, lon_step_deg, dtype=np.float64),
    )


def _candidate_route_score(
    map_model,
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


def _build_best_scenario_mapping(
    template: ScenarioSpec,
    *,
    best: dict[str, Any],
    scenario_name: str,
    region_name: str,
    description: str,
) -> dict[str, Any]:
    out = template.to_mapping()
    out["name"] = scenario_name
    out["initial_lat_deg"] = float(best["initial_lat_deg"])
    out.pop("initial_lat_rad", None)
    out["initial_lon_deg"] = float(best["initial_lon_deg"])
    out.pop("initial_lon_rad", None)
    out["initial_heading_deg"] = float(best["initial_heading_deg"])
    out.pop("initial_heading_rad", None)
    metadata = dict(out.get("metadata", {}))
    metadata.update(
        {
            "region_name": region_name,
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
    out["description"] = description
    return out


def _segment_to_degree_mapping(
    segment: StraightSegmentSpec | ConstantRateTurnSegmentSpec | CoordinatedTurnSegmentSpec,
) -> dict[str, Any]:
    if isinstance(segment, StraightSegmentSpec):
        return {
            "type": "straight",
            "duration_s": float(segment.duration_s),
            "speed_mps": float(segment.speed_mps),
            "flight_path_angle_deg": float(np.rad2deg(segment.flight_path_angle_rad)),
            "roll_deg": float(np.rad2deg(segment.roll_rad)),
            "label": segment.label,
        }
    if isinstance(segment, ConstantRateTurnSegmentSpec):
        out: dict[str, Any] = {
            "type": "constant_rate_turn",
            "duration_s": float(segment.duration_s),
            "speed_mps": float(segment.speed_mps),
            "heading_rate_degps": float(np.rad2deg(segment.heading_rate_radps)),
            "flight_path_angle_deg": float(np.rad2deg(segment.flight_path_angle_rad)),
            "label": segment.label,
        }
        if segment.roll_rad is not None:
            out["roll_deg"] = float(np.rad2deg(segment.roll_rad))
        return out
    if isinstance(segment, CoordinatedTurnSegmentSpec):
        out = {
            "type": "coordinated_turn",
            "duration_s": float(segment.duration_s),
            "speed_mps": float(segment.speed_mps),
            "bank_angle_deg": float(np.rad2deg(segment.bank_angle_rad)),
            "flight_path_angle_deg": float(np.rad2deg(segment.flight_path_angle_rad)),
            "roll_in_duration_s": float(segment.roll_in_duration_s),
            "roll_out_duration_s": float(segment.roll_out_duration_s),
            "smooth": bool(segment.smooth),
            "label": segment.label,
        }
        if segment.gravity_mps2 is not None:
            out["gravity_mps2"] = float(segment.gravity_mps2)
        return out
    raise TypeError(f"Unsupported segment type: {type(segment)!r}")


def _scenario_to_degree_mapping(scenario: ScenarioSpec) -> dict[str, Any]:
    return {
        "name": scenario.name,
        "initial_lat_deg": float(np.rad2deg(scenario.initial_lat_rad)),
        "initial_lon_deg": float(np.rad2deg(scenario.initial_lon_rad)),
        "initial_height_m": float(scenario.initial_height_m),
        "initial_heading_deg": float(np.rad2deg(scenario.initial_heading_rad)),
        "default_dt_s": float(scenario.default_dt_s),
        "description": scenario.description,
        "metadata": dict(scenario.metadata),
        "segments": [_segment_to_degree_mapping(seg) for seg in scenario.segments],
    }


def _search_best_route(
    *,
    gravity_map,
    gravity_manifest,
    template: ScenarioSpec,
    dt_s: float,
    search_lat_min: float | None,
    search_lat_max: float | None,
    search_lon_min: float | None,
    search_lon_max: float | None,
    search_lat_step_deg: float,
    search_lon_step_deg: float,
    headings_deg: list[float],
    scenario_name: str,
    region_name: str,
) -> tuple[ScenarioSpec, dict[str, Any]]:
    lat_min = (
        float(search_lat_min)
        if search_lat_min is not None
        else float(gravity_manifest.lat_bounds_deg[0])
    )
    lat_max = (
        float(search_lat_max)
        if search_lat_max is not None
        else float(gravity_manifest.lat_bounds_deg[1])
    )
    lon_min = (
        float(search_lon_min)
        if search_lon_min is not None
        else float(gravity_manifest.lon_bounds_deg[0])
    )
    lon_max = (
        float(search_lon_max)
        if search_lon_max is not None
        else float(gravity_manifest.lon_bounds_deg[1])
    )

    lat_values = np.arange(lat_min, lat_max + 1.0e-12, search_lat_step_deg)
    lon_values = np.arange(lon_min, lon_max + 1.0e-12, search_lon_step_deg)
    candidates: list[dict[str, Any]] = []
    for lat_deg in lat_values:
        for lon_deg in lon_values:
            for heading_deg in headings_deg:
                score = _candidate_route_score(
                    gravity_map,
                    lat_deg=float(lat_deg),
                    lon_deg=float(lon_deg),
                    heading_deg=float(heading_deg),
                    template=template,
                    dt_s=dt_s,
                )
                if score is not None:
                    candidates.append(score)

    if not candidates:
        raise RuntimeError(
            f"No in-bounds candidate routes were found inside {region_name!r}."
        )

    candidates.sort(key=lambda row: float(row["information_score"]), reverse=True)
    best = dict(candidates[0])
    mapping = _build_best_scenario_mapping(
        template,
        best=best,
        scenario_name=scenario_name,
        region_name=region_name,
        description=(
            f"Public offshore maritime demo route selected inside {region_name} "
            "using the gravity-information score."
        ),
    )
    return ScenarioSpec.from_mapping(mapping), best


def _write_scenario(path: Path, scenario: ScenarioSpec) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_scenario_to_degree_mapping(scenario), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a maritime photonic demo pack.")
    parser.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    parser.add_argument("--template-scenario", default=str(DEFAULT_TEMPLATE_SCENARIO))
    parser.add_argument("--sequence-profile", default=str(DEFAULT_SEQUENCE_PROFILE))
    parser.add_argument("--gravity-manifest", default=str(DEFAULT_GRAVITY_MANIFEST))
    parser.add_argument("--region-name", default=DEFAULT_REGION_NAME)
    parser.add_argument("--scenario-name", default="")
    parser.add_argument("--raw-bathymetry-csv", default=str(DEFAULT_RAW_BATHY))
    parser.add_argument("--processed-bathymetry", default=str(DEFAULT_PROCESSED_BATHY))
    parser.add_argument("--bathymetry-manifest", default=str(DEFAULT_BATHY_MANIFEST))
    parser.add_argument("--demo-pack-manifest", default=str(DEFAULT_DEMO_PACK_MANIFEST))
    parser.add_argument("--dt-s", type=float, default=2.0)
    parser.add_argument("--route-search-dt-s", type=float, default=5.0)
    parser.add_argument("--margin-deg", type=float, default=0.18)
    parser.add_argument("--lat-step-deg", type=float, default=0.02)
    parser.add_argument("--lon-step-deg", type=float, default=0.03)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--force-refresh-bathymetry", action="store_true")
    parser.add_argument("--search-route", action="store_true")
    parser.add_argument("--search-lat-min", type=float)
    parser.add_argument("--search-lat-max", type=float)
    parser.add_argument("--search-lon-min", type=float)
    parser.add_argument("--search-lon-max", type=float)
    parser.add_argument("--search-lat-step-deg", type=float, default=0.08)
    parser.add_argument("--search-lon-step-deg", type=float, default=0.08)
    parser.add_argument(
        "--headings-deg",
        type=float,
        nargs="+",
        default=(0.0, 30.0, 60.0, 90.0, 120.0, 150.0),
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    scenario_path = Path(args.scenario).expanduser().resolve()
    template_scenario_path = Path(args.template_scenario).expanduser().resolve()
    sequence_profile_path = Path(args.sequence_profile).expanduser().resolve()
    gravity_manifest_path = Path(args.gravity_manifest).expanduser().resolve()
    raw_bathy_path = Path(args.raw_bathymetry_csv).expanduser().resolve()
    processed_bathy_path = Path(args.processed_bathymetry).expanduser().resolve()
    bathy_manifest_path = Path(args.bathymetry_manifest).expanduser().resolve()
    demo_pack_manifest_path = Path(args.demo_pack_manifest).expanduser().resolve()

    gravity_map, gravity_manifest, _, _ = load_regional_map_from_manifest(
        gravity_manifest_path
    )

    if args.search_route:
        template = _load_scenario(template_scenario_path)
        scenario_name = (
            str(args.scenario_name).strip()
            if str(args.scenario_name).strip()
            else scenario_path.stem
        )
        scenario, best = _search_best_route(
            gravity_map=gravity_map,
            gravity_manifest=gravity_manifest,
            template=template,
            dt_s=float(args.route_search_dt_s),
            search_lat_min=args.search_lat_min,
            search_lat_max=args.search_lat_max,
            search_lon_min=args.search_lon_min,
            search_lon_max=args.search_lon_max,
            search_lat_step_deg=float(args.search_lat_step_deg),
            search_lon_step_deg=float(args.search_lon_step_deg),
            headings_deg=[float(x) for x in args.headings_deg],
            scenario_name=scenario_name,
            region_name=str(args.region_name),
        )
        _write_scenario(scenario_path, scenario)
    else:
        scenario = _load_scenario(scenario_path)
        best = None

    lat_axis_deg, lon_axis_deg = _truth_bounds_deg(
        scenario,
        dt_s=float(args.dt_s),
        margin_deg=float(args.margin_deg),
        lat_step_deg=float(args.lat_step_deg),
        lon_step_deg=float(args.lon_step_deg),
    )

    if not raw_bathy_path.exists() or args.force_refresh_bathymetry:
        _sample_regular_bathymetry_csv(
            lat_axis_deg,
            lon_axis_deg,
            out_path=raw_bathy_path,
            max_workers=int(args.max_workers),
        )

    bathy_grid, bathy_manifest = process_regular_csv_bathymetry_grid(
        raw_bathy_path,
        processed_npz_path=processed_bathy_path,
        manifest_path=bathy_manifest_path,
        region_name=str(args.region_name),
        source_name="GEBCO_WMS_fixture",
        reference_surface_height_m=0.0,
        default_method="linear",
        bounds_error=False,
        fill_value_m=np.nan,
        metadata_extra={
            "wms_service": "https://wms.gebco.net/mapserv",
            "wms_layer": "GEBCO_LATEST_2",
        },
        project_root=PROJECT_ROOT,
    )
    if not isinstance(bathy_grid, BathymetryGrid):
        raise TypeError("Expected BathymetryGrid from process_regular_csv_bathymetry_grid.")

    demo_manifest = RegionalDemoPackManifest(
        region_name=str(args.region_name),
        gravity_manifest_path=str(gravity_manifest_path.resolve()),
        bathymetry_manifest_path=str(bathy_manifest_path.resolve()),
        scenario_path=str(scenario_path.resolve()),
        sequence_profile_path=str(sequence_profile_path.resolve()),
        notes=[
            "Gravity source is a processed regional gravity window already present in the repo-local cache.",
            "Bathymetry source is sampled from the public GEBCO WMS around the demo trajectory bounds.",
        ],
        metadata={
            "gravity_region": gravity_manifest.region_name,
            "bathymetry_region": bathy_manifest.region_name,
            "bathymetry_shape": list(bathy_manifest.shape),
            **({} if best is None else {"route_selection": best}),
        },
    )
    demo_manifest.write_json(demo_pack_manifest_path)

    print(f"Prepared scenario: {scenario_path}")
    if best is not None:
        print(
            "Best candidate: "
            f"lat={best['initial_lat_deg']:.3f} deg, "
            f"lon={best['initial_lon_deg']:.3f} deg, "
            f"heading={best['initial_heading_deg']:.1f} deg, "
            f"score={best['information_score']:.3f}"
        )
    print(f"Prepared bathymetry raw fixture: {raw_bathy_path}")
    print(f"Prepared bathymetry grid: {processed_bathy_path}")
    print(f"Prepared bathymetry manifest: {bathy_manifest_path}")
    print(f"Prepared demo pack manifest: {demo_pack_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
