#!/usr/bin/env python3
"""
Prepare the Norwegian maritime demo pack.

This script builds a small real bathymetry fixture from the public GEBCO WMS,
processes it into a BathymetryGrid cache, and writes a compact manifest linking
the in-repo Norwegian-margin gravity fixture and the bathymetry product for the
regional demo.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import urllib.parse
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from gravnav.datasets.bathymetry_loader import (
    DEMO_PACK_MANIFEST_NAME,
    PROCESSED_GRID_NAME,
    PROCESSED_MANIFEST_NAME,
    RAW_FIXTURE_NAME,
    BathymetryGrid,
    RegionalDemoPackManifest,
    load_bathymetry_manifest,
    process_regular_csv_bathymetry_grid,
)
from gravnav.datasets.gravity_loader import load_regional_manifest
from gravnav.simulation.runner import resolve_truth_trajectory
from gravnav.truth.scenarios import ScenarioSpec
from gravnav.utils.config import load_config_mapping

DEFAULT_SCENARIO = PROJECT_ROOT / "configs/scenarios/norwegian_margin_maritime.json"
DEFAULT_SEQUENCE_PROFILE = PROJECT_ROOT / "configs/sequence_profiles/norwegian_margin_maritime_demo.json"
DEFAULT_GRAVITY_MANIFEST = PROJECT_ROOT / "data/gravity_maps/processed/norwegian_margin_gravity_map_manifest.json"
DEFAULT_RAW_BATHY = PROJECT_ROOT / "data/bathymetry/raw/norwegian_margin_public" / RAW_FIXTURE_NAME
DEFAULT_PROCESSED_BATHY = PROJECT_ROOT / "data/bathymetry/processed" / PROCESSED_GRID_NAME
DEFAULT_BATHY_MANIFEST = PROJECT_ROOT / "data/bathymetry/processed" / PROCESSED_MANIFEST_NAME
DEFAULT_DEMO_PACK_MANIFEST = PROJECT_ROOT / "data/bathymetry/processed" / DEMO_PACK_MANIFEST_NAME


def _query_gebco_elevation_m(lat_deg: float, lon_deg: float) -> float:
    bbox = f"{lat_deg - 5.0e-4:.6f},{lon_deg - 5.0e-4:.6f},{lat_deg + 5.0e-4:.6f},{lon_deg + 5.0e-4:.6f}"
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
    raise RuntimeError(f"GEBCO WMS response did not contain value_list for lat={lat_deg}, lon={lon_deg}.")


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


def _truth_bounds_deg(scenario_path: Path, *, dt_s: float, margin_deg: float) -> tuple[np.ndarray, np.ndarray]:
    mapping = load_config_mapping(scenario_path)
    scenario = ScenarioSpec.from_mapping(mapping)
    truth = resolve_truth_trajectory(scenario, dt_s=dt_s)
    lat_min = float(np.rad2deg(np.min(truth.lat_rad))) - margin_deg
    lat_max = float(np.rad2deg(np.max(truth.lat_rad))) + margin_deg
    lon_min = float(np.rad2deg(np.min(truth.lon_rad))) - margin_deg
    lon_max = float(np.rad2deg(np.max(truth.lon_rad))) + margin_deg
    return (
        np.arange(lat_min, lat_max + 1.0e-12, dt_s * 0.0 + 0.02, dtype=np.float64),
        np.arange(lon_min, lon_max + 1.0e-12, dt_s * 0.0 + 0.03, dtype=np.float64),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the Norwegian maritime photonic demo pack.")
    parser.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    parser.add_argument("--sequence-profile", default=str(DEFAULT_SEQUENCE_PROFILE))
    parser.add_argument("--gravity-manifest", default=str(DEFAULT_GRAVITY_MANIFEST))
    parser.add_argument("--raw-bathymetry-csv", default=str(DEFAULT_RAW_BATHY))
    parser.add_argument("--processed-bathymetry", default=str(DEFAULT_PROCESSED_BATHY))
    parser.add_argument("--bathymetry-manifest", default=str(DEFAULT_BATHY_MANIFEST))
    parser.add_argument("--demo-pack-manifest", default=str(DEFAULT_DEMO_PACK_MANIFEST))
    parser.add_argument("--dt-s", type=float, default=2.0)
    parser.add_argument("--margin-deg", type=float, default=0.18)
    parser.add_argument("--lat-step-deg", type=float, default=0.02)
    parser.add_argument("--lon-step-deg", type=float, default=0.03)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--force-refresh-bathymetry", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    scenario_path = Path(args.scenario).expanduser().resolve()
    sequence_profile_path = Path(args.sequence_profile).expanduser().resolve()
    gravity_manifest_path = Path(args.gravity_manifest).expanduser().resolve()
    raw_bathy_path = Path(args.raw_bathymetry_csv).expanduser().resolve()
    processed_bathy_path = Path(args.processed_bathymetry).expanduser().resolve()
    bathy_manifest_path = Path(args.bathymetry_manifest).expanduser().resolve()
    demo_pack_manifest_path = Path(args.demo_pack_manifest).expanduser().resolve()

    gravity_manifest = load_regional_manifest(gravity_manifest_path)
    lat_axis_deg, lon_axis_deg = _truth_bounds_deg(
        scenario_path,
        dt_s=float(args.dt_s),
        margin_deg=float(args.margin_deg),
    )
    if float(args.lat_step_deg) != 0.02:
        lat_axis_deg = np.arange(lat_axis_deg[0], lat_axis_deg[-1] + 1.0e-12, float(args.lat_step_deg), dtype=np.float64)
    if float(args.lon_step_deg) != 0.03:
        lon_axis_deg = np.arange(lon_axis_deg[0], lon_axis_deg[-1] + 1.0e-12, float(args.lon_step_deg), dtype=np.float64)

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
        region_name="norwegian_margin_public",
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
        region_name="norwegian_margin_maritime_demo",
        gravity_manifest_path=str(gravity_manifest_path.resolve()),
        bathymetry_manifest_path=str(bathy_manifest_path.resolve()),
        scenario_path=str(scenario_path.resolve()),
        sequence_profile_path=str(sequence_profile_path.resolve()),
        notes=[
            "Gravity source is the bundled Norwegian-margin fixture already processed in the repo.",
            "Bathymetry source is sampled from the public GEBCO WMS around the demo trajectory bounds.",
        ],
        metadata={
            "gravity_region": gravity_manifest.region_name,
            "bathymetry_region": bathy_manifest.region_name,
            "bathymetry_shape": list(bathy_manifest.shape),
        },
    )
    demo_manifest.write_json(demo_pack_manifest_path)

    print(f"Prepared bathymetry raw fixture: {raw_bathy_path}")
    print(f"Prepared bathymetry grid: {processed_bathy_path}")
    print(f"Prepared bathymetry manifest: {bathy_manifest_path}")
    print(f"Prepared demo pack manifest: {demo_pack_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
