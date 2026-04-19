#!/usr/bin/env python3
"""
Offline bathymetry-proxy shim for external gravity-only regions.

Why this exists
---------------
The full multimodal onboarding path for a new external region goes

    gravity CSV  ->  gravity NPZ + manifest + scenario
                  +  GEBCO WMS bathymetry CSV  (network)
                  +  WMM magnetic grid         (network)
                  +  HYCOM currents grid       (network)
                  =  RegionalDemoPackManifest usable by the ML corpus builder.

During the wave-1 onboarding turn we need all 3 external regions
(mid_atlantic_ridge, iceland_greenland_margin, mariana_approach) registered
into the `all_wave1` corpus without network access. Faking bathymetry from
pure noise would inject meaningless signal, so instead this shim uses a
bounded *admittance proxy*: over oceanic crust the short-wavelength gravity
disturbance and seafloor topography are related by an approximately linear
admittance transfer function (Sandwell & Smith, 1997; Watts, 2001). The
proxy preserves the real spatial structure of the gravity map while making
it explicit via manifest metadata that the bathymetry channel is a proxy,
not an independent measurement.

The corpus builder and downstream sequence matcher treat bathymetry as an
optional supporting channel; this shim keeps that channel populated with
geophysically-informed structure so leave-one-region-out generalization
tests on the 7-region corpus are meaningful.

Outputs
-------
- data/bathymetry/processed/<region>_public_offline_bathymetry_grid.npz
- data/bathymetry/processed/<region>_public_offline_bathymetry_manifest.json
- data/bathymetry/processed/<region>_public_demo_pack.json

The produced demo-pack manifest matches the schema written by
``prepare_norwegian_maritime_demo.py`` and is a drop-in entry for
`all_wave1`.
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

from gravnav.datasets.bathymetry_loader import (
    BathymetryGrid,
    RegionalBathymetryManifest,
    RegionalDemoPackManifest,
)
from gravnav.datasets.gravity_loader import load_regional_map_from_manifest


# Mid-ocean values representative of deep pelagic crust. The proxy is intended
# to produce bathymetry variation on the correct *scale* and *sign* for the
# oceanic regions we target in wave-1; the absolute depth baseline varies by
# region so we expose it as a CLI knob.
DEFAULT_MEAN_DEPTH_M = -3500.0
# Empirical short-wavelength admittance over oceanic crust is typically in the
# 15-35 m/mGal band (Smith & Sandwell 1997; Watts 2001). Using 20 m/mGal keeps
# the derived topography within physical limits for this corpus.
DEFAULT_ADMITTANCE_M_PER_MGAL = 20.0
# Cap the proxy between physical limits for ocean seafloor so the corpus
# builder never sees pathological values.
MIN_PROXY_ELEVATION_M = -8000.0
MAX_PROXY_ELEVATION_M = -50.0


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _build_proxy_bathymetry(
    *,
    gravity_manifest_path: Path,
    mean_depth_m: float,
    admittance_m_per_mgal: float,
    region_name: str,
) -> tuple[BathymetryGrid, dict[str, Any]]:
    gravity_map, gravity_manifest, _, _ = load_regional_map_from_manifest(
        gravity_manifest_path
    )
    grid_mgal = gravity_map.disturbance_grid_mgal
    admittance = float(admittance_m_per_mgal)
    baseline = float(mean_depth_m)

    # Remove the regional mean of the gravity field before applying admittance
    # so the baseline depth argument controls the absolute depth level while the
    # proxy variation captures only the spatial structure.
    grid_mgal_demeaned = grid_mgal - float(np.mean(grid_mgal))
    elevation_grid_m = np.clip(
        baseline + admittance * grid_mgal_demeaned,
        MIN_PROXY_ELEVATION_M,
        MAX_PROXY_ELEVATION_M,
    ).astype(np.float64)

    lat_axis_deg = np.rad2deg(np.asarray(gravity_map.lat_axis_rad, dtype=np.float64))
    lon_axis_deg = np.rad2deg(np.asarray(gravity_map.lon_axis_rad, dtype=np.float64))

    stats = {
        "gravity_manifest_path": _relative(gravity_manifest_path),
        "gravity_region_name": gravity_manifest.region_name,
        "mean_depth_m": baseline,
        "admittance_m_per_mgal": admittance,
        "gravity_demeaned_mgal_std": float(np.std(grid_mgal_demeaned)),
        "elevation_min_m": float(np.min(elevation_grid_m)),
        "elevation_max_m": float(np.max(elevation_grid_m)),
        "elevation_mean_m": float(np.mean(elevation_grid_m)),
        "elevation_std_m": float(np.std(elevation_grid_m)),
    }

    bathymetry = BathymetryGrid(
        lat_axis_deg=lat_axis_deg,
        lon_axis_deg=lon_axis_deg,
        elevation_grid_m=elevation_grid_m,
        reference_surface_height_m=0.0,
        default_method="linear",
        bounds_error=False,
        fill_value_m=np.nan,
        name=f"{region_name}_offline_bathymetry_proxy",
        metadata={
            "region_name": str(region_name),
            "source_kind": "gravity_admittance_proxy",
            "proxy_notes": [
                "Short-wavelength bathymetry proxy derived from the regional "
                "gravity disturbance via a fixed admittance transfer function.",
                "This is NOT an independent bathymetry measurement; it is a "
                "geophysically-motivated proxy used only to keep the "
                "multimodal ML channel populated for offline corpus builds.",
            ],
            **stats,
        },
    )
    return bathymetry, stats


def _write_bathymetry_manifest(
    bathymetry: BathymetryGrid,
    *,
    grid_path: Path,
    manifest_path: Path,
    region_name: str,
    source_name: str,
    raw_data_path: Path,
    stats: dict[str, Any],
) -> RegionalBathymetryManifest:
    lat_axis = np.asarray(bathymetry.lat_axis_deg, dtype=np.float64)
    lon_axis = np.asarray(bathymetry.lon_axis_deg, dtype=np.float64)
    manifest = RegionalBathymetryManifest(
        region_name=str(region_name),
        source_name=str(source_name),
        source_kind="gravity_admittance_proxy",
        raw_data_path=_relative(raw_data_path),
        processed_grid_path=_relative(grid_path),
        manifest_path=_relative(manifest_path),
        elevation_units="m",
        reference_surface_height_m=0.0,
        lat_bounds_deg=(float(lat_axis[0]), float(lat_axis[-1])),
        lon_bounds_deg=(float(lon_axis[0]), float(lon_axis[-1])),
        spacing_deg=(
            float(np.mean(np.diff(lat_axis))),
            float(np.mean(np.diff(lon_axis))),
        ),
        shape=(int(lat_axis.size), int(lon_axis.size)),
        interpolation=str(bathymetry.default_method),
        notes=[
            "Derived from regional gravity disturbance via a linear admittance proxy.",
            "Used only as a structural supporting channel for offline corpus builds.",
        ],
        metadata=stats,
    )
    manifest.write_json(manifest_path)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an offline bathymetry-proxy demo pack for a wave-1 region."
    )
    parser.add_argument("--region-name", required=True)
    parser.add_argument("--gravity-manifest", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--sequence-profile",
        default=str(PROJECT_ROOT / "configs/sequence_profiles/public_offshore_locked.json"),
    )
    parser.add_argument("--processed-bathymetry", required=True)
    parser.add_argument("--bathymetry-manifest", required=True)
    parser.add_argument("--demo-pack-manifest", required=True)
    parser.add_argument(
        "--mean-depth-m",
        type=float,
        default=DEFAULT_MEAN_DEPTH_M,
    )
    parser.add_argument(
        "--admittance-m-per-mgal",
        type=float,
        default=DEFAULT_ADMITTANCE_M_PER_MGAL,
    )
    parser.add_argument(
        "--source-name",
        default="gravity_admittance_proxy_v1",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    gravity_manifest_path = Path(args.gravity_manifest).expanduser().resolve()
    scenario_path = Path(args.scenario).expanduser().resolve()
    sequence_profile_path = Path(args.sequence_profile).expanduser().resolve()
    processed_bathymetry_path = Path(args.processed_bathymetry).expanduser().resolve()
    bathymetry_manifest_path = Path(args.bathymetry_manifest).expanduser().resolve()
    demo_pack_manifest_path = Path(args.demo_pack_manifest).expanduser().resolve()

    bathymetry, stats = _build_proxy_bathymetry(
        gravity_manifest_path=gravity_manifest_path,
        mean_depth_m=float(args.mean_depth_m),
        admittance_m_per_mgal=float(args.admittance_m_per_mgal),
        region_name=str(args.region_name),
    )
    processed_bathymetry_path.parent.mkdir(parents=True, exist_ok=True)
    bathymetry.to_npz(processed_bathymetry_path)
    _write_bathymetry_manifest(
        bathymetry,
        grid_path=processed_bathymetry_path,
        manifest_path=bathymetry_manifest_path,
        region_name=str(args.region_name),
        source_name=str(args.source_name),
        raw_data_path=gravity_manifest_path,
        stats=stats,
    )

    demo_manifest = RegionalDemoPackManifest(
        region_name=str(args.region_name),
        gravity_manifest_path=str(gravity_manifest_path),
        bathymetry_manifest_path=str(bathymetry_manifest_path),
        scenario_path=str(scenario_path),
        sequence_profile_path=str(sequence_profile_path),
        notes=[
            "Offline wave-1 demo pack: bathymetry is a gravity-admittance proxy.",
            "No magnetic or current channels are attached in the offline build.",
        ],
        metadata={
            "bathymetry_source_kind": "gravity_admittance_proxy",
            "offline_build": True,
            "proxy_stats": stats,
        },
    )
    demo_manifest.write_json(demo_pack_manifest_path)

    print(
        json.dumps(
            {
                "region_name": str(args.region_name),
                "bathymetry_grid": _relative(processed_bathymetry_path),
                "bathymetry_manifest": _relative(bathymetry_manifest_path),
                "demo_pack_manifest": _relative(demo_pack_manifest_path),
                "proxy_stats": stats,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
