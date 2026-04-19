#!/usr/bin/env python3
"""
Extract a regional regular-grid CSV from the public Scripps/Sandwell marine gravity NetCDF.

This is a format-conversion helper only. The downstream gravity prep path still consumes
the existing supported `regular_csv` input format.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import netCDF4
import numpy as np


def _wrap_longitudes_deg(lon_deg: np.ndarray) -> np.ndarray:
    wrapped = (np.asarray(lon_deg, dtype=np.float64) + 180.0) % 360.0 - 180.0
    wrapped[np.isclose(wrapped, -180.0)] = 180.0
    return wrapped


def _read_netcdf_axes_and_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with netCDF4.Dataset(path) as ds:
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64).copy()
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64).copy()

        grid_var = None
        for name in ("z", "grav", "gravity", "gravity_anomaly"):
            if name in ds.variables:
                grid_var = ds.variables[name]
                break
        if grid_var is None:
            available = ", ".join(sorted(str(name) for name in ds.variables))
            raise KeyError(
                f"Could not find gravity variable in {path.name}. Variables: {available}"
            )
        grid = np.asarray(grid_var[:], dtype=np.float64).copy()
    return lat, lon, grid


def _extract_region(
    *,
    lat_axis_deg: np.ndarray,
    lon_axis_deg: np.ndarray,
    grid_mgal: np.ndarray,
    lat_min_deg: float,
    lat_max_deg: float,
    lon_min_deg: float,
    lon_max_deg: float,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if grid_mgal.shape != (lat_axis_deg.size, lon_axis_deg.size):
        if grid_mgal.shape == (lon_axis_deg.size, lat_axis_deg.size):
            grid_mgal = grid_mgal.T
        else:
            raise ValueError(
                "Unexpected gravity grid shape. "
                f"Expected {(lat_axis_deg.size, lon_axis_deg.size)} or "
                f"{(lon_axis_deg.size, lat_axis_deg.size)}, got {grid_mgal.shape}."
            )

    wrapped_lon_axis_deg = _wrap_longitudes_deg(lon_axis_deg)
    lon_order = np.argsort(wrapped_lon_axis_deg)
    lon_axis_sorted_deg = wrapped_lon_axis_deg[lon_order]
    grid_sorted = grid_mgal[:, lon_order]

    lat_keep = (
        (lat_axis_deg >= float(lat_min_deg))
        & (lat_axis_deg <= float(lat_max_deg))
    )
    lon_keep = (
        (lon_axis_sorted_deg >= float(lon_min_deg))
        & (lon_axis_sorted_deg <= float(lon_max_deg))
    )
    if not np.any(lat_keep):
        raise ValueError("Latitude bounds do not intersect the Scripps grid.")
    if not np.any(lon_keep):
        raise ValueError("Longitude bounds do not intersect the Scripps grid.")

    lat_subset = lat_axis_deg[lat_keep][:: int(stride)]
    lon_subset = lon_axis_sorted_deg[lon_keep][:: int(stride)]
    grid_subset = grid_sorted[np.ix_(lat_keep, lon_keep)][:: int(stride), :: int(stride)]
    if grid_subset.shape != (lat_subset.size, lon_subset.size):
        raise ValueError(
            "Subset shape mismatch after stride application. "
            f"Got grid {grid_subset.shape}, lat {lat_subset.size}, lon {lon_subset.size}."
        )
    return lat_subset, lon_subset, grid_subset


def _write_regular_csv(
    *,
    out_path: Path,
    lat_axis_deg: np.ndarray,
    lon_axis_deg: np.ndarray,
    disturbance_grid_mgal: np.ndarray,
    source_url: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "lat_deg",
                "lon_deg",
                "disturbance_mgal",
                "source_url",
            ]
        )
        for lat_idx, lat_deg in enumerate(lat_axis_deg):
            for lon_idx, lon_deg in enumerate(lon_axis_deg):
                value = float(disturbance_grid_mgal[lat_idx, lon_idx])
                if not np.isfinite(value):
                    continue
                writer.writerow(
                    [
                        f"{float(lat_deg):.8f}",
                        f"{float(lon_deg):.8f}",
                        f"{value:.6f}",
                        source_url,
                    ]
                )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract one regional regular-grid CSV from the Scripps/Sandwell gravity NetCDF."
    )
    parser.add_argument("--input-netcdf", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--lat-min", required=True, type=float)
    parser.add_argument("--lat-max", required=True, type=float)
    parser.add_argument("--lon-min", required=True, type=float)
    parser.add_argument("--lon-max", required=True, type=float)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Optional integer decimation stride on the source 1-arcmin grid.",
    )
    parser.add_argument(
        "--source-url",
        default="https://topex.ucsd.edu/pub/global_grav_1min/grav_33.1.nc",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    in_path = Path(args.input_netcdf).expanduser().resolve()
    out_path = Path(args.output_csv).expanduser().resolve()
    stride = max(1, int(args.stride))

    lat_axis_deg, lon_axis_deg, grid_mgal = _read_netcdf_axes_and_grid(in_path)
    lat_subset, lon_subset, grid_subset = _extract_region(
        lat_axis_deg=lat_axis_deg,
        lon_axis_deg=lon_axis_deg,
        grid_mgal=grid_mgal,
        lat_min_deg=float(args.lat_min),
        lat_max_deg=float(args.lat_max),
        lon_min_deg=float(args.lon_min),
        lon_max_deg=float(args.lon_max),
        stride=stride,
    )
    _write_regular_csv(
        out_path=out_path,
        lat_axis_deg=lat_subset,
        lon_axis_deg=lon_subset,
        disturbance_grid_mgal=grid_subset,
        source_url=str(args.source_url),
    )
    print(
        f"Wrote {out_path} with shape {grid_subset.shape} "
        f"for lat [{lat_subset[0]}, {lat_subset[-1]}], "
        f"lon [{lon_subset[0]}, {lon_subset[-1]}]."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
