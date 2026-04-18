#!/usr/bin/env python3
"""
Prepare public magnetic/current assets for one maritime demo pack.

This generic path assumes the input demo pack already references a processed
gravity map plus a bathymetry surface such as GEBCO/EMODnet. It augments that
pack with:

- scalar magnetic total field from NOAA WMMHR2025
- scalar magnetic anomaly as WMMHR2025 - WMM2025
- static surface current field from HYCOM GLBy0.08 point queries
- an optional tide-correction config reference
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable
import urllib.parse
import urllib.request
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from gravnav.datasets.bathymetry_loader import (
    RegionalDemoPackManifest,
    ResolvedRegionalDemoPack,
    load_demo_pack_manifest,
    resolve_regional_demo_pack,
)
from gravnav.datasets.current_loader import process_regular_csv_current_field
from gravnav.datasets.magnetic_loader import process_regular_csv_magnetic_grid

DEFAULT_VENDOR_CACHE_DIR = PROJECT_ROOT / "data/magnetic/raw/vendor_cache"
DEFAULT_CURRENT_LAT_STEP_DEG = 0.04
DEFAULT_CURRENT_LON_STEP_DEG = 0.06
DEFAULT_CURRENT_DEPTH_M = 0.0
DEFAULT_MAX_WORKERS = 6
HYCOM_DATASET_URL = "https://ncss.hycom.org/thredds/ncss/grid/GLBy0.08/latest"

WMM_VARIANTS = {
    "wmm2025": {
        "zip_url": "https://www.ngdc.noaa.gov/geomag/WMM/data/WMM2025/wmm2025_Linux.zip",
        "archive_dir": "wmm2025_Linux",
        "exe_name": "wmm_file",
    },
    "wmmhr2025": {
        "zip_url": "https://www.ngdc.noaa.gov/geomag/WMMHR/data/WMMHR2025/wmmhr2025_Linux.zip",
        "archive_dir": "wmmhr2025_Linux",
        "exe_name": "wmmhr_file",
    },
}


def _decimal_year_from_iso8601(iso_time: str) -> float:
    dt = datetime.fromisoformat(str(iso_time).replace("Z", "+00:00")).astimezone(
        timezone.utc
    )
    year_start = datetime(dt.year, 1, 1, tzinfo=timezone.utc)
    next_year_start = datetime(dt.year + 1, 1, 1, tzinfo=timezone.utc)
    elapsed = (dt - year_start).total_seconds()
    duration = (next_year_start - year_start).total_seconds()
    return float(dt.year + elapsed / duration)


def _regular_axis(min_deg: float, max_deg: float, step_deg: float) -> np.ndarray:
    if step_deg <= 0.0:
        raise ValueError("step_deg must be positive.")
    return np.arange(
        float(min_deg),
        float(max_deg) + 1.0e-12,
        float(step_deg),
        dtype=np.float64,
    )


def _download_file(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120.0) as resp:
        payload = resp.read()
    out_path.write_bytes(payload)
    return out_path


def _ensure_wmm_variant(variant: str, *, vendor_cache_dir: Path) -> Path:
    cfg = WMM_VARIANTS[variant]
    archive_dir = vendor_cache_dir / cfg["archive_dir"]
    build_dir = archive_dir / "build"
    exe_path = build_dir / cfg["exe_name"]
    if exe_path.exists():
        return exe_path

    vendor_cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = vendor_cache_dir / f"{cfg['archive_dir']}.zip"
    if not zip_path.exists():
        _download_file(str(cfg["zip_url"]), zip_path)

    if not archive_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(vendor_cache_dir)

    subprocess.run(
        ["make", "clean", cfg["exe_name"]],
        cwd=archive_dir,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if not exe_path.exists():
        raise FileNotFoundError(f"Failed to build {variant} executable at {exe_path}.")
    return exe_path


def _run_wmm_file(
    exe_path: Path,
    *,
    decimal_year: float,
    lat_axis_deg: np.ndarray,
    lon_axis_deg: np.ndarray,
    altitude_m: float,
    use_height_above_ellipsoid: bool,
    out_dir: Path,
    tag: str,
) -> dict[tuple[float, float], float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = exe_path.parent
    unique_tag = f"{out_dir.name}_{tag}"
    input_path = work_dir / f"{unique_tag}_input.txt"
    output_path = work_dir / f"{unique_tag}_output.txt"
    height_mode = "E" if use_height_above_ellipsoid else "M"
    altitude_token = f"M{float(altitude_m):.3f}"
    lines = []
    for lat_deg in lat_axis_deg:
        for lon_deg in lon_axis_deg:
            lines.append(
                f"{decimal_year:.8f} {height_mode} {altitude_token} "
                f"{float(lat_deg):.6f} {float(lon_deg):.6f}"
            )
    input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run(
        [str(exe_path), "f", str(input_path.name), str(output_path.name)],
        cwd=work_dir,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return _parse_wmm_output_total_field(output_path)


def _parse_wmm_output_total_field(path: Path) -> dict[tuple[float, float], float]:
    rows: dict[tuple[float, float], float] = {}
    started = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Date Coord-System Altitude Latitude Longitude"):
            started = True
            continue
        if not started:
            continue
        tokens = line.split()
        if len(tokens) < 14:
            continue
        lat_deg = float(tokens[3])
        lon_deg = float(tokens[4])
        total_field_nt = float(tokens[13])
        rows[(lat_deg, lon_deg)] = total_field_nt
    if not rows:
        raise ValueError(f"Failed to parse any WMM output rows from {path}.")
    return rows


def _write_regular_magnetic_csv(
    path: Path,
    *,
    lat_axis_deg: np.ndarray,
    lon_axis_deg: np.ndarray,
    total_field_rows_nt: dict[tuple[float, float], float],
    anomaly_rows_nt: dict[tuple[float, float], float],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["lat_deg", "lon_deg", "total_field_nt", "anomaly_nt"])
        for lat_deg in lat_axis_deg:
            for lon_deg in lon_axis_deg:
                key = (float(lat_deg), float(lon_deg))
                writer.writerow(
                    [
                        f"{float(lat_deg):.6f}",
                        f"{float(lon_deg):.6f}",
                        f"{float(total_field_rows_nt[key]):.6f}",
                        f"{float(anomaly_rows_nt[key]):.6f}",
                    ]
                )
    return path


def _hycom_point_query_url(
    *,
    lat_deg: float,
    lon_deg: float,
    depth_m: float,
    snapshot_time_iso: str | None,
) -> str:
    query = {
        "var": ["water_u", "water_v"],
        "latitude": f"{float(lat_deg):.6f}",
        "longitude": f"{float(lon_deg):.6f}",
        "vertCoord": f"{float(depth_m):.3f}",
        "accept": "csv",
    }
    if snapshot_time_iso is not None:
        query["time"] = str(snapshot_time_iso)
    return HYCOM_DATASET_URL + "?" + urllib.parse.urlencode(query, doseq=True)


def _parse_hycom_point_csv(payload: str) -> tuple[str, float, float]:
    reader = csv.DictReader(io.StringIO(payload))
    rows = [row for row in reader if row]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one HYCOM row, got {len(rows)}.")
    row = rows[0]
    time_iso = str(row["time"])
    east_current_mps = float(row['water_u[unit="m/s"]'])
    north_current_mps = float(row['water_v[unit="m/s"]'])
    return time_iso, north_current_mps, east_current_mps


def _query_hycom_point(
    *,
    lat_deg: float,
    lon_deg: float,
    depth_m: float,
    snapshot_time_iso: str | None,
) -> tuple[str, float, float]:
    url = _hycom_point_query_url(
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        depth_m=depth_m,
        snapshot_time_iso=snapshot_time_iso,
    )
    with urllib.request.urlopen(url, timeout=60.0) as resp:
        payload = resp.read().decode("utf-8")
    return _parse_hycom_point_csv(payload)


def _fill_invalid_current_rows(
    rows: list[tuple[float, float, float, float]],
    *,
    lat_axis_deg: np.ndarray,
    lon_axis_deg: np.ndarray,
) -> tuple[list[tuple[float, float, float, float]], dict[str, object]]:
    lat_lookup = {
        float(lat): i for i, lat in enumerate(np.asarray(lat_axis_deg, dtype=np.float64))
    }
    lon_lookup = {
        float(lon): j for j, lon in enumerate(np.asarray(lon_axis_deg, dtype=np.float64))
    }
    north_grid = np.full((lat_axis_deg.size, lon_axis_deg.size), np.nan, dtype=np.float64)
    east_grid = np.full((lat_axis_deg.size, lon_axis_deg.size), np.nan, dtype=np.float64)
    for lat_deg, lon_deg, north_mps, east_mps in rows:
        i = lat_lookup[float(lat_deg)]
        j = lon_lookup[float(lon_deg)]
        north_grid[i, j] = float(north_mps)
        east_grid[i, j] = float(east_mps)

    valid_mask = np.isfinite(north_grid) & np.isfinite(east_grid)
    num_invalid = int(np.size(valid_mask) - np.count_nonzero(valid_mask))
    if num_invalid == 0:
        return rows, {
            "num_invalid_points": 0,
            "fill_strategy": "none",
            "all_invalid_fallback_zero": False,
        }
    if not np.any(valid_mask):
        north_grid[:] = 0.0
        east_grid[:] = 0.0
        fill_strategy = "all_zero_fallback"
        all_invalid = True
    else:
        valid_indices = np.argwhere(valid_mask)
        invalid_indices = np.argwhere(~valid_mask)
        for invalid_i, invalid_j in invalid_indices:
            deltas = valid_indices - np.array([invalid_i, invalid_j], dtype=np.int64)
            best_idx = int(np.argmin(np.sum(deltas.astype(np.float64) ** 2, axis=1)))
            src_i, src_j = valid_indices[best_idx]
            north_grid[invalid_i, invalid_j] = north_grid[src_i, src_j]
            east_grid[invalid_i, invalid_j] = east_grid[src_i, src_j]
        fill_strategy = "nearest_valid"
        all_invalid = False

    filled_rows: list[tuple[float, float, float, float]] = []
    for lat_deg in lat_axis_deg:
        for lon_deg in lon_axis_deg:
            i = lat_lookup[float(lat_deg)]
            j = lon_lookup[float(lon_deg)]
            filled_rows.append(
                (
                    float(lat_deg),
                    float(lon_deg),
                    float(north_grid[i, j]),
                    float(east_grid[i, j]),
                )
            )
    return filled_rows, {
        "num_invalid_points": num_invalid,
        "fill_strategy": fill_strategy,
        "all_invalid_fallback_zero": all_invalid,
    }


def _write_regular_current_csv(
    path: Path,
    *,
    lat_axis_deg: np.ndarray,
    lon_axis_deg: np.ndarray,
    depth_m: float,
    snapshot_time_iso: str,
    max_workers: int,
) -> tuple[Path, str, dict[str, object]]:
    from concurrent.futures import ThreadPoolExecutor

    path.parent.mkdir(parents=True, exist_ok=True)
    points = [(float(lat), float(lon)) for lat in lat_axis_deg for lon in lon_axis_deg]

    def one(point: tuple[float, float]) -> tuple[float, float, float, float]:
        lat_deg, lon_deg = point
        _, north_mps, east_mps = _query_hycom_point(
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            depth_m=depth_m,
            snapshot_time_iso=snapshot_time_iso,
        )
        return lat_deg, lon_deg, north_mps, east_mps

    rows: list[tuple[float, float, float, float]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        for row in ex.map(one, points):
            rows.append(row)

    rows, fill_stats = _fill_invalid_current_rows(
        rows,
        lat_axis_deg=lat_axis_deg,
        lon_axis_deg=lon_axis_deg,
    )
    rows.sort(key=lambda item: (item[0], item[1]))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "lat_deg",
                "lon_deg",
                "depth_m",
                "north_current_mps",
                "east_current_mps",
            ]
        )
        for lat_deg, lon_deg, north_mps, east_mps in rows:
            writer.writerow(
                [
                    f"{lat_deg:.6f}",
                    f"{lon_deg:.6f}",
                    f"{float(depth_m):.3f}",
                    f"{north_mps:.9f}",
                    f"{east_mps:.9f}",
                ]
            )
    return path, snapshot_time_iso, fill_stats


def _derive_output_paths(
    *,
    project_root: Path,
    region_name: str,
    input_pack_path: Path,
) -> dict[str, Path]:
    safe_region = str(region_name).strip().replace(" ", "_")
    pack_stem = input_pack_path.stem
    if pack_stem.endswith("_demo_pack"):
        output_pack_name = f"{pack_stem[:-10]}_multimodal_demo_pack.json"
    else:
        output_pack_name = f"{pack_stem}_multimodal.json"
    return {
        "magnetic_raw_csv": project_root / "data" / "magnetic" / "raw" / safe_region / f"{safe_region}_wmmhr_wmm2025_grid.csv",
        "magnetic_processed_npz": project_root / "data" / "magnetic" / "processed" / f"{safe_region}_magnetic_grid.npz",
        "magnetic_manifest": project_root / "data" / "magnetic" / "processed" / f"{safe_region}_magnetic_manifest.json",
        "current_raw_csv": project_root / "data" / "current" / "raw" / safe_region / f"{safe_region}_hycom_surface_current.csv",
        "current_processed_npz": project_root / "data" / "current" / "processed" / f"{safe_region}_current_field.npz",
        "current_manifest": project_root / "data" / "current" / "processed" / f"{safe_region}_current_manifest.json",
        "output_pack": input_pack_path.parent / output_pack_name,
    }


def _augment_demo_pack(
    base_manifest: RegionalDemoPackManifest,
    *,
    magnetic_manifest_path: Path,
    current_manifest_path: Path,
    tide_config_path: Path | None,
    snapshot_time_iso: str,
    decimal_year: float,
) -> RegionalDemoPackManifest:
    notes = list(base_manifest.notes)
    notes.extend(
        [
            "Magnetic map prepared from official NOAA WMMHR2025 total field.",
            "Magnetic anomaly prepared as WMMHR2025 total field minus WMM2025 total field.",
            "Current field prepared from official HYCOM GLBy0.08 point queries.",
        ]
    )
    if tide_config_path is not None:
        notes.append("Tide correction config reference preserved for runtime use.")
    metadata = dict(base_manifest.metadata)
    metadata.update(
        {
            "multimodal_public_pack": True,
            "snapshot_time_iso": str(snapshot_time_iso),
            "magnetic_decimal_year": float(decimal_year),
        }
    )
    return replace(
        base_manifest,
        magnetic_manifest_path=str(magnetic_manifest_path.resolve()),
        current_manifest_path=str(current_manifest_path.resolve()),
        tide_config_path=(
            None if tide_config_path is None else str(tide_config_path.resolve())
        ),
        notes=notes,
        metadata=metadata,
    )


def _prepare_assets(
    demo_pack: ResolvedRegionalDemoPack,
    *,
    input_pack_path: Path,
    output_pack_path: Path | None,
    vendor_cache_dir: Path,
    tide_config_path: Path | None,
    current_lat_step_deg: float,
    current_lon_step_deg: float,
    current_depth_m: float,
    snapshot_time_iso: str | None,
    magnetic_decimal_year: float | None,
    max_workers: int,
) -> Path:
    paths = _derive_output_paths(
        project_root=PROJECT_ROOT,
        region_name=demo_pack.manifest.region_name,
        input_pack_path=input_pack_path,
    )
    if output_pack_path is None:
        output_pack_path = paths["output_pack"]

    if snapshot_time_iso is None:
        center_lat = float(np.mean(demo_pack.bathymetry_grid.lat_bounds_deg))
        center_lon = float(np.mean(demo_pack.bathymetry_grid.lon_bounds_deg))
        snapshot_time_iso, _, _ = _query_hycom_point(
            lat_deg=center_lat,
            lon_deg=center_lon,
            depth_m=current_depth_m,
            snapshot_time_iso=None,
        )
    if magnetic_decimal_year is None:
        magnetic_decimal_year = _decimal_year_from_iso8601(snapshot_time_iso)

    lat_axis_mag = np.asarray(demo_pack.bathymetry_grid.lat_axis_deg, dtype=np.float64)
    lon_axis_mag = np.asarray(demo_pack.bathymetry_grid.lon_axis_deg, dtype=np.float64)

    wmmhr_exe = _ensure_wmm_variant("wmmhr2025", vendor_cache_dir=vendor_cache_dir)
    wmm_exe = _ensure_wmm_variant("wmm2025", vendor_cache_dir=vendor_cache_dir)
    model_cache_dir = vendor_cache_dir / "runs" / demo_pack.manifest.region_name
    wmmhr_rows = _run_wmm_file(
        wmmhr_exe,
        decimal_year=float(magnetic_decimal_year),
        lat_axis_deg=lat_axis_mag,
        lon_axis_deg=lon_axis_mag,
        altitude_m=0.0,
        use_height_above_ellipsoid=False,
        out_dir=model_cache_dir,
        tag="wmmhr2025",
    )
    wmm_rows = _run_wmm_file(
        wmm_exe,
        decimal_year=float(magnetic_decimal_year),
        lat_axis_deg=lat_axis_mag,
        lon_axis_deg=lon_axis_mag,
        altitude_m=0.0,
        use_height_above_ellipsoid=False,
        out_dir=model_cache_dir,
        tag="wmm2025",
    )
    anomaly_rows = {key: float(wmmhr_rows[key] - wmm_rows[key]) for key in wmmhr_rows}
    _write_regular_magnetic_csv(
        paths["magnetic_raw_csv"],
        lat_axis_deg=lat_axis_mag,
        lon_axis_deg=lon_axis_mag,
        total_field_rows_nt=wmmhr_rows,
        anomaly_rows_nt=anomaly_rows,
    )
    _, magnetic_manifest = process_regular_csv_magnetic_grid(
        paths["magnetic_raw_csv"],
        region_name=str(demo_pack.manifest.region_name),
        source_name="NOAA_WMMHR2025_minus_WMM2025",
        processed_map_path=paths["magnetic_processed_npz"],
        manifest_path=paths["magnetic_manifest"],
        project_root=PROJECT_ROOT,
        reference_height_m=0.0,
        notes=[
            "Total field from NOAA WMMHR2025 sampled on the demo bathymetry grid.",
            "Anomaly defined as WMMHR2025 total field minus WMM2025 total field.",
        ],
        metadata_extra={
            "snapshot_time_iso": str(snapshot_time_iso),
            "decimal_year": float(magnetic_decimal_year),
            "primary_model": "WMMHR2025",
            "reference_model": "WMM2025",
        },
    )

    lat_min, lat_max = demo_pack.bathymetry_grid.lat_bounds_deg
    lon_min, lon_max = demo_pack.bathymetry_grid.lon_bounds_deg
    lat_axis_current = _regular_axis(lat_min, lat_max, current_lat_step_deg)
    lon_axis_current = _regular_axis(lon_min, lon_max, current_lon_step_deg)
    _, _, current_fill_stats = _write_regular_current_csv(
        paths["current_raw_csv"],
        lat_axis_deg=lat_axis_current,
        lon_axis_deg=lon_axis_current,
        depth_m=float(current_depth_m),
        snapshot_time_iso=str(snapshot_time_iso),
        max_workers=int(max_workers),
    )
    _, current_manifest = process_regular_csv_current_field(
        paths["current_raw_csv"],
        region_name=str(demo_pack.manifest.region_name),
        source_name="HYCOM_GLBy0.08_latest_point_csv",
        processed_grid_path=paths["current_processed_npz"],
        manifest_path=paths["current_manifest"],
        project_root=PROJECT_ROOT,
        notes=[
            "Surface current field sampled from HYCOM GLBy0.08 latest point queries.",
            "water_u mapped to east_current_mps and water_v mapped to north_current_mps.",
        ],
        metadata_extra={
            "snapshot_time_iso": str(snapshot_time_iso),
            "depth_m": float(current_depth_m),
            "dataset_url": HYCOM_DATASET_URL,
            **current_fill_stats,
        },
    )

    augmented_manifest = _augment_demo_pack(
        load_demo_pack_manifest(input_pack_path),
        magnetic_manifest_path=paths["magnetic_manifest"],
        current_manifest_path=paths["current_manifest"],
        tide_config_path=tide_config_path,
        snapshot_time_iso=str(snapshot_time_iso),
        decimal_year=float(magnetic_decimal_year),
    )
    augmented_manifest.write_json(output_pack_path)

    print(f"Prepared magnetic manifest: {magnetic_manifest.manifest_path}")
    print(f"Prepared current manifest: {current_manifest.manifest_path}")
    print(f"Prepared augmented demo pack: {output_pack_path.resolve()}")
    print(f"Snapshot time: {snapshot_time_iso}")
    print(f"Magnetic decimal year: {magnetic_decimal_year:.8f}")
    return output_pack_path.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare public magnetic/current assets for one maritime demo pack."
    )
    parser.add_argument(
        "--demo-pack-manifest",
        required=True,
        help="Existing gravity+bathymetry demo-pack manifest to augment.",
    )
    parser.add_argument(
        "--output-demo-pack-manifest",
        default="",
        help="Optional output path for the augmented demo pack.",
    )
    parser.add_argument(
        "--vendor-cache-dir",
        default=str(DEFAULT_VENDOR_CACHE_DIR),
        help="Local cache directory for NOAA WMM/WMMHR downloads and builds.",
    )
    parser.add_argument(
        "--tide-config",
        default="",
        help="Optional tide/datum correction config referenced by the augmented demo pack.",
    )
    parser.add_argument(
        "--snapshot-time-iso",
        default="",
        help="Optional fixed ISO-8601 UTC time for HYCOM queries.",
    )
    parser.add_argument(
        "--magnetic-decimal-year",
        type=float,
        default=None,
        help="Optional decimal year override for WMM/WMMHR evaluation.",
    )
    parser.add_argument(
        "--current-lat-step-deg",
        type=float,
        default=DEFAULT_CURRENT_LAT_STEP_DEG,
    )
    parser.add_argument(
        "--current-lon-step-deg",
        type=float,
        default=DEFAULT_CURRENT_LON_STEP_DEG,
    )
    parser.add_argument(
        "--current-depth-m",
        type=float,
        default=DEFAULT_CURRENT_DEPTH_M,
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    input_pack_path = Path(args.demo_pack_manifest).expanduser().resolve()
    output_pack_path = (
        None
        if not str(args.output_demo_pack_manifest).strip()
        else Path(args.output_demo_pack_manifest).expanduser().resolve()
    )
    demo_pack = resolve_regional_demo_pack(input_pack_path)
    _prepare_assets(
        demo_pack,
        input_pack_path=input_pack_path,
        output_pack_path=output_pack_path,
        vendor_cache_dir=Path(args.vendor_cache_dir).expanduser().resolve(),
        tide_config_path=(
            None
            if not str(args.tide_config).strip()
            else Path(args.tide_config).expanduser().resolve()
        ),
        current_lat_step_deg=float(args.current_lat_step_deg),
        current_lon_step_deg=float(args.current_lon_step_deg),
        current_depth_m=float(args.current_depth_m),
        snapshot_time_iso=(
            None
            if not str(args.snapshot_time_iso).strip()
            else str(args.snapshot_time_iso).strip()
        ),
        magnetic_decimal_year=(
            None
            if args.magnetic_decimal_year is None
            else float(args.magnetic_decimal_year)
        ),
        max_workers=int(args.max_workers),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
