#!/usr/bin/env python3
"""
Prepare official EMODnet Bathymetry tiles for an existing Norway maritime demo pack.

This script keeps the regional demo-pack structure intact and swaps only the
bathymetry surface to EMODnet's 2022 mean-depth WCS coverage. Any existing
magnetic/current/tide references already present in the pack are preserved.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import urllib.parse
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from PIL import Image

from gravnav.datasets.bathymetry_loader import (
    RegionalDemoPackManifest,
    ResolvedRegionalDemoPack,
    load_demo_pack_manifest,
    process_regular_array_bathymetry_grid,
    resolve_regional_demo_pack,
)

EMODNET_WCS_URL = "https://ows.emodnet-bathymetry.eu/wcs"
DEFAULT_COVERAGE_ID = "emodnet__mean_2022"
EMODNET_NATIVE_STEP_DEG = 0.0010416666666666667
EMODNET_ORIGIN_LAT_DEG = 89.99947916666666
EMODNET_ORIGIN_LON_DEG = -70.49947916666666
EMODNET_NODATA_ABS_THRESHOLD = 1.0e20


def _aligned_axis(
    *,
    axis_min_deg: float,
    axis_max_deg: float,
    origin_deg: float,
    step_deg: float,
    size: int,
    descending: bool,
) -> np.ndarray:
    if size <= 0:
        raise ValueError("size must be positive.")
    if descending:
        start_idx = int(
            np.ceil((float(origin_deg) - float(axis_max_deg)) / float(step_deg) - 1.0e-9)
        )
        axis = float(origin_deg) - float(step_deg) * np.arange(start_idx, start_idx + int(size))
    else:
        start_idx = int(
            np.ceil((float(axis_min_deg) - float(origin_deg)) / float(step_deg) - 1.0e-9)
        )
        axis = float(origin_deg) + float(step_deg) * np.arange(start_idx, start_idx + int(size))
    return np.asarray(axis, dtype=np.float64)


def _parse_emodnet_tiff(
    tiff_path: Path,
    *,
    lat_min_deg: float,
    lat_max_deg: float,
    lon_min_deg: float,
    lon_max_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    with Image.open(tiff_path) as img:
        depth_grid = np.asarray(img, dtype=np.float64)
    if depth_grid.ndim != 2:
        raise ValueError(f"Expected 2D TIFF raster, got shape {depth_grid.shape}.")

    invalid_mask = (~np.isfinite(depth_grid)) | (np.abs(depth_grid) >= EMODNET_NODATA_ABS_THRESHOLD)
    num_invalid = int(np.count_nonzero(invalid_mask))
    if num_invalid == depth_grid.size:
        raise ValueError("EMODnet raster contains no finite depth samples in the requested bounds.")

    if num_invalid > 0:
        valid_indices = np.argwhere(~invalid_mask)
        invalid_indices = np.argwhere(invalid_mask)
        for invalid_i, invalid_j in invalid_indices:
            deltas = valid_indices - np.array([invalid_i, invalid_j], dtype=np.int64)
            best_idx = int(np.argmin(np.sum(deltas.astype(np.float64) ** 2, axis=1)))
            src_i, src_j = valid_indices[best_idx]
            depth_grid[invalid_i, invalid_j] = depth_grid[src_i, src_j]

    lat_desc = _aligned_axis(
        axis_min_deg=float(lat_min_deg),
        axis_max_deg=float(lat_max_deg),
        origin_deg=EMODNET_ORIGIN_LAT_DEG,
        step_deg=EMODNET_NATIVE_STEP_DEG,
        size=int(depth_grid.shape[0]),
        descending=True,
    )
    lon_axis = _aligned_axis(
        axis_min_deg=float(lon_min_deg),
        axis_max_deg=float(lon_max_deg),
        origin_deg=EMODNET_ORIGIN_LON_DEG,
        step_deg=EMODNET_NATIVE_STEP_DEG,
        size=int(depth_grid.shape[1]),
        descending=False,
    )
    elevation_grid = -np.flipud(depth_grid)
    lat_axis = lat_desc[::-1].copy()
    return lat_axis, lon_axis, elevation_grid, {
        "num_invalid_points": num_invalid,
        "fill_strategy": "nearest_valid" if num_invalid > 0 else "none",
        "coverage_id": DEFAULT_COVERAGE_ID,
        "native_step_deg": EMODNET_NATIVE_STEP_DEG,
    }


def _coverage_url(
    *,
    coverage_id: str,
    lat_min_deg: float,
    lat_max_deg: float,
    lon_min_deg: float,
    lon_max_deg: float,
) -> str:
    query = urllib.parse.urlencode(
        {
            "SERVICE": "WCS",
            "REQUEST": "GetCoverage",
            "VERSION": "2.0.1",
            "COVERAGEID": str(coverage_id),
            "FORMAT": "image/tiff",
            "SUBSET": [
                f"Lat({float(lat_min_deg):.8f},{float(lat_max_deg):.8f})",
                f"Long({float(lon_min_deg):.8f},{float(lon_max_deg):.8f})",
            ],
        },
        doseq=True,
    )
    return f"{EMODNET_WCS_URL}?{query}"


def _download_emodnet_tiff(
    out_path: Path,
    *,
    coverage_id: str,
    lat_min_deg: float,
    lat_max_deg: float,
    lon_min_deg: float,
    lon_max_deg: float,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = _coverage_url(
        coverage_id=coverage_id,
        lat_min_deg=lat_min_deg,
        lat_max_deg=lat_max_deg,
        lon_min_deg=lon_min_deg,
        lon_max_deg=lon_max_deg,
    )
    with urllib.request.urlopen(url, timeout=180.0) as resp:
        payload = resp.read()
    out_path.write_bytes(payload)
    if payload[:5].lower().startswith(b"<?xml") or payload[:9].lower().startswith(b"<servicee"):
        raise ValueError(f"EMODnet WCS returned XML instead of TIFF for {out_path.name}.")
    return out_path


def _derive_output_paths(
    *,
    region_name: str,
    input_pack_path: Path,
) -> dict[str, Path]:
    safe_region = str(region_name).strip().replace(" ", "_")
    pack_stem = input_pack_path.stem
    if pack_stem.endswith("_demo_pack"):
        output_pack_name = f"{pack_stem[:-10]}_emodnet_demo_pack.json"
    else:
        output_pack_name = f"{pack_stem}_emodnet.json"
    return {
        "raw_tiff": PROJECT_ROOT / "data" / "bathymetry" / "raw" / safe_region / f"{safe_region}_emodnet_mean_2022.tif",
        "processed_npz": PROJECT_ROOT / "data" / "bathymetry" / "processed" / f"{safe_region}_emodnet_bathymetry_grid.npz",
        "manifest": PROJECT_ROOT / "data" / "bathymetry" / "processed" / f"{safe_region}_emodnet_bathymetry_manifest.json",
        "output_pack": input_pack_path.parent / output_pack_name,
    }


def _augment_demo_pack(
    base_manifest: RegionalDemoPackManifest,
    *,
    bathymetry_manifest_path: Path,
    coverage_id: str,
) -> RegionalDemoPackManifest:
    notes = list(base_manifest.notes)
    notes.append(
        f"Bathymetry upgraded to official EMODnet Bathymetry WCS coverage `{coverage_id}`."
    )
    metadata = dict(base_manifest.metadata)
    metadata.update(
        {
            "emodnet_bathymetry": True,
            "emodnet_coverage_id": str(coverage_id),
        }
    )
    return replace(
        base_manifest,
        bathymetry_manifest_path=str(bathymetry_manifest_path.resolve()),
        notes=notes,
        metadata=metadata,
    )


def _prepare_emodnet_bathymetry(
    demo_pack: ResolvedRegionalDemoPack,
    *,
    input_pack_path: Path,
    output_pack_path: Path | None,
    coverage_id: str,
) -> Path:
    paths = _derive_output_paths(
        region_name=demo_pack.manifest.region_name,
        input_pack_path=input_pack_path,
    )
    if output_pack_path is None:
        output_pack_path = paths["output_pack"]

    lat_min, lat_max = demo_pack.bathymetry_grid.lat_bounds_deg
    lon_min, lon_max = demo_pack.bathymetry_grid.lon_bounds_deg
    raw_tiff = _download_emodnet_tiff(
        paths["raw_tiff"],
        coverage_id=coverage_id,
        lat_min_deg=float(lat_min),
        lat_max_deg=float(lat_max),
        lon_min_deg=float(lon_min),
        lon_max_deg=float(lon_max),
    )
    lat_axis, lon_axis, elevation_grid, parse_metadata = _parse_emodnet_tiff(
        raw_tiff,
        lat_min_deg=float(lat_min),
        lat_max_deg=float(lat_max),
        lon_min_deg=float(lon_min),
        lon_max_deg=float(lon_max),
    )
    _, manifest = process_regular_array_bathymetry_grid(
        lat_axis_deg=lat_axis,
        lon_axis_deg=lon_axis,
        elevation_grid_m=elevation_grid,
        raw_data_path=raw_tiff,
        processed_npz_path=paths["processed_npz"],
        manifest_path=paths["manifest"],
        region_name=str(demo_pack.manifest.region_name),
        source_name=f"EMODnet_Bathymetry_{coverage_id}",
        source_kind="emodnet_wcs_tiff",
        reference_surface_height_m=0.0,
        default_method="linear",
        bounds_error=False,
        fill_value_m=np.nan,
        metadata_extra={
            "coverage_id": str(coverage_id),
            "wcs_url": EMODNET_WCS_URL,
            "requested_bounds_deg": {
                "lat": [float(lat_min), float(lat_max)],
                "lon": [float(lon_min), float(lon_max)],
            },
            **parse_metadata,
        },
        project_root=PROJECT_ROOT,
    )
    augmented_manifest = _augment_demo_pack(
        load_demo_pack_manifest(input_pack_path),
        bathymetry_manifest_path=paths["manifest"],
        coverage_id=coverage_id,
    )
    augmented_manifest.write_json(output_pack_path)

    print(f"Prepared EMODnet bathymetry manifest: {manifest.manifest_path}")
    print(f"Prepared EMODnet demo pack: {Path(output_pack_path).expanduser().resolve()}")
    return Path(output_pack_path).expanduser().resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replace the bathymetry surface in a demo pack with official EMODnet bathymetry."
    )
    parser.add_argument(
        "--demo-pack-manifest",
        required=True,
        help="Existing demo-pack manifest whose bathymetry should be replaced.",
    )
    parser.add_argument(
        "--output-demo-pack-manifest",
        default="",
        help="Optional output path for the EMODnet-upgraded demo pack.",
    )
    parser.add_argument(
        "--coverage-id",
        default=DEFAULT_COVERAGE_ID,
        help=f"EMODnet WCS coverage id to use. Default: {DEFAULT_COVERAGE_ID}.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_pack_path = Path(args.demo_pack_manifest).expanduser().resolve()
    demo_pack = resolve_regional_demo_pack(input_pack_path)
    output_pack_path = (
        None
        if not args.output_demo_pack_manifest
        else Path(args.output_demo_pack_manifest).expanduser().resolve()
    )
    _prepare_emodnet_bathymetry(
        demo_pack,
        input_pack_path=input_pack_path,
        output_pack_path=output_pack_path,
        coverage_id=str(args.coverage_id),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
