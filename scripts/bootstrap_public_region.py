#!/usr/bin/env python3
"""
Bootstrap one external public region through gravity prep, bathymetry demo-pack prep,
and multimodal augmentation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RECIPE_PATH = PROJECT_ROOT / "configs/ml/public_region_recipes.json"


def _load_recipes(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _resolve_recipe(path: Path, region: str) -> dict[str, Any]:
    recipes = _load_recipes(path)
    if region not in recipes:
        raise KeyError(f"Unknown public region recipe {region!r}.")
    return dict(recipes[region])


def _abs(path_str: str | None) -> str | None:
    if path_str is None:
        return None
    return str((PROJECT_ROOT / str(path_str)).resolve())


def _build_gravity_command(
    *,
    recipe: dict[str, Any],
    raw_format: str,
    raw_path: str,
) -> list[str]:
    gravity = dict(recipe["gravity"])
    cmd = [
        sys.executable,
        str((PROJECT_ROOT / "scripts/prepare_public_gravity_region.py").resolve()),
        "--raw-format",
        str(raw_format),
        "--raw-path",
        str(Path(raw_path).expanduser().resolve()),
        "--region-name",
        str(recipe["region_name"]),
        "--source-name",
        str(gravity["source_name"]),
        "--processed-map-path",
        _abs(str(gravity["processed_map_path"])),
        "--manifest-path",
        _abs(str(gravity["manifest_path"])),
        "--scenario-out",
        _abs(str(gravity["scenario_out"])),
        "--report-out",
        _abs(str(gravity["report_out"])),
        "--template-scenario",
        _abs(str(gravity["template_scenario"])),
        "--search-lat-min",
        str(gravity["search_lat_min"]),
        "--search-lat-max",
        str(gravity["search_lat_max"]),
        "--search-lon-min",
        str(gravity["search_lon_min"]),
        "--search-lon-max",
        str(gravity["search_lon_max"]),
        "--lat-step-deg",
        str(gravity["lat_step_deg"]),
        "--lon-step-deg",
        str(gravity["lon_step_deg"]),
        "--coarse-density-scale",
        str(gravity["coarse_density_scale"]),
        "--final-density-scale",
        str(gravity["final_density_scale"]),
        "--final-margin-deg",
        str(gravity["final_margin_deg"]),
        "--dt-s",
        str(gravity["dt_s"]),
    ]
    cmd.extend(["--headings-deg", *[str(heading) for heading in gravity["headings_deg"]]])
    if gravity.get("crop_lat_min") is not None:
        cmd.extend(["--crop-lat-min", str(gravity["crop_lat_min"])])
    if gravity.get("crop_lat_max") is not None:
        cmd.extend(["--crop-lat-max", str(gravity["crop_lat_max"])])
    if gravity.get("crop_lon_min") is not None:
        cmd.extend(["--crop-lon-min", str(gravity["crop_lon_min"])])
    if gravity.get("crop_lon_max") is not None:
        cmd.extend(["--crop-lon-max", str(gravity["crop_lon_max"])])
    return cmd


def _build_demo_pack_command(
    *,
    recipe: dict[str, Any],
) -> list[str]:
    gravity = dict(recipe["gravity"])
    demo_pack = dict(recipe["demo_pack"])
    return [
        sys.executable,
        str((PROJECT_ROOT / "scripts/prepare_norwegian_maritime_demo.py").resolve()),
        "--scenario",
        _abs(str(gravity["scenario_out"])),
        "--sequence-profile",
        _abs(str(demo_pack["sequence_profile"])),
        "--gravity-manifest",
        _abs(str(gravity["manifest_path"])),
        "--region-name",
        str(recipe["region_name"]),
        "--raw-bathymetry-csv",
        _abs(str(demo_pack["raw_bathymetry_csv"])),
        "--processed-bathymetry",
        _abs(str(demo_pack["processed_bathymetry"])),
        "--bathymetry-manifest",
        _abs(str(demo_pack["bathymetry_manifest"])),
        "--demo-pack-manifest",
        _abs(str(demo_pack["base_demo_pack_manifest"])),
        "--dt-s",
        str(demo_pack["dt_s"]),
        "--margin-deg",
        str(demo_pack["margin_deg"]),
        "--lat-step-deg",
        str(demo_pack["lat_step_deg"]),
        "--lon-step-deg",
        str(demo_pack["lon_step_deg"]),
        "--max-workers",
        str(demo_pack["max_workers"]),
    ]


def _build_multimodal_command(
    *,
    recipe: dict[str, Any],
) -> list[str]:
    demo_pack = dict(recipe["demo_pack"])
    multimodal = dict(recipe["multimodal"])
    cmd = [
        sys.executable,
        str((PROJECT_ROOT / "scripts/prepare_public_multimodal_region.py").resolve()),
        "--demo-pack-manifest",
        _abs(str(demo_pack["base_demo_pack_manifest"])),
        "--output-demo-pack-manifest",
        _abs(str(multimodal["final_demo_pack_manifest"])),
        "--current-lat-step-deg",
        str(multimodal["current_lat_step_deg"]),
        "--current-lon-step-deg",
        str(multimodal["current_lon_step_deg"]),
        "--current-depth-m",
        str(multimodal["current_depth_m"]),
        "--max-workers",
        str(multimodal["max_workers"]),
    ]
    tide_config = _abs(multimodal.get("tide_config"))
    if tide_config is not None:
        cmd.extend(["--tide-config", tide_config])
    return cmd


def _run_command(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap one external public region through the generic prep pipeline."
    )
    parser.add_argument("--region", required=True)
    parser.add_argument(
        "--recipe-path",
        default=str(DEFAULT_RECIPE_PATH),
    )
    parser.add_argument(
        "--stage",
        choices=("gravity", "demo_pack", "multimodal", "all"),
        default="all",
    )
    parser.add_argument(
        "--raw-format",
        choices=("regular_csv", "regular_xyz", "scattered_xyz"),
        default="",
        help="Required when stage includes gravity.",
    )
    parser.add_argument(
        "--raw-path",
        default="",
        help="Required when stage includes gravity. Overrides the suggested raw path in the recipe.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    recipe_path = Path(args.recipe_path).expanduser().resolve()
    recipe = _resolve_recipe(recipe_path, str(args.region))
    stage = str(args.stage)
    raw_path = (
        str(Path(args.raw_path).expanduser().resolve())
        if str(args.raw_path).strip()
        else _abs(str(recipe["suggested_raw_path"]))
    )
    if stage in {"gravity", "all"} and not str(args.raw_format).strip():
        raise SystemExit("--raw-format is required when stage includes gravity.")

    if stage in {"gravity", "all"}:
        _run_command(
            _build_gravity_command(
                recipe=recipe,
                raw_format=str(args.raw_format),
                raw_path=str(raw_path),
            ),
            dry_run=bool(args.dry_run),
        )
    if stage in {"demo_pack", "all"}:
        _run_command(
            _build_demo_pack_command(recipe=recipe),
            dry_run=bool(args.dry_run),
        )
    if stage in {"multimodal", "all"}:
        _run_command(
            _build_multimodal_command(recipe=recipe),
            dry_run=bool(args.dry_run),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
