"""
Dataset loaders and cache helpers for regional gravity products.

The runtime simulator still consumes ``gravnav.physics.gravity_map.GravityGridMap``.
This package only handles ingestion, validation, caching, and lightweight
metadata for user-provided regional products.
"""

from .bathymetry_loader import (
    BathymetryGrid,
    RegionalBathymetryManifest,
    RegionalDemoPackManifest,
    ResolvedRegionalDemoPack,
    ensure_regional_bathymetry_grid,
    load_bathymetry_grid_from_manifest,
    load_bathymetry_manifest,
    load_demo_pack_manifest,
    load_regular_csv_bathymetry_grid,
    process_regular_csv_bathymetry_grid,
    resolve_regional_demo_pack,
)
from .gravity_loader import (
    RegionalGravityMapManifest,
    ensure_norwegian_margin_map,
    ensure_regional_gravity_map,
    load_regular_csv_gravity_map,
    load_regional_map_from_manifest,
    load_regional_manifest,
    process_regular_csv_gravity_map,
)

__all__ = [
    "BathymetryGrid",
    "RegionalBathymetryManifest",
    "RegionalDemoPackManifest",
    "ResolvedRegionalDemoPack",
    "RegionalGravityMapManifest",
    "ensure_regional_bathymetry_grid",
    "ensure_norwegian_margin_map",
    "ensure_regional_gravity_map",
    "load_bathymetry_grid_from_manifest",
    "load_bathymetry_manifest",
    "load_demo_pack_manifest",
    "load_regular_csv_bathymetry_grid",
    "load_regular_csv_gravity_map",
    "load_regional_map_from_manifest",
    "load_regional_manifest",
    "process_regular_csv_bathymetry_grid",
    "process_regular_csv_gravity_map",
    "resolve_regional_demo_pack",
]
