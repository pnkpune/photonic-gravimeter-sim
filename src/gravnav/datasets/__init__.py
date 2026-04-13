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
from .current_loader import (
    CurrentFieldGrid,
    RegionalCurrentFieldManifest,
    load_current_field_from_manifest,
    load_current_manifest,
    load_regular_csv_current_field,
    process_regular_csv_current_field,
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
from .magnetic_loader import (
    MagneticGrid,
    RegionalMagneticMapManifest,
    load_magnetic_grid_from_manifest,
    load_magnetic_manifest,
    load_regular_csv_magnetic_grid,
    process_regular_csv_magnetic_grid,
)

__all__ = [
    "BathymetryGrid",
    "CurrentFieldGrid",
    "MagneticGrid",
    "RegionalBathymetryManifest",
    "RegionalCurrentFieldManifest",
    "RegionalDemoPackManifest",
    "ResolvedRegionalDemoPack",
    "RegionalGravityMapManifest",
    "RegionalMagneticMapManifest",
    "ensure_regional_bathymetry_grid",
    "ensure_norwegian_margin_map",
    "ensure_regional_gravity_map",
    "load_bathymetry_grid_from_manifest",
    "load_bathymetry_manifest",
    "load_current_field_from_manifest",
    "load_current_manifest",
    "load_demo_pack_manifest",
    "load_magnetic_grid_from_manifest",
    "load_magnetic_manifest",
    "load_regular_csv_bathymetry_grid",
    "load_regular_csv_current_field",
    "load_regular_csv_gravity_map",
    "load_regular_csv_magnetic_grid",
    "load_regional_map_from_manifest",
    "load_regional_manifest",
    "process_regular_csv_bathymetry_grid",
    "process_regular_csv_current_field",
    "process_regular_csv_gravity_map",
    "process_regular_csv_magnetic_grid",
    "resolve_regional_demo_pack",
]
