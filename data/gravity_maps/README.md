# Gravity Map Data

This repository uses a two-stage gravity-map workflow:

1. raw regional inputs live under `data/gravity_maps/raw/`
2. processed simulator-ready caches live under `data/gravity_maps/processed/`

The simulator itself still consumes `gravnav.physics.gravity_map.GravityGridMap`. The dataset layer only validates raw inputs and converts them into that existing runtime format.

## Current Layout

- `raw/norwegian_margin/`
  - landing zone for a Norwegian-margin regular-grid gravity product
  - includes a small tracked fixture so the regional benchmark path is runnable in tests and local smoke runs
- `processed/`
  - compressed `.npz` caches written by the dataset layer
  - sidecar manifests describing source, bounds, units, spacing, reference height, and interpolation assumptions
- `synthetic/`
  - reserved for saved synthetic map assets when needed

## V1 Raw Format

The first supported raw format is a complete regular-grid CSV table with columns:

- `lat_deg`
- `lon_deg`
- `disturbance_mgal`
- optional `vertical_gradient_mgal_per_m`

Each row is one grid point. The loader reconstructs the rectilinear grid, verifies that the grid is complete and duplicate-free, and writes a `GravityGridMap` cache.

## Processed Cache Artifacts

For the Norwegian-margin path, the default processed outputs are:

- `data/gravity_maps/processed/norwegian_margin_gravity_map.npz`
- `data/gravity_maps/processed/norwegian_margin_gravity_map_manifest.json`

The manifest records:

- source name and source kind
- raw input path
- processed cache path
- latitude/longitude bounds
- grid spacing
- disturbance units
- reference height
- interpolation assumptions

## Important Boundaries

- The bundled Norwegian-margin fixture is a small in-repo regional benchmark input, not a full survey product.
- Replacing that fixture with a public regional gravity grid is a drop-in operation as long as the CSV schema is preserved.
- No higher-order upward/downward continuation is performed here; the runtime map still uses the existing `GravityGridMap` vertical model.
