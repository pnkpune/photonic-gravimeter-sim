# Norwegian Margin Raw Fixture

This directory is the raw-input landing zone for a Norwegian-margin regional gravity product.

For the repo's default smoke path, it contains a small bundled regular-grid CSV fixture:

- `norwegian_margin_fixture.csv`

CSV schema:

- `lat_deg`
- `lon_deg`
- `disturbance_mgal`
- optional `vertical_gradient_mgal_per_m`

The v1 loader expects a complete rectilinear grid in point-table form. Operators can replace the bundled fixture with a public regional product that preserves the same schema.

The processed cache is written to:

- `data/gravity_maps/processed/norwegian_margin_gravity_map.npz`
- `data/gravity_maps/processed/norwegian_margin_gravity_map_manifest.json`

The bundled fixture is intentionally small and tracked so the real-data-ready code path is runnable in CI and local smoke tests without external downloads.
