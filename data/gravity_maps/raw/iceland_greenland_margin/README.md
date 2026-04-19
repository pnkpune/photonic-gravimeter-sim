# Iceland-Greenland Margin Raw Gravity Input

Place one supported raw gravity product here before running the external-region bootstrap.

Supported formats:

- regular-grid CSV with `lat_deg`, `lon_deg`, `disturbance_mgal`
- regular-grid XYZ text
- scattered XYZ text

Suggested default filename for the bootstrap script:

- `region_input.xyz`

Example bootstrap:

```bash
python3 scripts/bootstrap_public_region.py \
  --region iceland_greenland_margin \
  --raw-format scattered_xyz \
  --raw-path data/gravity_maps/raw/iceland_greenland_margin/region_input.xyz \
  --dry-run
```
