# Public Scripps Global Gravity Source

This directory stores the latest public Scripps/Sandwell marine free-air gravity grid
used to extract regional raw inputs for `world_wave1`.

Current source:

- `https://topex.ucsd.edu/pub/global_grav_1min/grav_33.1.nc`
- Indexed on April 18, 2026 as the latest `grav_33.1.nc` release dated January 4, 2026

Regional extraction helper:

```bash
python3 scripts/export_sandwell_gravity_region.py \
  --input-netcdf data/gravity_maps/raw/public_sio_global/grav_33.1.nc \
  --output-csv data/gravity_maps/raw/mid_atlantic_ridge/region_input.csv \
  --lat-min 33 --lat-max 41 \
  --lon-min -38 --lon-max -22 \
  --stride 2
```
