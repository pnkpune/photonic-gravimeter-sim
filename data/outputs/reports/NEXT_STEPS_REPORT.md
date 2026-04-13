# Next-Steps Report

**Date:** 2026-04-13  
**Branch:** `feature/photonic-digital-twin`  
**Repo state:** calibrated phase-domain photonic gravimeter digital twin integrated, two frozen maritime demo packs rerun, branch diagnosis locked, `58` tests passing

## Current Branch Scope

This branch was deliberately kept narrow:

- replace the older mission-facing photonic wrapper with a physically richer cold-atom Raman digital twin
- calibrate that sensor model against fixed operating regimes
- rerun only the two frozen maritime demo packs:
  - Norwegian margin
  - Helgeland offshore
- use photonic telemetry to determine whether the remaining failure is sensor-limited or region-limited

This branch does **not** add new modalities, new estimator families, or more generic regional tooling.

## What Was Implemented

- phase-domain photonic gravimeter digital twin in `src/gravnav/sensors/photonic_gravimeter.py`
- three locked calibration presets:
  - `photonic_gravimeter_lab_static`
  - `photonic_gravimeter_maritime_benign`
  - `photonic_gravimeter_maritime_rough`
- run-level photonic telemetry summaries
- calibration harness:
  - `scripts/run_photonic_calibration.py`
- compact branch checkpoint report generator:
  - `scripts/generate_photonic_branch_checkpoint.py`
- demo-runner photonic summary export in:
  - `scripts/run_maritime_demo.py`

## Calibration Result

The digital twin now has a defensible calibration surface.

| Preset | Key result | Outcome |
| --- | --- | --- |
| `lab_static` | zero-disturbance error `0`, doubled-`T` scale ratio `3.999918`, gradient and wavefront terms visible | pass |
| `maritime_benign` | valid fraction `0.925`, median contrast `0.660`, vibration suppression `40.0x` | pass |
| `maritime_rough` | vibration suppression `3.18x`, failures dominated by `low_contrast` and `tilt_limit` | stressed but physically coherent |

Important interpretation:

- benign maritime operation is usable
- rough maritime operation is near the edge of viability
- the rough-mode failure pattern is physical, not a software artifact

## Frozen Demo Reruns

### Norwegian Margin

Median across seeds `42/123/777` under the calibrated digital twin:

| Mode | Reported output | Live INS RMSE [m] | Earth-signature RMSE [m] | Live INS CEP95 [m] | Earth-signature CEP95 [m] | HMI horiz |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `live_ins` | `live_ins` | 241.526 | 241.526 | 427.030 | 427.030 | 0.000 |
| `photonic_gravity` | `sequence` | 241.526 | 219.837 | 427.030 | 395.241 | 0.000 |
| `photonic_gravity_bathymetry` | `sequence` | 241.526 | 201.691 | 427.030 | 392.602 | 0.000 |
| `photonic_gravity_bathymetry_lag` | `lag_smoothed` | 241.526 | 205.490 | 427.030 | 393.373 | 0.000 |

Result:

- first-region success survives the physics upgrade
- the best first-region output on this branch is the observe-only `photonic_gravity_bathymetry` sequence path

### Helgeland Offshore

Median across seeds `42/123/777` under the same digital twin:

| Mode | Reported output | Live INS RMSE [m] | Earth-signature RMSE [m] | Live INS CEP95 [m] | Earth-signature CEP95 [m] | HMI horiz |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `live_ins` | `live_ins` | 161.608 | 161.608 | 278.771 | 278.771 | 0.000 |
| `photonic_gravity` | `sequence` | 161.608 | 206.035 | 278.771 | 330.615 | 0.000 |
| `photonic_gravity_bathymetry` | `sequence` | 161.608 | 199.773 | 278.771 | 330.615 | 0.000 |
| `photonic_gravity_bathymetry_lag` | `lag_smoothed` | 161.608 | 195.002 | 278.771 | 311.620 | 0.696 |

Result:

- Helgeland still fails
- the lag output remains non-promotable because RMSE, CEP95, and HMI all fail

## Telemetry Diagnosis

Median photonic telemetry on the promoted lag path:

| Region | Valid fraction | Median contrast | Tilt exceedance fraction | Dominant rejection |
| --- | ---: | ---: | ---: | --- |
| `norwegian_margin` | 0.764 | 0.677 | 0.227 | `tilt_limit` |
| `helgeland_offshore` | 0.764 | 0.677 | 0.227 | `tilt_limit` |

This is the decisive branch result.

Norwegian margin passes under this exact sensor envelope. Helgeland fails under the same sensor envelope. So the remaining bottleneck is **not** primarily the photonic sensor model. It is the regional packaging and Earth-signature distinctiveness story for the second region.

## Branch Decision

The next branch should be:

- `feature/regional-telemetry-hardening`

Why:

- the digital twin is now credible enough for diagnosis
- first-region performance survives the sensor upgrade
- second-region failure is still real
- telemetry shows that Helgeland is failing under the same sensor conditions that still allow Norwegian margin to beat INS

What should **not** happen next:

- do not add magnetic aiding yet
- do not reopen PF-to-INS feedback
- do not retune the photonic model against navigation outcome

## Next Most Impactful Work

### 1. Regional telemetry hardening

Build a region-aware failure analysis layer for the frozen demo packs:

- align navigation error spikes with photonic telemetry degradations
- correlate matcher failure with local gravity texture, bathymetry texture, and route geometry
- quantify where Helgeland loses distinctiveness relative to Norwegian margin

Deliverable:

- one compact two-region diagnostic report that attributes failure segment-by-segment

### 2. Demo-pack hardening, not new modalities

Strengthen the frozen second-region package without changing the estimator family:

- keep the photonic digital twin fixed
- keep gravity and bathymetry only
- improve route packaging and map-window realism using only region information, not navigation outcome

Success criterion:

- second region must beat live INS without nonzero horizontal HMI under the same sensor model

### 3. Only after that, consider the next passive cue

If the two-region gravity-led story becomes stable, then add the next orthogonal passive channel:

- magnetic aiding

But not before the two-region regional story is clean.

## Reproduction Commands

Calibration:

```bash
python3 scripts/run_photonic_calibration.py \
  --output-dir /tmp/gravnav_photonic_calibration
```

Norwegian-margin rerun:

```bash
python3 scripts/run_maritime_demo.py \
  --demo-pack-manifest data/bathymetry/processed/norwegian_margin_maritime_demo_pack.json \
  --output-dir /tmp/gravnav_photonic_nm
```

Helgeland rerun:

```bash
python3 scripts/run_maritime_demo.py \
  --demo-pack-manifest data/bathymetry/processed/helgeland_offshore_demo_pack.json \
  --output-dir /tmp/gravnav_photonic_hel
```

Branch checkpoint:

```bash
python3 scripts/generate_photonic_branch_checkpoint.py \
  --region-one-name norwegian_margin \
  --region-one-summary /tmp/gravnav_photonic_nm/hardware_tied_maritime_demo_summary.json \
  --region-one-photonic-summary /tmp/gravnav_photonic_nm/hardware_tied_maritime_demo_photonic_summary.json \
  --region-two-name helgeland_offshore \
  --region-two-summary /tmp/gravnav_photonic_hel/hardware_tied_maritime_demo_summary.json \
  --region-two-photonic-summary /tmp/gravnav_photonic_hel/hardware_tied_maritime_demo_photonic_summary.json \
  --calibration-summary /tmp/gravnav_photonic_calibration/photonic_calibration_summary.json \
  --output-dir /tmp/gravnav_photonic_checkpoint
```
