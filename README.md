# photonic-gravimeter-sim

`photonic-gravimeter-sim` is a GPS-denied navigation simulator for passive, stealth-compatible missions. The product target is not a standalone gravimeter. It is an INS-centered navigation stack where gravity is the primary Earth-signature anchor and other passive channels are allowed only to reduce ambiguity without reintroducing GNSS.

The current milestone tag is `v0.1-off-grid-maritime-demo`. The current working branch is `feature/photonic-digital-twin`.

## Branch Scope

This branch is intentionally narrow:

- keep the existing estimator family unchanged
- replace the older mission-facing photonic wrapper with a phase-domain cold-atom digital twin
- calibrate that sensor model against fixed operating regimes
- rerun only the two frozen maritime demo packs:
  - Norwegian margin
  - Helgeland offshore
- use telemetry to decide whether the remaining failure is sensor-limited or region-limited

This branch does not add magnetic aiding, new filter families, or more generic regional tooling.

## Current Result

The digital twin is now calibrated and integrated behind the existing public sensor interface:

- `PhotonicGravimeterSpec`
- `PhotonicGravimeterMeasurement`
- `PhotonicGravimeterSensor`

It explicitly models:

- Raman Mach-Zehnder phase accumulation with `k_eff T^2`
- sensitivity-function vibration phase and accelerometer-assisted compensation
- gravity-gradient, Coriolis / rotation, chirp, Zeeman, Stark, and wavefront terms
- fringe contrast, transition probability, phase inversion, cadence, warm-up, and validity gating
- per-sample telemetry for branch diagnosis

## Calibration Checkpoint

Three locked presets ship with the branch:

- `photonic_gravimeter_lab_static`
- `photonic_gravimeter_maritime_benign`
- `photonic_gravimeter_maritime_rough`

Calibration outcome from `scripts/run_photonic_calibration.py`:

| Preset | Key result | Outcome |
| --- | --- | --- |
| `lab_static` | zero-disturbance error `0`, doubled-`T` scale ratio `3.999918`, gradient and wavefront terms visible | pass |
| `maritime_benign` | valid fraction `0.925`, median contrast `0.660`, vibration suppression `40.0x` | pass |
| `maritime_rough` | vibration suppression `3.18x`, failures dominated by `low_contrast` and `tilt_limit` | pass as stressed regime |

Important interpretation:

- the benign preset is operational
- the rough preset is intentionally near the edge of usability
- the rough regime is degraded for physical reasons, not numerical failure

## Frozen Demo Reruns

### Norwegian Margin

Median across seeds `42/123/777` under the calibrated digital twin:

| Mode | Reported output | Live INS RMSE [m] | Earth-signature RMSE [m] | Live INS CEP95 [m] | Earth-signature CEP95 [m] | HMI horiz |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `live_ins` | `live_ins` | 241.526 | 241.526 | 427.030 | 427.030 | 0.000 |
| `surrogate_gravity` | `sequence` | 241.526 | 219.616 | 427.030 | 391.954 | 0.000 |
| `photonic_gravity` | `sequence` | 241.526 | 219.837 | 427.030 | 395.241 | 0.000 |
| `photonic_gravity_bathymetry` | `sequence` | 241.526 | 201.691 | 427.030 | 392.602 | 0.000 |
| `photonic_gravity_bathymetry_lag` | `lag_smoothed` | 241.526 | 205.490 | 427.030 | 393.373 | 0.000 |

Takeaway:

- the digital twin keeps the first-region win intact
- the best first-region output on this branch is the observe-only `photonic_gravity_bathymetry` sequence estimate
- the lag output is still safe here, but it is no longer the best median performer

### Helgeland Offshore

Median across seeds `42/123/777` under the same calibrated digital twin:

| Mode | Reported output | Live INS RMSE [m] | Earth-signature RMSE [m] | Live INS CEP95 [m] | Earth-signature CEP95 [m] | HMI horiz |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `live_ins` | `live_ins` | 161.608 | 161.608 | 278.771 | 278.771 | 0.000 |
| `surrogate_gravity` | `sequence` | 161.608 | 209.473 | 278.771 | 331.689 | 0.000 |
| `photonic_gravity` | `sequence` | 161.608 | 206.035 | 278.771 | 330.615 | 0.000 |
| `photonic_gravity_bathymetry` | `sequence` | 161.608 | 199.773 | 278.771 | 330.615 | 0.000 |
| `photonic_gravity_bathymetry_lag` | `lag_smoothed` | 161.608 | 195.002 | 278.771 | 311.620 | 0.696 |

Takeaway:

- Helgeland still fails
- the lag output is not promotable because RMSE, CEP95, and HMI all fail the acceptance gate
- adding another modality now would blur the diagnosis

## Telemetry Diagnosis

The new branch checkpoint compares photonic telemetry across the two frozen regions.

Median photonic telemetry on the promoted `photonic_gravity_bathymetry_lag` path:

| Region | Valid fraction | Median contrast | Tilt exceedance fraction | Dominant rejection |
| --- | ---: | ---: | ---: | --- |
| `norwegian_margin` | 0.764 | 0.677 | 0.227 | `tilt_limit` |
| `helgeland_offshore` | 0.764 | 0.677 | 0.227 | `tilt_limit` |

That matters because:

- Norwegian margin passes under this exact sensor envelope
- Helgeland fails under the same sensor envelope
- the branch diagnosis is therefore `region_limited`, not `sensor_limited`

Current next-branch decision:

- `feature/regional-telemetry-hardening`

## What This Branch Established

- the photonic model is now physically credible enough to interpret navigation results as sensor-driven rather than wrapper-driven
- first-region performance survives the physics upgrade
- second-region failure is still real
- the remaining bottleneck is regional distinctiveness and telemetry-aware demo-pack hardening, not a missing photonic phase term

## What Not To Do Next

Do not do these on top of this branch result:

- add magnetic aiding yet
- reopen PF-to-INS feedback
- retune the digital twin against navigation outcome
- claim a generalized multi-region off-grid result

## Core Commands

Run the photonic calibration matrix:

```bash
python3 scripts/run_photonic_calibration.py \
  --output-dir /tmp/gravnav_photonic_calibration
```

Run the Norwegian-margin frozen demo pack:

```bash
python3 scripts/run_maritime_demo.py \
  --demo-pack-manifest data/bathymetry/processed/norwegian_margin_maritime_demo_pack.json \
  --output-dir /tmp/gravnav_photonic_nm
```

Run the Helgeland frozen demo pack:

```bash
python3 scripts/run_maritime_demo.py \
  --demo-pack-manifest data/bathymetry/processed/helgeland_offshore_demo_pack.json \
  --output-dir /tmp/gravnav_photonic_hel
```

Generate the branch checkpoint report:

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

Run the full regression suite:

```bash
PYTHONPATH=src python3 -m pytest -q
```

## Validation

Current regression status:

- `58 passed`

Primary branch artifacts are generated, not tracked:

- calibration summary and report
- frozen-region demo summaries and reports
- photonic branch checkpoint summary and report

The tracked source of truth for branch intent is:

- [data/outputs/reports/NEXT_STEPS_REPORT.md](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/reports/NEXT_STEPS_REPORT.md)
