# photonic-gravimeter-sim

`photonic-gravimeter-sim` is a GPS-denied navigation simulator for passive, stealth-compatible missions. The target product is an INS-centered navigation stack where gravity is the primary Earth-signature anchor and additional passive data are used only to reduce ambiguity without reintroducing GNSS.

The frozen milestone tag is `v0.1-off-grid-maritime-demo`. Current development is on `feature/photonic-digital-twin`.

## Accuracy Summary

What has helped accuracy the most so far, ranked by demonstrated impact:

1. `sequence-based gravity matching`
2. `bathymetry / acoustic terrain aiding`
3. `bounded-lag smoothing`
4. `scalar magnetic anomaly aiding`
5. `PF + gradient`
6. `tide / datum correction`

Main validated numbers:

- synthetic Norway benchmark:
  - live INS `103.512 m`
  - best PF `96.902 m`
  - best sequence `92.808 m`
  - best lag-smoothed output `98.268 m`
- frozen realistic Norwegian-margin demo at milestone:
  - live INS `241.526 m`
  - photonic gravity + bathymetry `197.839 m`
  - photonic gravity + bathymetry lag `197.371 m`
  - horizontal HMI `0.000`
- current public three-region branch result:
  - Norwegian margin best validated reported output: `219.124 m` vs live INS `241.526 m`
  - Helgeland and Nordland now have useful raw Earth-signature windows, but the latest strict stateful selector falls back to INS there to keep HMI at `0.000`

What has not helped:

- `current-aware prior` currently degrades all three public regions
- direct feedback / replay / recentering heuristics have not produced a robust win
- the photonic digital twin improved realism and diagnostics, not raw accuracy by itself

## Current Branch State

This branch now contains two major layers on top of the milestone baseline:

- a phase-domain photonic gravimeter digital twin
- Priority 9 Norway-first public-data expansion

The estimator family is still unchanged:

- live INS + depth + velocity
- observe-only gravity sequence matcher
- bounded-lag delayed output
- ambiguity-aware publication and fallback logic

No ML path is active on this branch. No GNSS is used in the demo paths.

## Photonic Digital Twin

The photonic sensor path is no longer a simple noise wrapper. The current `PhotonicGravimeterSpec` / `PhotonicGravimeterSensor` path models:

- Raman Mach-Zehnder phase accumulation with `k_eff T^2`
- sensitivity-function vibration phase and accelerometer-assisted correction
- gravity-gradient, Coriolis / rotation, chirp, Zeeman, Stark, and wavefront terms
- fringe contrast, transition probability, phase inversion, cadence, warm-up, and validity gating
- per-sample telemetry for validity, contrast, residual phase, and rejection reasons

Tracked calibration presets:

- `photonic_gravimeter_lab_static`
- `photonic_gravimeter_maritime_benign`
- `photonic_gravimeter_maritime_rough`

Calibration status remains coherent:

- `lab_static` passes the scale-factor and zero-disturbance checks
- `maritime_benign` stays operational
- `maritime_rough` is degraded for physical reasons, mainly `low_contrast` / `tilt_limit`

## Priority 9 Public-Data Expansion

The Norway-first public-data stack now supports:

- tide and datum correction
- upgraded public bathymetry / acoustic terrain
- scalar magnetic anomaly aiding
- current-aware motion prior

Implemented public-data paths:

- tide correction: [tides.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/physics/tides.py)
- magnetic grid loader: [magnetic_loader.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/datasets/magnetic_loader.py)
- magnetometer model: [magnetometer.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/sensors/magnetometer.py)
- current grid loader: [current_loader.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/datasets/current_loader.py)
- current-profile sensor: [current_profile.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/sensors/current_profile.py)
- EMODnet bathymetry prep: [prepare_public_emodnet_bathymetry.py](/Users/pranav/Downloads/photonic-gravimeter-sim/scripts/prepare_public_emodnet_bathymetry.py)
- multimodal public-pack prep: [prepare_public_multimodal_norway.py](/Users/pranav/Downloads/photonic-gravimeter-sim/scripts/prepare_public_multimodal_norway.py)

Current public sources used:

- gravity: existing Norwegian regional gravity packs already in the repo workflow
- bathymetry: EMODnet Bathymetry 2022 WCS tiles
- magnetic: WMM2025 + WMMHR2025-derived anomaly path
- currents: HYCOM GLBy0.08 point sampling

## Current Three-Region Result

The three frozen Norway public regions are:

- Norwegian margin
- Helgeland offshore
- Nordland offshore

All results below are medians across seeds `42/123/777`.

| Region | Best validated reported output | Live INS RMSE [m] | Reported RMSE [m] | Live INS CEP95 [m] | Reported CEP95 [m] | Horizontal HMI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Norwegian margin | `photonic_gravity_tide_acoustic_magnetic` sequence | 241.526 | 219.124 | 427.030 | 393.852 | 0.000 |
| Helgeland offshore | strict selector falls back to `live_ins` | 161.608 | 161.608 | 278.771 | 278.771 | 0.000 |
| Nordland offshore | strict selector falls back to `live_ins` | 257.644 | 257.644 | 457.781 | 457.781 | 0.000 |

What changed after the EMODnet upgrade:

- Norwegian margin stayed usable, but EMODnet did not outperform the earlier regional bathymetry pack there.
- Helgeland and Nordland stopped being terrain-starved. Raw sequence and lag tracks improved materially in informative windows.
- A stricter stateful selector was then tested. It removed the weak-region HMI leak completely, but only by falling all the way back to INS in Helgeland and Nordland.
- That means publication-policy hardening was necessary, but it is not sufficient to create a promotable weak-region win.

Important negative result:

- the current-aware prior degrades all three regions in the current public setup
- the strongest current branch result is still the tide + acoustic + magnetic path without current-aware promotion

## Current Diagnosis

The main bottleneck is no longer missing sensor physics or missing public-data plumbing.

What the branch now establishes:

- the photonic model is credible enough to interpret navigation results as sensor-driven
- public tide, magnetic, current, and EMODnet bathymetry paths are integrated end to end
- EMODnet creates useful terrain information in the weak regions
- the remaining problem is robust publication of partial Earth-signature wins, not lack of modality plumbing

Current practical diagnosis:

- Norwegian margin is still a clean gravity-led win
- Helgeland and Nordland have useful raw Earth-signature content, but the current safe publication policy can only preserve integrity there by reverting to INS
- current-aware correction is not ready for promotion
- output-policy hardening alone is now close to exhausted as a source of new gains

## Recommended Next Work

The next effective step is not another new modality and not ML. The next effective step is a stronger delayed-output estimator:

- move beyond publication heuristics toward a better delayed-output formulation
- keep the current gravity-led multi-modal stack fixed
- improve the delayed-output path itself rather than only changing which existing path gets published
- likely direction: smoother / factor-graph-style delayed estimation, or another structured delayed estimator that uses the same passive channels but produces a better-calibrated posterior

Only after that is stable should new channels or ML be considered.

## Core Commands

Run the photonic calibration matrix:

```bash
python3 scripts/run_photonic_calibration.py \
  --output-dir /tmp/gravnav_photonic_calibration
```

Prepare a public Norway multimodal demo pack from an existing bathymetry demo pack:

```bash
python3 scripts/prepare_public_multimodal_norway.py \
  --demo-pack-manifest data/bathymetry/processed/helgeland_offshore_demo_pack.json
```

Upgrade a public demo pack to EMODnet bathymetry:

```bash
python3 scripts/prepare_public_emodnet_bathymetry.py \
  --demo-pack-manifest data/bathymetry/processed/helgeland_offshore_priority9_demo_pack.json
```

Run the three public Norway demo packs:

```bash
python3 scripts/run_maritime_demo.py \
  --demo-pack-manifest data/bathymetry/processed/norwegian_margin_maritime_priority9_emodnet_demo_pack.json \
  --output-dir /tmp/gravnav_priority9_emodnet_nm
```

```bash
python3 scripts/run_maritime_demo.py \
  --demo-pack-manifest data/bathymetry/processed/helgeland_offshore_priority9_emodnet_demo_pack.json \
  --output-dir /tmp/gravnav_priority9_emodnet_hel
```

```bash
python3 scripts/run_maritime_demo.py \
  --demo-pack-manifest data/bathymetry/processed/nordland_offshore_priority9_emodnet_demo_pack.json \
  --output-dir /tmp/gravnav_priority9_emodnet_nord
```

Run the targeted regression coverage used for the current branch checkpoint:

```bash
PYTHONPATH=src python3 -m pytest \
  tests/test_bathymetry_loader.py \
  tests/test_prepare_public_emodnet_bathymetry.py \
  tests/test_prepare_public_multimodal_norway.py -q
```

## Validation

Latest targeted regression status on this branch:

- `17 passed` for the public-bathymetry, multimodal preparation, and stateful selector path

Latest public three-region demo artifacts from this work:

- [/private/tmp/gravnav_priority9_emodnet_nm_hybrid3/hardware_tied_maritime_demo_report.md](/private/tmp/gravnav_priority9_emodnet_nm_hybrid3/hardware_tied_maritime_demo_report.md)
- [/private/tmp/gravnav_priority9_emodnet_hel_hybrid3/hardware_tied_maritime_demo_report.md](/private/tmp/gravnav_priority9_emodnet_hel_hybrid3/hardware_tied_maritime_demo_report.md)
- [/private/tmp/gravnav_priority9_emodnet_nord_hybrid3/hardware_tied_maritime_demo_report.md](/private/tmp/gravnav_priority9_emodnet_nord_hybrid3/hardware_tied_maritime_demo_report.md)

The tracked roadmap remains:

- [NEXT_STEPS_REPORT.md](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/reports/NEXT_STEPS_REPORT.md)
