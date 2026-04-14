# photonic-gravimeter-sim

`photonic-gravimeter-sim` is a GPS-denied navigation simulator for passive, stealth-compatible missions. The target product is an INS-centered navigation stack where gravity is the primary Earth-signature anchor and additional passive data are used only to reduce ambiguity without reintroducing GNSS.

The frozen milestone tag is `v0.1-off-grid-maritime-demo`. Current development is on `feature/photonic-digital-twin`.

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
| Helgeland offshore | no validated Earth-signature promotion yet | 161.608 | 161.175 | 278.771 | 278.853 | 0.007 |
| Nordland offshore | no validated Earth-signature promotion yet | 257.644 | 257.552 | 457.781 | 457.781 | 0.010 |

What changed after the EMODnet upgrade:

- Norwegian margin stayed usable, but EMODnet did not outperform the earlier regional bathymetry pack there.
- Helgeland and Nordland stopped being terrain-starved. Raw sequence and lag tracks improved materially in informative windows.
- The remaining blocker is now publication robustness. The runtime-safe selector only harvests a small fraction of the informative weak-region samples, and the evaluated hybrid output still leaks small horizontal HMI in Helgeland and Nordland.

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
- Helgeland and Nordland have useful raw Earth-signature content but are not yet promotable because the evaluated reported output still shows nonzero horizontal HMI
- current-aware correction is not ready for promotion

## Recommended Next Work

The next effective step is not another new modality. It is product-output hardening:

- make the publication layer more locally selective without becoming overconfident
- use the already integrated ambiguity diagnostics, covariance, and modality-specific information ratios more effectively
- keep evaluating against zero-HMI acceptance, not just RMSE gains

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

- `16 passed` for the public-bathymetry and public-multimodal preparation path

Latest public three-region demo artifacts from this work:

- [/private/tmp/gravnav_priority9_emodnet_nm_hybrid2/hardware_tied_maritime_demo_report.md](/private/tmp/gravnav_priority9_emodnet_nm_hybrid2/hardware_tied_maritime_demo_report.md)
- [/private/tmp/gravnav_priority9_emodnet_hel_hybrid2/hardware_tied_maritime_demo_report.md](/private/tmp/gravnav_priority9_emodnet_hel_hybrid2/hardware_tied_maritime_demo_report.md)
- [/private/tmp/gravnav_priority9_emodnet_nord_hybrid2/hardware_tied_maritime_demo_report.md](/private/tmp/gravnav_priority9_emodnet_nord_hybrid2/hardware_tied_maritime_demo_report.md)

The tracked roadmap remains:

- [NEXT_STEPS_REPORT.md](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/reports/NEXT_STEPS_REPORT.md)
