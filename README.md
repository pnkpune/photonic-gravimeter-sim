# photonic-gravimeter-sim

`photonic-gravimeter-sim` is a GPS-denied navigation simulator for passive, stealth-compatible missions. The target product is an INS-centered navigation stack where gravity is the primary Earth-signature anchor and additional passive data are used only to reduce ambiguity without reintroducing GNSS.

The frozen milestone tag is `v0.1-off-grid-maritime-demo`. Current development is on `feature/photonic-digital-twin`.

This README is the shortest repo-level answer to four questions:

1. What problem is this simulator trying to solve?
2. What has actually been tried so far?
3. What improved accuracy, what only improved realism, and what failed?
4. What do the current results mean for the next engineering step?

For the full roadmap and historical planning context, see [data/outputs/reports/NEXT_STEPS_REPORT.md](data/outputs/reports/NEXT_STEPS_REPORT.md). This README is the current branch-status narrative.

## Accuracy Summary

Before reading the result tables, the key point is:

- these are **run-level horizontal position-error statistics in meters**
- they are **not** cumulative drift totals summed over days
- `RMSE [m]` means root-mean-square horizontal error across all sampled times in a run
- `CEP95 [m]` means empirical 95th-percentile horizontal error radius across the run
- `HMI` means the fraction of sampled times where the reported solution was hazardously misleading relative to the horizontal alert limit

Mission context for the headline numbers:

- validated aided baseline:
  - scenario: `maritime_baseline`
  - duration: `2040 s` (`34 min`)
  - purpose: prove the simulator and constrained INS aiding stack are numerically stable before gravity-feedback claims
- synthetic Norway benchmark:
  - scenario family: synthetic maritime denied-navigation benchmark
  - purpose: compare INS, PF, sequence, and lag-smoothed estimators under controlled synthetic gravity distinctiveness
- Norwegian-margin regional fixture benchmark:
  - scenario family: maritime regional benchmark on the bundled Norwegian-margin gravity fixture
  - purpose: test whether the synthetic estimator ranking survives on a real ingest/cache path
- frozen realistic Norwegian-margin maritime demo:
  - scenario: `norwegian_margin_maritime`
  - duration: about `4500 s` (`1.25 h`)
  - route pattern: `11` straight `300 s` legs plus `10` coordinated `120 s` turns at `4.0 m/s`
  - purpose: first realistic off-grid maritime/UUV-style passive-navigation demo
- current three-region public branch result:
  - regions: Norwegian margin, Helgeland offshore, Nordland offshore
  - aggregation: medians across seeds `42/123/777`
  - purpose: test whether the same gravity-led passive stack generalizes across public Norway offshore regions while preserving integrity

What has helped accuracy the most so far, ranked by demonstrated impact:

1. `sequence-based gravity matching`
2. `bathymetry / acoustic terrain aiding`
3. `bounded-lag smoothing`
4. `scalar magnetic anomaly aiding`
5. `PF + gradient`
6. `tide / datum correction`

Main validated numbers:

- validated aided baseline vs IMU-only:
  - IMU-only INS RMSE `13750.213 m`
  - validated aided baseline INS RMSE `90.318 m`
- synthetic Norway benchmark:
  - live INS `103.512 m`
  - best PF `96.902 m`
  - best sequence `92.808 m`
  - best lag-smoothed output `98.268 m`
- Norwegian-margin regional fixture benchmark:
  - live INS `111.474 m`
  - best PF `131.762 m`
  - best sequence `110.219 m`
  - best lag-smoothed output `110.763 m`
  - horizontal HMI `0.000`
- frozen realistic Norwegian-margin maritime demo at milestone:
  - live INS `241.526 m`
  - photonic gravity + bathymetry sequence `197.839 m`
  - photonic gravity + bathymetry lag `197.371 m`
  - horizontal HMI `0.000`
- current public three-region branch result:
  - Norwegian margin best validated reported output: `219.124 m` vs live INS `241.526 m`
  - Helgeland and Nordland now have useful raw Earth-signature windows, but the latest strict stateful selector falls back to INS there to keep HMI at `0.000`

What has not helped:

- `current-aware prior` currently degrades all three public regions
- direct feedback / replay / recentering heuristics have not produced a robust win
- the photonic digital twin improved realism and diagnostics, not raw accuracy by itself
- stricter publication logic removed weak-region HMI leaks, but only by falling back to INS

## Core Approach

The current architecture is still deliberately conservative:

- INS performs the high-rate state propagation.
- Gravity disturbance is the mandatory global Earth-signature.
- Gravity history is matched with a sequence estimator rather than pointwise only.
- Bathymetry / acoustic terrain and magnetic anomaly are optional ambiguity-reduction channels.
- The strongest product output today is delayed, not aggressively closed-loop.
- Any output that cannot maintain integrity must fall back to INS.

The estimator family on the current branch is:

- live INS + depth + velocity
- observe-only gravity sequence matcher
- bounded-lag delayed output
- ambiguity-aware publication and fallback logic

No ML path is active on this branch. No GNSS is used in the demo paths.

## Core Equations

These are the main equations a reader needs to understand the implemented physics and estimation logic.

### 1. Gravity disturbance

The scalar gravity map and scalar gravimeter use same-point gravity disturbance:

```text
delta_g(P) = g(P) - gamma(P)
```

where:

- `g(P)` is actual gravity at point `P`
- `gamma(P)` is normal gravity at the same point

The regular-grid map stores disturbance at a reference height `h_ref` and uses a local vertical correction:

```text
delta_g(lat, lon, h)
  = delta_g_ref(lat, lon)
  + (d delta_g / d h) * (h - h_ref)
```

This is implemented in [src/gravnav/physics/gravity_map.py](src/gravnav/physics/gravity_map.py).

### 2. Photonic gravimeter phase model

The photonic path is a phase-domain cold-atom Raman Mach-Zehnder digital twin. The core phase budget is:

```text
Phi
  = Phi_g
  + Phi_vibration
  + Phi_gradient
  + Phi_rotation
  + Phi_wavefront
  + Phi_Zeeman
  + Phi_lightshift
  + Phi_chirp
  + Phi_detection
```

The leading gravity term scales as:

```text
Phi_g ~ k_eff * g * T^2
```

where:

- `k_eff` is the effective Raman wavevector
- `T` is the pulse interrogation time

The measured interferometer fringe is modeled as:

```text
P = P0 + (C / 2) * cos(Phi + phi_bias)
```

where:

- `P` is transition probability
- `P0` is the mid-fringe offset
- `C` is fringe contrast
- `phi_bias` is the operating-point phase bias

The sensor then inverts phase back to a navigation-facing gravity estimate. This is implemented in [src/gravnav/sensors/photonic_gravimeter.py](src/gravnav/sensors/photonic_gravimeter.py).

### 3. Tide and datum correction

The tide layer currently uses a lightweight harmonic correction model:

```text
eta(lat, lon, t) = sum_i A_i(lat, lon) * cos(omega_i t + k_i lon + phi_i)
```

This drives:

```text
h_ref,eff = h_ref,base + eta_surface
g_corr = g_meas - (g_ocean_loading + g_solid_earth)
```

This is implemented in [src/gravnav/physics/tides.py](src/gravnav/physics/tides.py).

### 4. Sequence-based map matching

The core gravity matcher is a sliding-window HMM / Viterbi-style sequence estimator. Over a candidate path `x_1:K` and measurements `z_1:K`, it maximizes:

```text
log p(x_1:K, z_1:K)
  =
  sum_k log p(z_k | x_k)
  + sum_k log p(x_k | x_{k-1})
  + sum_k log p(x_k | x_k^INS)
```

The emission term can include:

- gravity disturbance residual
- gravity gradient residual
- bathymetry depth residual
- bathymetry slope / rugosity residual
- magnetic total-field residual
- magnetic gradient residual

This is implemented in [src/gravnav/estimators/gravity_sequence_match.py](src/gravnav/estimators/gravity_sequence_match.py).

### 5. Bathymetry and magnetic additive likelihoods

The current multi-modal matcher is still additive and gravity-led. In simplified form:

```text
L_total
  =
  L_gravity
  + L_gradient
  + w_b * L_bathymetry
  + w_bg * L_bathymetry_gradient
  + w_br * L_bathymetry_rugosity
  + w_m * L_magnetic
  + w_mg * L_magnetic_gradient
```

The important design choice is that gravity remains mandatory. Other channels reduce ambiguity; they do not replace gravity.

### 6. Protection levels and HMI

The repository uses covariance-derived protection levels rather than claiming certified RAIM-style integrity. For the horizontal plane:

```text
HPL = k_sigma * sqrt(lambda_max(P_NE))
```

where:

- `P_NE` is the 2 x 2 North-East covariance
- `lambda_max` is its largest eigenvalue
- `k_sigma` is the horizontal sigma multiplier

Horizontal hazardously misleading information is flagged in simulation when:

```text
HMI_h = [ horizontal_error > HAL and HPL <= HAL ]
```

This logic is implemented in [src/gravnav/estimators/integrity.py](src/gravnav/estimators/integrity.py).

## What We Have Tried So Far

This section is chronological. It is the quickest way for a new reader to understand what the project has already learned.

### Phase 1: Stabilize the baseline simulator

What was tried:

- repair the truth-to-IMU interface
- enforce interval-consistent propagation
- constrain velocity and depth fusion so they stop destabilizing the INS

Result:

- IMU-only baseline diverged to `13750.213 m` RMSE
- validated aided baseline came down to `90.318 m`

Meaning:

- the simulator itself is numerically sound
- depth and velocity aids are essential local stabilizers
- gravity work after this point was no longer fighting a broken baseline

Primary report:

- [data/outputs/reports/validated_maritime_baseline_report.md](data/outputs/reports/validated_maritime_baseline_report.md)

### Phase 2: PF gravity matching

What was tried:

- observe-only PF gravity matching
- PF with horizontal gradient
- direct PF feedback into the INS

Result:

- observe-only PF improved over live INS in the synthetic benchmark
- direct PF feedback was unstable and tuning-sensitive
- PF never became the best validated architecture

Meaning:

- gravity map matching is useful
- naive closed-loop feedback is dangerous
- observe-only use is defensible; direct injection is not yet

### Phase 3: Sequence matching and bounded-lag smoothing

What was tried:

- replace pointwise-only matching with a sliding-window sequence matcher
- connect consecutive windows with transition and center priors
- produce delayed sequence and lag-smoothed outputs

Result:

- synthetic benchmark:
  - live INS `103.512 m`
  - best PF `96.902 m`
  - best sequence `92.808 m`
  - best lag-smoothed `98.268 m`

Meaning:

- using trajectory history matters
- sequence matching became the main Earth-signature estimator
- lag smoothing helped, but sequence itself was the strongest early gain

Primary report:

- [data/outputs/reports/norwegian_margin_regional_benchmark_report.md](data/outputs/reports/norwegian_margin_regional_benchmark_report.md)

### Phase 4: First regional real-data path

What was tried:

- ingest a Norwegian-margin regional gravity fixture through a real dataset/cache path
- rerun PF, sequence, and lag outputs without changing the estimator family

Result:

- live INS `111.474 m`
- best PF `131.762 m`
- best sequence `110.219 m`
- best lag-smoothed `110.763 m`
- HMI `0.000`

Meaning:

- the synthetic ranking survived on real regional ingest
- gains shrank sharply
- the synthetic environment overstated gravity distinctiveness

### Phase 5: Realistic maritime demo with bathymetry

What was tried:

- maritime/UUV-style public Norway theater
- photonic gravity path
- bathymetry support
- bounded-lag delayed product output

Result at the frozen milestone:

- live INS `241.526 m`
- photonic gravity + bathymetry sequence `197.839 m`
- photonic gravity + bathymetry lag `197.371 m`
- horizontal HMI `0.000`

Meaning:

- this was the first realistic, promotable off-grid demo
- bathymetry became the strongest additive cue
- bounded-lag output became a defensible delayed product path

### Phase 6: Second-region validation

What was tried:

- hold the stack semi-frozen
- validate on Helgeland offshore

Result:

- live INS `161.608 m`
- promoted lag output `199.273 m`
- HMI nonzero
- acceptance `FAIL`

Meaning:

- the Norwegian-margin story did not generalize cleanly
- the project needed either better regional realism or better delayed estimation
- it was not yet justified to claim a robust multi-region product

Primary report:

- [data/outputs/reports/second_region_validation/second_region_validation_report.md](data/outputs/reports/second_region_validation/second_region_validation_report.md)

### Phase 7: Photonic digital twin

What was tried:

- replace the previous photonic wrapper with a phase-domain digital twin
- add warm-up, cadence, contrast loss, vibration correction, systematics, and telemetry
- calibrate lab-static, maritime-benign, and maritime-rough regimes

Result:

- physics realism improved substantially
- diagnostics became much better
- raw navigation accuracy did not jump by itself

Meaning:

- sensor realism was necessary for credibility
- sensor realism alone was not the main remaining accuracy bottleneck
- the weak-region problem shifted from “sensor model too crude” toward “posterior / publication / regional distinctiveness”

### Phase 8: Priority 9 public-data expansion

What was tried:

- tide and datum correction
- public EMODnet bathymetry / acoustic terrain
- scalar magnetic anomaly aiding
- current-aware motion prior

Result:

- EMODnet made Helgeland and Nordland informative instead of terrain-starved
- magnetic anomaly gave a small positive bump
- tide correction improved physical consistency more than raw accuracy
- current-aware prior is still negative

Meaning:

- the repo now has the modality plumbing needed for a serious gravity-led passive stack
- the bottleneck is no longer “missing channels”
- the bottleneck is “how to turn intermittent local wins into a safe promotable output”

### Phase 9: Publication hardening

What was tried:

- ambiguity diagnostics
- adaptive grid
- integrity-envelope fixes
- stricter stateful runtime selector for `lag -> sequence -> INS`

Result:

- Norwegian margin still publishes a validated win
- Helgeland and Nordland no longer leak HMI
- but they now stay on INS fallback under the strict selector

Meaning:

- publication hardening was necessary
- publication hardening alone is now close to exhausted as a source of new gains
- the next likely gain is a stronger delayed-output estimator, not more heuristic switching

## Current Branch State

This branch now contains two major layers on top of the milestone baseline:

- a phase-domain photonic gravimeter digital twin
- Priority 9 Norway-first public-data expansion

The current codebase includes:

- photonic gravity digital twin in [src/gravnav/sensors/photonic_gravimeter.py](src/gravnav/sensors/photonic_gravimeter.py)
- sequence matcher in [src/gravnav/estimators/gravity_sequence_match.py](src/gravnav/estimators/gravity_sequence_match.py)
- runtime integration in [src/gravnav/simulation/runner.py](src/gravnav/simulation/runner.py)
- demo and output-policy logic in [scripts/run_maritime_demo.py](scripts/run_maritime_demo.py)
- tide correction in [src/gravnav/physics/tides.py](src/gravnav/physics/tides.py)
- magnetic and current dataset support in [src/gravnav/datasets/](src/gravnav/datasets/)

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

- tide correction: [src/gravnav/physics/tides.py](src/gravnav/physics/tides.py)
- magnetic grid loader: [src/gravnav/datasets/magnetic_loader.py](src/gravnav/datasets/magnetic_loader.py)
- magnetometer model: [src/gravnav/sensors/magnetometer.py](src/gravnav/sensors/magnetometer.py)
- current grid loader: [src/gravnav/datasets/current_loader.py](src/gravnav/datasets/current_loader.py)
- current-profile sensor: [src/gravnav/sensors/current_profile.py](src/gravnav/sensors/current_profile.py)
- EMODnet bathymetry prep: [scripts/prepare_public_emodnet_bathymetry.py](scripts/prepare_public_emodnet_bathymetry.py)
- multimodal public-pack prep: [scripts/prepare_public_multimodal_norway.py](scripts/prepare_public_multimodal_norway.py)

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

What these numbers mean:

- Norwegian margin is still a real validated win.
- Helgeland and Nordland are no longer devoid of useful Earth-signature information.
- The current problem is that the present delayed-output and publication stack cannot yet turn those local weak-region improvements into a promotable output without falling back to INS.
- That is why the repo is now bottlenecked by delayed estimation quality, not by missing modality plumbing.

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

## Important Reports

If you are new to the repo, read these in order:

1. [README.md](README.md)
2. [data/outputs/reports/NEXT_STEPS_REPORT.md](data/outputs/reports/NEXT_STEPS_REPORT.md)
3. [data/outputs/reports/validated_maritime_baseline_report.md](data/outputs/reports/validated_maritime_baseline_report.md)
4. [data/outputs/reports/norwegian_margin_regional_benchmark_report.md](data/outputs/reports/norwegian_margin_regional_benchmark_report.md)
5. [data/outputs/reports/second_region_validation/second_region_validation_report.md](data/outputs/reports/second_region_validation/second_region_validation_report.md)

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

- [data/outputs/reports/NEXT_STEPS_REPORT.md](data/outputs/reports/NEXT_STEPS_REPORT.md)
