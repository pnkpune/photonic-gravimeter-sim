# photonic-gravimeter-sim

`photonic-gravimeter-sim` is a gravity-aided navigation simulation repository for GPS-denied missions. It is organized as a real navigation stack, not a notebook demo: WGS84 geodesy and normal gravity at the bottom, truth and sensor models on top, then INS/fusion/map-matching/integrity, then simulation runners, metrics, plots, and reproducible output artifacts.

The repository is currently maritime/UUV-first. Gravity is treated as an aiding source inside an inertial navigation stack, not as a standalone navigation sensor.

## What The Repo Does

The current code path supports this end-to-end flow:

1. build a truth trajectory in geodetic/ECEF/NED-consistent coordinates
2. sample a synthetic or grid-based gravity-disturbance map along that trajectory
3. simulate IMU, scalar gravimeter, depth, and velocity-aid measurements
4. propagate a local-level error-state INS
5. apply constrained aiding updates and either particle-filter or sequence-based gravity map matching
6. optionally analyze local gravity observability, run observe-only sequence matching, and evaluate experimental feedback policies
7. compute navigation and integrity metrics
8. save run archives, JSON summaries, figures, and a markdown report

The main validated runnable entry points are:

- [scripts/run_single_scenario.py](/Users/pranav/Downloads/photonic-gravimeter-sim/scripts/run_single_scenario.py)
- [scripts/generate_validation_report.py](/Users/pranav/Downloads/photonic-gravimeter-sim/scripts/generate_validation_report.py)
- [scripts/benchmark_filters.py](/Users/pranav/Downloads/photonic-gravimeter-sim/scripts/benchmark_filters.py)

## Current Validated Baseline

The repo has a validated single-scenario maritime baseline saved under `data/outputs/`.

Baseline setup:

- scenario: `maritime_baseline`
- duration: `2040.0 s`
- sample interval: `2.0 s`
- truth samples: `1021`
- sensor stack: nav-grade IMU + scalar gravimeter + depth aid + velocity aid
- map matching: particle filter enabled for diagnostics
- PF-to-INS feedback: disabled in the validated baseline

Key metrics from the saved validation bundle:

| Metric | IMU-only | Aided baseline |
| --- | ---: | ---: |
| INS horizontal RMSE [m] | 13750.213 | 90.318 |
| INS CEP95 [m] | 21579.851 | 169.129 |
| INS vertical RMSE [m] | 2058.759 | 0.387 |
| PF horizontal RMSE [m] | n/a | 128.574 |
| PF CEP95 [m] | n/a | 228.032 |
| Gravimeter RMSE [m/s^2] | n/a | 9.423829e-06 |

Important conclusions from the validated run:

- the aided baseline reduces INS horizontal RMSE by about `152.2x` relative to IMU-only
- the aided baseline reduces INS CEP95 by about `127.6x`
- the default single-run simulation is now numerically stable and no longer diverges to kilometer-scale error
- the validated production path uses conservative constrained fusion for velocity and depth aiding
- PF map matching is useful today as a diagnostic/observe-only layer, but closed-loop PF position feedback is not yet part of the validated baseline

Latest repo state beyond the validated baseline:

- the repo now includes an observability analysis layer for gravity-information scoring along a trajectory
- the PF now exports posterior NED eigenvalues/eigenvectors to support geometry-aware feedback decisions
- the runner now supports an observability-aware directional PF-to-INS feedback path
- the repo now also includes a sliding-window Viterbi/HMM-style gravity sequence matcher as an alternative map-matching path
- the repo now also includes an experimental delayed sequence-to-INS feedback path
- neither feedback path is currently claimed as part of the validated baseline

## Priority 4 Outcome

Priority 4 is now implemented as an observe-only alternative estimator path:

- estimator: [gravity_sequence_match.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/estimators/gravity_sequence_match.py)
- runner integration: [runner.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/simulation/runner.py)
- benchmark entry point: [benchmark_filters.py](/Users/pranav/Downloads/photonic-gravimeter-sim/scripts/benchmark_filters.py)

Current maritime baseline benchmark, seed `42`, with the current synthetic map and safe INS settings:

| Matcher | Gradient | Map-matcher horizontal RMSE [m] | Map-matcher CEP95 [m] |
| --- | --- | ---: | ---: |
| PF | no | 98.943 | 180.590 |
| PF | yes | 96.902 | 178.540 |
| Sequence | no | 99.043 | 179.497 |
| Sequence | yes | 92.808 | 168.130 |

Important interpretation:

- the validated INS baseline is unchanged; INS RMSE stays `103.512 m` because the sequence matcher is observe-only today
- sequence matching with gradient is the best current map-matching estimator on this benchmark
- this means Priority 4 is worth keeping as a first-class estimator path before attempting any delayed feedback design

Delayed sequence-feedback status:

- the repo now contains an experimental delayed sequence-to-INS horizontal feedback controller
- the default policy is intentionally conservative and currently fires zero updates on the maritime baseline
- a bounded tuning pass showed that simple current-state bias transfer from delayed sequence estimates is not safe enough yet: active variants degraded INS RMSE from `103.5 m` to roughly `214.6 m`, `828.2 m`, and `7.2 km`
- the current repo conclusion is that this controller is useful as a negative result and benchmark path, not as a production feedback design
- the correct next closed-loop design is therefore not more gain tuning; it is a replay/smoother-aware delayed-update architecture

The full saved report is:

- [validated_maritime_baseline_report.md](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/reports/validated_maritime_baseline_report.md)

## Saved Artifacts

Validation figures:

- [aided_navigation_overview.png](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/figures/validated_maritime_baseline/aided_navigation_overview.png)
- [aided_ground_track_local_ned.png](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/figures/validated_maritime_baseline/aided_ground_track_local_ned.png)
- [aided_position_error_ned.png](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/figures/validated_maritime_baseline/aided_position_error_ned.png)
- [aided_gravimeter_history.png](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/figures/validated_maritime_baseline/aided_gravimeter_history.png)
- [aided_pf_diagnostics.png](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/figures/validated_maritime_baseline/aided_pf_diagnostics.png)
- [imu_only_vs_aided_horizontal_error.png](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/figures/validated_maritime_baseline/imu_only_vs_aided_horizontal_error.png)

Saved run bundles:

- [maritime_baseline_aided.npz](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/runs/validated_maritime_baseline/maritime_baseline_aided.npz)
- [maritime_baseline_aided_summary.json](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/runs/validated_maritime_baseline/maritime_baseline_aided_summary.json)
- [maritime_baseline_aided_metrics.json](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/runs/validated_maritime_baseline/maritime_baseline_aided_metrics.json)
- [maritime_baseline_imu_only.npz](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/runs/validated_maritime_baseline/maritime_baseline_imu_only.npz)
- [maritime_baseline_imu_only_summary.json](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/runs/validated_maritime_baseline/maritime_baseline_imu_only_summary.json)
- [maritime_baseline_imu_only_metrics.json](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/runs/validated_maritime_baseline/maritime_baseline_imu_only_metrics.json)

Supporting roadmap/report artifact:

- [NEXT_STEPS_REPORT.md](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/reports/NEXT_STEPS_REPORT.md)

## Conventions

These conventions are used across the physics and navigation layers:

- angles: radians internally
- distances: meters
- velocity: m/s
- acceleration and gravity: m/s^2
- gravity anomaly display: mGal
- Earth-fixed Cartesian frame: ECEF
- local navigation frame: NED
- latitude/longitude inputs to physics functions: geodetic latitude, east-positive longitude
- timestamps: seconds since scenario start

## Implemented Code

### Physics

- [earth.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/physics/earth.py)
  WGS84 reference ellipsoid, normal gravity, gravity gradients, and geodetic <-> ECEF conversion.
- [frames.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/physics/frames.py)
  ECEF/ENU/NED transforms, Earth-rate and transport-rate helpers, DCM/quaternion utilities, and local-level frame helpers.
- [kinematics.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/physics/kinematics.py)
  Numerical derivatives/integration, navigation scalars, and DCM/quaternion propagation from body rates.
- [gravity_map.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/physics/gravity_map.py)
  Scalar gravity-disturbance maps, interpolation, synthetic anomaly generation, and NPZ persistence.
- [corrections.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/physics/corrections.py)
  Free-air, Bouguer, atmospheric, Eotvos, and moving-base disturbance-recovery helpers.

### Sensors

- [base.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/sensors/base.py)
  Shared sensor abstractions, validation, clipping, stochastic scaling, and low-pass helpers.
- [imu.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/sensors/imu.py)
  IMU truth generation and a stateful IMU model with bias, white noise, and saturation.
- [gravimeter.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/sensors/gravimeter.py)
  Scalar gravimeter physics, moving-base reduction, disturbance handling, and a stateful sensor model.
- [depth.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/sensors/depth.py)
  Depth/height helpers, hydrostatic conversions, and a depth-aiding sensor model.
- [velocity_aid.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/sensors/velocity_aid.py)
  NED/body-frame velocity aiding helpers and a stateful velocity-aid model.

### Truth

- [trajectory.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/truth/trajectory.py)
  Truth trajectory containers and builders from position, velocity, and attitude histories.
- [vehicle_models.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/truth/vehicle_models.py)
  Straight legs, coordinated turns, smooth turns, and profile-to-trajectory conversion.
- [scenarios.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/truth/scenarios.py)
  Declarative scenario specifications, built-in scenarios, and scenario-to-trajectory orchestration.

### Estimators

- [error_state_ins.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/estimators/error_state_ins.py)
  Local-level closed-loop error-state INS propagation and linearized aiding models.
- [fusion.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/estimators/fusion.py)
  Innovation gating, stacked linear updates, constrained depth fusion, constrained velocity aiding, and directional PF pseudo-measurement construction.
- [map_match_pf.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/estimators/map_match_pf.py)
  Particle-filter gravity map matching with INS-prior coupling, geodetic/NED particle-cloud utilities, and posterior eigenstructure extraction for directional feedback.
- [feedback_policy.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/estimators/feedback_policy.py)
  Observability-aware directional PF-to-INS feedback gating, experimental delayed sequence-feedback gating, covariance inflation, persistence logic, and feedback diagnostics.
- [integrity.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/estimators/integrity.py)
  NIS/NEES checks, protection-level calculations, alert-limit evaluation, and history monitoring.
- [gravity_sequence_match.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/estimators/gravity_sequence_match.py)
  Sliding-window discrete candidate-grid sequence matching with forward/backward marginals, Viterbi path extraction, and delayed sequence estimates.

### Analysis, Simulation And Plots

- [observability.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/analysis/observability.py)
  Local gravity-gradient estimation, sliding-window observability Gramian tracking, eigenvalue analysis, information-density scoring, and feedback-recommendation diagnostics.

- [runner.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/simulation/runner.py)
  End-to-end single-scenario orchestration, including observe-only PF mode, observe-only sequence mode, legacy full-3D PF feedback, and the directional-feedback hook.
- [results.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/simulation/results.py)
  Typed result/log containers and JSON/NPZ persistence.
- [metrics.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/simulation/metrics.py)
  Navigation, sensor, and integrity metrics.
- [monte_carlo.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/simulation/monte_carlo.py)
  Monte Carlo study orchestration and aggregate summaries.
- [nav_plots.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/plots/nav_plots.py)
  Navigation result plotting helpers.
- [monte_carlo_plots.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/plots/monte_carlo_plots.py)
  Monte Carlo plotting helpers.

### Utilities

- [config.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/utils/config.py)
  Config-path resolution, YAML/JSON/TOML loading, scenario loading/export, and recursive config merge.
- [rng.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/utils/rng.py)
  Reproducible RNG creation and spawned stream helpers.
- [units.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/utils/units.py)
  Shared unit conversions for gravimetry, IMU datasheet units, angles, and speed.

## Built-In Scenarios

Built into [scenarios.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/truth/scenarios.py):

- `maritime_baseline`
- `uuv_long_endurance`
- `uav_stress_case`

## How To Run

Generate a single baseline run bundle:

```bash
python3 scripts/run_single_scenario.py \
  --output-dir /tmp/gravnav_single_run \
  --run-id baseline \
  --seed 123 \
  --dt-s 2.0 \
  --pf-particles 64 \
  --map-grid-size 101
```

Generate the saved validation figures and report:

```bash
python3 scripts/generate_validation_report.py
```

Run the PF/sequence comparison benchmark:

```bash
python3 scripts/benchmark_filters.py
```

The default CLI path uses JSON configs under `configs/` and does not require PyYAML.

## Tests

The current non-empty regression tests are:

- [conftest.py](/Users/pranav/Downloads/photonic-gravimeter-sim/tests/conftest.py)
- [test_earth.py](/Users/pranav/Downloads/photonic-gravimeter-sim/tests/test_earth.py)
- [test_frames.py](/Users/pranav/Downloads/photonic-gravimeter-sim/tests/test_frames.py)
- [test_truth_models.py](/Users/pranav/Downloads/photonic-gravimeter-sim/tests/test_truth_models.py)
- [test_config.py](/Users/pranav/Downloads/photonic-gravimeter-sim/tests/test_config.py)
- [test_error_state_ins.py](/Users/pranav/Downloads/photonic-gravimeter-sim/tests/test_error_state_ins.py)
- [test_imu.py](/Users/pranav/Downloads/photonic-gravimeter-sim/tests/test_imu.py)
- [test_cli_smoke.py](/Users/pranav/Downloads/photonic-gravimeter-sim/tests/test_cli_smoke.py)
- [test_gravity_gradiometer.py](/Users/pranav/Downloads/photonic-gravimeter-sim/tests/test_gravity_gradiometer.py)
- [test_map_match_pf.py](/Users/pranav/Downloads/photonic-gravimeter-sim/tests/test_map_match_pf.py)
- [test_observability.py](/Users/pranav/Downloads/photonic-gravimeter-sim/tests/test_observability.py)
- [test_sequence_match.py](/Users/pranav/Downloads/photonic-gravimeter-sim/tests/test_sequence_match.py)

The current regression suite completes with `28 passed`.

## Current Limits

What is implemented:

- the WGS84/geodesy/frame foundation
- truth trajectories and motion profiles
- IMU, gravimeter, depth, and velocity-aid sensor models
- local-level INS propagation and constrained aiding fusion
- particle-filter gravity map matching
- sequence-based gravity map matching
- observability analysis and directional-feedback policy infrastructure
- an experimental delayed sequence-feedback path with benchmarked failure envelopes
- integrity monitoring, run logging, metrics, plots, and a reproducible validation report
- a stable single-scenario maritime baseline

What is not yet validated or still scaffold-only:

- closed-loop PF-to-INS position feedback as part of the production baseline
- the new observability-aware directional PF feedback path as a proven improvement over the observe-only baseline
- delayed sequence-to-INS feedback as part of the production baseline
- [sensor_plots.py](/Users/pranav/Downloads/photonic-gravimeter-sim/src/gravnav/plots/sensor_plots.py)
- [run_monte_carlo.py](/Users/pranav/Downloads/photonic-gravimeter-sim/scripts/run_monte_carlo.py)
- [make_synthetic_map.py](/Users/pranav/Downloads/photonic-gravimeter-sim/scripts/make_synthetic_map.py)
- broader sensor-specific and estimator-specific test coverage beyond the current core suite, especially `test_gravimeter.py`
- YAML config population and the notebook layer
- `pyproject.toml` packaging/setup wiring

## Practical Interpretation

Today this repo is best understood as a validated gravity-aided navigation simulation baseline for maritime-style motion, with a strong physics foundation and a stable end-to-end run path. It is already useful for:

- testing inertial-plus-gravity aiding behavior against a controlled truth model
- benchmarking IMU-only versus aided navigation
- benchmarking PF versus sequence-based gravity map matching under scalar and scalar-plus-gradient likelihoods
- generating reproducible figures, metrics, and report artifacts
- experimenting with observability-aware PF feedback policies without changing the validated observe-only baseline
- preserving negative results for delayed sequence feedback instead of hiding them behind retuning
- serving as the base for later Monte Carlo studies and gravity-feedback tuning

It is not yet a finished research product for arbitrary scenarios or a fully validated closed-loop gravity-feedback system.
