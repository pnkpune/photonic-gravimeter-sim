# photonic-gravimeter-sim

`photonic-gravimeter-sim` is a gravity-aided navigation simulation repository built around a real geodesy/navigation foundation instead of a toy Earth model.

The current codebase starts from:

- a WGS84 normal-gravity and geodesy layer
- a consistent ECEF/NED frame and rotation layer
- a concrete gravity-disturbance map and synthetic-map layer
- a gravity-reduction and correction layer for disturbance/anomaly products
- truth-trajectory and vehicle-motion builders
- sensor models for IMU, scalar gravimeter, depth aiding, and velocity aiding
- an initial local-level error-state INS, fusion, map-matching, and integrity layer
- an initial end-to-end simulation, persistence, metrics, and Monte Carlo layer
- an initial navigation and Monte Carlo plotting layer
- one runnable single-scenario script plus baseline JSON configs
- an initial regression-test layer for core geodesy/config/CLI paths
- shared utilities for RNG, units, and config loading

The Python package lives under `src/gravnav`.

## Scope

The repository is being built as a full simulation stack for gravity-aided navigation in GPS-denied settings, with a maritime/UUV-first bias and room for higher-dynamic UAV stress cases.

The guiding idea is:

1. Build the Earth, frame, and unit conventions first.
2. Build truth motion and sensor models on top of those conventions.
3. Add estimators, simulation runners, metrics, and plotting only after the physics base is stable.

## Repository Conventions

These conventions are already reflected in the implemented modules:

- angles: radians internally
- distances: meters
- velocity: m/s
- acceleration and gravity: m/s^2
- gravity anomaly display/reporting: mGal
- Earth-fixed Cartesian frame: ECEF
- local navigation frame: NED
- latitude/longitude inputs to physics functions: geodetic latitude, east-positive longitude
- timestamps: seconds since scenario start

## Implemented Modules

### Physics

- `src/gravnav/physics/earth.py`
  WGS84 reference ellipsoid, normal gravity, gravity gradients, and geodetic <-> ECEF conversion helpers.

- `src/gravnav/physics/frames.py`
  ECEF/ENU/NED frame transforms, Earth-rate and transport-rate helpers, DCM/quaternion utilities, and local-level frame helpers.

- `src/gravnav/physics/kinematics.py`
  Numerical derivatives/integration, velocity-derived navigation scalars, and DCM/quaternion propagation from body rates.

- `src/gravnav/physics/gravity_map.py`
  Regular-grid scalar gravity-disturbance maps, bilinear/nearest interpolation, optional reference-height handling, synthetic anomaly-map generation, and lightweight NPZ persistence helpers.

- `src/gravnav/physics/corrections.py`
  Gravity reduction/correction helpers including atmospheric, free-air, Bouguer, Eotvos, and stationary/moving-base disturbance-recovery workflows.

### Sensors

- `src/gravnav/sensors/base.py`
  Shared sensor abstractions: validation helpers, clipping, stochastic scaling, first-order low-pass helpers, and stateful sensor base classes.

- `src/gravnav/sensors/imu.py`
  IMU truth helpers plus a stateful IMU sensor model with bias, white noise, saturation, and body-frame measurement generation.

- `src/gravnav/sensors/gravimeter.py`
  Scalar gravimeter helpers for disturbance/normal-gravity handling, moving-base reduction, Eotvos terms, and a stateful scalar gravimeter model.

- `src/gravnav/sensors/depth.py`
  Signed-depth helpers, simple hydrostatic conversions, and a stateful scalar depth-aiding sensor model.

- `src/gravnav/sensors/velocity_aid.py`
  Generic NED/body-frame velocity-aid helpers plus a stateful vector velocity-aiding sensor model.

### Truth

- `src/gravnav/truth/trajectory.py`
  Canonical truth-trajectory containers plus builders from position, velocity, and attitude histories.

- `src/gravnav/truth/vehicle_models.py`
  Kinematic motion primitives such as straight legs, coordinated turns, smooth bank-in/bank-out turns, and profile-to-trajectory conversion.

- `src/gravnav/truth/scenarios.py`
  Declarative scenario specifications, segment schemas, named built-in scenarios, and scenario-to-trajectory orchestration.

### Estimators

- `src/gravnav/estimators/error_state_ins.py`
  Local-level closed-loop error-state INS propagation, process-noise handling, and direct linearized aiding models for velocity, position, and depth.

- `src/gravnav/estimators/fusion.py`
  Measurement packaging, innovation gating, stacked linear updates, and convenience wrappers for velocity, position, depth, and later custom aiding measurements.

- `src/gravnav/estimators/map_match_pf.py`
  Position-only particle-filter gravity map matching with gravity/depth likelihoods, INS-prior coupling, resampling, and geodetic/NED particle-cloud utilities.

- `src/gravnav/estimators/integrity.py`
  Integrity and consistency tooling including NIS/NEES checks, chi-square bounds, protection-level calculations, alert-limit evaluation, and time-history monitoring.

### Simulation

- `src/gravnav/simulation/runner.py`
  End-to-end single-scenario orchestration including runner config, aiding schedules, initial covariance setup, truth-to-sensor-to-estimator execution, and logging into scenario results.

- `src/gravnav/simulation/results.py`
  Typed run-result containers, sensor/estimator log containers, extraction helpers, and lightweight JSON/NPZ persistence for later plotting, benchmarking, and Monte Carlo analysis.

- `src/gravnav/simulation/metrics.py`
  Simulation performance metrics and summaries including scalar/vector error metrics, NED position-error summaries, integrity summaries, and top-level scenario metrics derived from run results.

- `src/gravnav/simulation/monte_carlo.py`
  Monte Carlo study orchestration, per-run RNG provenance, aggregate metric summaries, optional parallel execution, config resolution, and study/result archive handling.

### Plots

- `src/gravnav/plots/nav_plots.py`
  Navigation-result plotting helpers for ground track, altitude/depth, NED velocity, yaw-pitch-roll, position error, PF diagnostics, integrity history, and overview figures.

- `src/gravnav/plots/monte_carlo_plots.py`
  Monte Carlo plotting helpers for metric histograms, ECDFs, distribution panels, aggregate comparisons, failure summaries, and study-to-study metric comparisons.

### Scripts

- `scripts/run_single_scenario.py`
  Minimal CLI entry point that loads baseline configs, builds or loads a gravity map, runs one full scenario, and writes result, summary, metrics, map, and effective-config artifacts.

### Configs

- `configs/scenarios/maritime_baseline.json`
  Runnable baseline scenario config mirroring the built-in maritime survey profile.

- `configs/sensors/imu_nav_grade.json`
- `configs/sensors/gravimeter_proto.json`
- `configs/sensors/depth_sensor.json`
- `configs/sensors/velocity_aid.json`
  Baseline JSON sensor configs consumed directly by `scripts/run_single_scenario.py` without requiring PyYAML.

### Tests

- `tests/test_earth.py`
- `tests/test_frames.py`
- `tests/test_truth_models.py`
- `tests/test_config.py`
- `tests/test_cli_smoke.py`
  Initial regression coverage for scalar geodesy/frame helpers, scenario degree-rate parsing, scenario-config fallback behavior, and the single-run CLI smoke path.

### Utilities

- `src/gravnav/utils/rng.py`
  RNG creation, seed/state capture, reproducible generator spawning, and Monte Carlo stream helpers.

- `src/gravnav/utils/units.py`
  Shared unit conversions for angles, gravimetry, IMU-style datasheet units, speed, and ppm/ppb/ppt-style scale factors.

- `src/gravnav/utils/config.py`
  Project-root discovery, config-path resolution, YAML/JSON/TOML loading, scenario loading/export, and recursive config merging.

## Current Status

Implemented now:

- Earth/geodesy foundation
- frames/rotations/local-level math
- concrete gravity-map representation and synthetic-map generation
- gravity reduction and correction workflows
- kinematics helpers
- IMU, gravimeter, depth, and velocity-aid sensor models
- truth trajectories, vehicle profiles, and named scenarios
- initial local-level INS propagation, linearized measurement fusion, PF-based gravity map matching, and integrity monitoring
- end-to-end scenario runner, simulation result/logging containers, persistence helpers, performance metrics, and Monte Carlo orchestration
- navigation and Monte Carlo plotting helpers
- runnable single-scenario script and baseline JSON configs
- initial regression tests for the core run path and known bug fixes
- RNG, units, and config utilities

Still scaffold-only:

- `src/gravnav/plots/sensor_plots.py`
- `scripts/run_monte_carlo.py`
- `scripts/make_synthetic_map.py`
- `scripts/benchmark_filters.py`
- empty tests such as `tests/test_imu.py`, `tests/test_gravimeter.py`, `tests/test_error_state_ins.py`, and `tests/test_map_match_pf.py`
- `configs/monte_carlo/*`
- empty YAML config stubs under `configs/scenarios/*.yaml` and `configs/sensors/*.yaml`
- `notebooks/*`
- `pyproject.toml`

So the repository currently contains the foundational physics, map, correction, truth, sensor, estimator, simulation-execution, core plotting layers, a practical single-run entry point, and a first regression-test layer, but not yet the sensor-specific plotting, broader script surface, or full test/reporting layer.

## Built-In Truth Scenarios

The current truth layer includes named scenario presets in `src/gravnav/truth/scenarios.py`:

- `maritime_baseline`
- `uuv_long_endurance`
- `uav_stress_case`

These are intended as baseline motion libraries for later simulation and estimator work.

## Near-Term Next Steps

The next meaningful layers to implement are:

1. polishing the map-match-to-INS feedback policies and scenario/config wiring
2. sensor-specific plots, Monte Carlo/helper scripts, broader test coverage, and broader config coverage

## Notes

- YAML config support in `src/gravnav/utils/config.py` requires `PyYAML`.
- The default runnable path uses the new JSON configs so it works without PyYAML.
- The repository is still in a foundation-building phase; packaging and test wiring are not finished yet.
