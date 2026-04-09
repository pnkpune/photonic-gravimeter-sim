# photonic-gravimeter-sim

`photonic-gravimeter-sim` is a gravity-aided navigation simulation repository for GPS-denied missions. It is built as a real navigation stack rather than a notebook demo: WGS84 geodesy and frame mechanics at the bottom, truth and sensor models above that, then INS propagation, gravity map matching, integrity logic, simulation runners, metrics, plots, and saved artifacts.

The project is currently maritime/UUV-first. Gravity is treated as an aiding source inside an inertial navigation system, not as a standalone navigator.

## Current Status

The repo is past the scaffold stage. It now has:

- a working WGS84 / ECEF / NED physics foundation
- truth trajectory generation for maritime/UUV/UAV-style motion
- IMU, scalar gravimeter, depth, velocity-aid, and gravity-gradiometer simulation
- a closed-loop local-level error-state INS with constrained aiding
- particle-filter gravity map matching
- sequence-based gravity map matching
- observability analysis for gravity-information content along a route
- benchmark and reporting scripts
- a validated maritime baseline with saved outputs and comparison against IMU-only

What is validated today:

- the single-scenario maritime baseline run path
- the core INS plus depth and velocity aiding path
- observe-only PF and sequence map-matching diagnostics
- the generated validation report and benchmark artifacts

What is implemented but not yet part of the validated production path:

- directional PF-to-INS feedback
- delayed sequence-to-INS feedback, including a fixed-lag replay path
- closed-loop gravity-feedback policies that actually improve INS metrics over the validated observe-only baseline

## What The Repo Does

The current end-to-end flow is:

1. generate a truth trajectory in geodetic, ECEF, and NED-consistent coordinates
2. sample a synthetic or grid-based gravity disturbance field along that trajectory
3. simulate IMU, scalar gravity, depth, velocity, and optional gradient measurements
4. propagate a local-level error-state INS
5. apply constrained aiding updates
6. run observe-only PF or sequence-based gravity map matching
7. optionally evaluate observability-aware or delayed feedback policies
8. compute navigation, map-matching, and integrity metrics
9. save figures, JSON summaries, NPZ bundles, and markdown reports

The main runnable entry points are:

- [run_single_scenario.py](/Users/pranav/Downloads/photonic-gravimeter-sim/scripts/run_single_scenario.py)
- [benchmark_filters.py](/Users/pranav/Downloads/photonic-gravimeter-sim/scripts/benchmark_filters.py)
- [generate_validation_report.py](/Users/pranav/Downloads/photonic-gravimeter-sim/scripts/generate_validation_report.py)

## Validated Baseline

The current validated baseline is a single-run maritime scenario saved under `data/outputs/validated_maritime_baseline`.

Baseline configuration:

- scenario: `maritime_baseline`
- duration: `2040.0 s`
- sample interval: `2.0 s`
- truth samples: `1021`
- sensor stack: nav-grade IMU + scalar gravimeter + depth aid + velocity aid
- gravity map: synthetic disturbance map
- feedback policy: conservative; PF-to-INS feedback disabled in the validated path

Validated metrics:

| Metric | IMU-only | Aided baseline |
| --- | ---: | ---: |
| INS horizontal RMSE [m] | 13750.213 | 90.318 |
| INS CEP95 [m] | 21579.851 | 169.129 |
| INS vertical RMSE [m] | 2058.759 | 0.387 |
| PF horizontal RMSE [m] | n/a | 128.574 |
| PF CEP95 [m] | n/a | 228.032 |
| Gravimeter RMSE [m/s^2] | n/a | 9.423829e-06 |

Practical interpretation:

- aided INS horizontal RMSE improves by about `152.2x` over IMU-only
- aided INS CEP95 improves by about `127.6x`
- the run path is stable and no longer diverges to kilometer-scale error
- the validated solution today is a conservative aided INS with map matching used as an observe-only diagnostic layer

Primary saved report:

- [validated_maritime_baseline_report.md](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/reports/validated_maritime_baseline_report.md)

Primary saved figures:

- [aided_navigation_overview.png](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/figures/validated_maritime_baseline/aided_navigation_overview.png)
- [aided_position_error_ned.png](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/figures/validated_maritime_baseline/aided_position_error_ned.png)
- [aided_pf_diagnostics.png](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/figures/validated_maritime_baseline/aided_pf_diagnostics.png)
- [imu_only_vs_aided_horizontal_error.png](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/figures/validated_maritime_baseline/imu_only_vs_aided_horizontal_error.png)

## Current Estimator Findings

The repo now has three distinct estimator stories:

1. `INS + constrained aiding`
   This is the validated production baseline.

2. `Observe-only PF map matching`
   Useful as a gravity-aiding diagnostic path and posterior-analysis tool, but not yet trusted for closed-loop feedback.

3. `Observe-only sequence matching`
   Currently the best map-matching estimator in the benchmark set, but still not fed back into the validated INS.

### PF vs Sequence Benchmark

Current benchmark on `maritime_baseline`, seed `42`, with the present synthetic map and safe INS settings:

| Matcher | Gradient | Map-matcher horizontal RMSE [m] | Map-matcher CEP95 [m] |
| --- | --- | ---: | ---: |
| PF | no | 98.943 | 180.590 |
| PF | yes | 96.902 | 178.540 |
| Sequence | no | 99.043 | 179.497 |
| Sequence | yes | 92.808 | 168.130 |

What this means:

- `sequence_plus_gradient` is the best current map-matching estimator in the repo
- the INS baseline remains `103.512 m` RMSE in these benchmark runs because the sequence path is observe-only
- sequence matching is worth keeping as a first-class estimator path even before any closed-loop sequence feedback is solved

### Directional PF Feedback

Directional PF-to-INS feedback is implemented, including:

- posterior eigenstructure extraction
- observability-aware gating
- covariance inflation and persistence logic
- benchmark sweeps and diagnostics

Current outcome:

- the route is informative enough for observability gating, but that is not the bottleneck
- PF-side improvement from scalar-plus-gradient likelihood does not currently translate into better INS metrics
- observability-conditioned directional feedback still does not beat the observe-only gradient baseline on INS RMSE

Conclusion:

- the current problem is PF posterior bias/calibration, not just missing gating logic
- more threshold tuning is not the right next closed-loop solution

### Delayed Sequence Feedback

The repo also contains an experimental delayed sequence-to-INS feedback controller.

Current outcome:

- two architectures now exist:
  - `bias_transfer`: the older current-state transfer path
  - `lag_replay`: a fixed-lag path that applies the delayed estimate at the delayed state and replays forward
- the default conservative settings for both remain a safe no-op on the maritime benchmark
- a relaxed fixed-lag replay benchmark does fire safely in the software sense, but still degrades performance:
  - `93` replayed sequence updates applied
  - INS horizontal RMSE worsened from `103.5 m` to `113.1 m`
  - horizontal HMI rose to about `38.5%`
- earlier relaxed current-state transfer variants were much worse, degrading INS horizontal RMSE to about `214.6 m`, `828.2 m`, and `7.2 km`

Conclusion:

- simple current-state transfer of delayed sequence bias is not a valid production feedback design
- fixed-lag replay is the right architectural direction, but the current replay controller is still not good enough to beat the observe-only baseline
- this entire area is currently useful as a documented negative result and benchmark harness, not as a production feedback path
- the next serious closed-loop sequence design would need replay plus retrodiction/smoother-aware delayed updates rather than direct horizontal pseudo-measurements alone

## System Coverage

The codebase currently covers these areas:

- Physics: WGS84 ellipsoid, normal gravity, ECEF/NED/ENU frames, kinematics, gravity maps, and gravity-reduction helpers
- Sensors: IMU, scalar gravimeter, gradiometer, depth aid, and velocity aid
- Truth: trajectory generation, motion profiles, and built-in scenarios
- Estimation: error-state INS, constrained fusion, PF map matching, sequence matching, integrity, and experimental feedback policies
- Analysis: observability scoring and route-level gravity-information diagnostics
- Simulation: single-run orchestration, Monte Carlo support, metrics, saved results, and benchmark/report scripts
- Plots: navigation and Monte Carlo figure generation for saved runs and comparisons

Built-in scenarios:

- `maritime_baseline`
- `uuv_long_endurance`
- `uav_stress_case`

## Conventions

Repo-wide conventions:

- internal angles: radians
- distance: meters
- velocity: m/s
- acceleration and gravity: m/s^2
- gravity anomaly display: mGal
- Earth-fixed Cartesian frame: ECEF
- local navigation frame: NED
- latitude/longitude inputs: geodetic latitude, east-positive longitude
- timestamps: seconds since scenario start

## How To Run

Single baseline run:

```bash
python3 scripts/run_single_scenario.py \
  --output-dir /tmp/gravnav_single_run \
  --run-id baseline \
  --seed 123 \
  --dt-s 2.0 \
  --pf-particles 64 \
  --map-grid-size 101
```

Generate the saved validation report:

```bash
python3 scripts/generate_validation_report.py
```

Run the estimator comparison benchmark:

```bash
python3 scripts/benchmark_filters.py
```

The default CLI path uses JSON configs under `configs/` and does not require PyYAML.

## Tests

The current automated regression suite covers:

- Earth/geodesy and frame math
- truth/scenario building
- config loading and CLI smoke execution
- INS propagation and constrained fusion
- IMU truth generation
- gradiometer modeling
- PF behavior
- observability logic
- sequence matching

Current status:

- `python3 -m pytest -q`
- `28 passed`

## Reports And Artifacts

Important in-repo documentation and generated outputs:

- [NEXT_STEPS_REPORT.md](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/reports/NEXT_STEPS_REPORT.md)
- [validated_maritime_baseline_report.md](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/reports/validated_maritime_baseline_report.md)

The repo also contains generated run bundles, figures, and comparison reports under:

- [data/outputs/reports](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/reports)
- [data/outputs/figures](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/figures)
- [data/outputs/runs](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/runs)

Most generated output under `data/outputs/` is intentionally ignored by Git; only selected report files are tracked.

## Current Limits

What is not yet validated or still incomplete:

- closed-loop PF-to-INS feedback as part of the production baseline
- closed-loop delayed sequence-to-INS feedback as part of the production baseline
- proof that any current gravity-feedback policy improves INS metrics beyond the validated observe-only baseline
- broader Monte Carlo and scenario sweeps as a standard documented workflow
- fuller plotting coverage for sensor-only diagnostics
- broader estimator and sensor test coverage beyond the current core suite
- YAML config population across the whole repo
- notebook and packaging layers

## Practical Interpretation

Today this repo should be understood as:

- a serious gravity-aided navigation simulator, not a toy notebook
- a validated maritime-style aided INS baseline with strong geodesy and frame conventions
- a useful benchmark platform for comparing scalar gravity, gradient likelihood, PF matching, and sequence matching
- a place where negative results are preserved explicitly instead of being hidden behind retuning

It should not yet be described as:

- a fully validated closed-loop gravity-feedback navigation system
- a finished product for arbitrary routes or platforms
- a proof that the current PF or delayed sequence feedback policies are ready for production use
