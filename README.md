# photonic-gravimeter-sim

`photonic-gravimeter-sim` is a gravity-aided navigation simulation repository for GPS-denied missions. It is built as a real navigation stack rather than a notebook demo: WGS84 geodesy and frame mechanics at the bottom, truth and sensor models above that, then INS propagation, gravity map matching, integrity logic, simulation runners, metrics, plots, and saved artifacts.

The project is currently maritime/UUV-first. Gravity is treated as an aiding source inside an inertial navigation system, not as a standalone navigator.

## Current Status

The repo is past the scaffold stage. It now has:

- a working WGS84 / ECEF / NED physics foundation
- a regional gravity dataset/cache layer that feeds the existing `GravityGridMap` runtime interface
- truth trajectory generation for maritime/UUV/UAV-style motion
- IMU, scalar gravimeter, depth, velocity-aid, and gravity-gradiometer simulation
- a closed-loop local-level error-state INS with constrained aiding
- particle-filter gravity map matching
- sequence-based gravity map matching
- a bounded-lag sequence smoother that publishes a separate delayed navigation track
- observability analysis for gravity-information content along a route
- benchmark and reporting scripts
- a validated maritime baseline with saved outputs and comparison against IMU-only
- a Norwegian-margin regional benchmark path with a tracked in-repo fixture and processed manifest
- a public-product Norwegian benchmark path built from the NAG-TEC Bouguer anomaly export

What is validated today:

- the single-scenario maritime baseline run path
- the Norwegian-margin ingest/cache path and regional benchmark smoke path
- the explicit-scenario public-product benchmark path and benchmark script path-resolution fix
- the core INS plus depth and velocity aiding path
- observe-only PF and sequence map-matching diagnostics
- the bounded-lag sequence smoother as a delayed navigation output
- the generated validation report, regional benchmark report, and benchmark artifacts

What is implemented but not yet part of the validated production path:

- directional PF-to-INS feedback
- delayed sequence-to-INS feedback, including a fixed-lag replay path
- closed-loop gravity-feedback policies that safely improve the live INS over the validated observe-only baseline

## What The Repo Does

The current end-to-end flow is:

1. generate a truth trajectory in geodetic, ECEF, and NED-consistent coordinates
2. build or load a synthetic or regional grid-based gravity disturbance field
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
- [generate_regional_benchmark_report.py](/Users/pranav/Downloads/photonic-gravimeter-sim/scripts/generate_regional_benchmark_report.py)

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

## Regional Benchmark

The repo now also has a gravity-first Norwegian-margin regional benchmark path.

What this path adds:

- a thin `gravnav.datasets` layer that ingests a regular-grid CSV product and converts it into the existing `GravityGridMap` cache format
- a processed cache plus manifest under `data/gravity_maps/processed/`
- a regional maritime scenario that stays inside the Norwegian-margin map bounds
- a benchmark/report path that compares live INS, observe-only PF, observe-only sequence matching, and the bounded-lag smoother without enabling any closed-loop gravity feedback

Current in-repo regional input:

- raw CSV: `data/gravity_maps/raw/norwegian_margin/norwegian_margin_fixture.csv`
- processed manifest: `data/gravity_maps/processed/norwegian_margin_gravity_map_manifest.json`
- report: [norwegian_margin_regional_benchmark_report.md](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/reports/norwegian_margin_regional_benchmark_report.md)

Current benchmark on `norwegian_margin_maritime`, seed `42`, `dt = 2.0 s`:

| Path | Horizontal RMSE [m] | CEP95 [m] |
| --- | ---: | ---: |
| Live INS | 111.474 | 196.457 |
| Best PF observe-only (`observe_plus_gradient`) | 131.762 | 253.578 |
| Best sequence observe-only (`sequence_plus_gradient`) | 110.219 | 195.754 |
| Bounded-lag smoother (`sequence_plus_gradient_lag_smoothed`) | 110.763 | 196.082 |

Interpretation:

- the estimator ranking survives: sequence matching still beats PF on the regional benchmark
- the bounded-lag smoother still beats the live INS with `0.0%` lag-output horizontal HMI
- the gains shrink sharply versus the synthetic maritime benchmark, so the synthetic map materially overstated gravity distinctiveness relative to the current Norwegian fixture
- this path is now useful as a realism check, but the current in-repo input is still a tracked fixture, not yet a full public survey product

### Public-Product Norwegian Benchmark

The repo now also has a public-product preparation path for the NAG-TEC Bouguer anomaly export (`bouguer_anomaly_geo.xyz`, DOI `10.22008/FK2/AQ38FS`). The ingest step bins the scattered geographic XYZ product into the existing `GravityGridMap` runtime interface, then selects a maritime-style route inside the processed map bounds.

Tracked public-product scenario:

- `norwegian_margin_public_maritime`

Current public-product benchmark route:

- initial latitude: `66.0 deg`
- initial longitude: `14.0 deg`
- initial heading: `45.0 deg`

Default public-product sequence settings were close but did not beat the live INS on that route. A tighter INS-centered observe-only sequence matcher did:

| Path | Horizontal RMSE [m] | CEP95 [m] |
| --- | ---: | ---: |
| Live INS | 104.322 | 187.937 |
| Tuned observe-only sequence | 91.721 | 139.566 |
| Tuned bounded-lag smoother | 98.144 | 172.432 |

Tuned sequence settings for this public-product result:

- sequence window size: `11`
- grid half-span north/east: `100 m`, `100 m`
- grid spacing north/east: `20 m`, `20 m`
- transition std north/east: `15 m`, `15 m`
- center prior std north/east: `50 m`, `50 m`

Interpretation:

- this is the first current public-product result in the repo where sequence map matching beats the live INS on both RMSE and CEP95
- the win is currently on the observe-only sequence estimate, not yet on a validated delayed navigation output
- the tuned bounded-lag smoother also beats the live INS on RMSE and CEP95, but it still shows nonzero lag-output horizontal HMI, so it is not yet promotable as a validated public-product navigation output
- this public-product path is still based on a scattered Bouguer-anomaly proxy binned into a regular grid, so it should be treated as a realistic benchmark step, not yet as a final geophysical truth product

## Current Estimator Findings

The repo now has four distinct estimator stories:

1. `INS + constrained aiding`
   This is the validated production baseline.

2. `Observe-only PF map matching`
   Useful as a gravity-aiding diagnostic path and posterior-analysis tool, but not yet trusted for closed-loop feedback.

3. `Observe-only sequence matching`
   Currently the best map-matching estimator in the benchmark set, but still not fed back into the validated INS.

4. `Bounded-lag sequence smoothing`
   A separate delayed navigation output that uses sequence-plus-gradient estimates without perturbing the live INS.

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

### Bounded-Lag Sequence Smoother

The repo now also has a fixed-lag delayed navigation output driven by `sequence_plus_gradient`.

Current benchmark on `maritime_baseline`, seed `42`:

| Path | Horizontal RMSE [m] | CEP95 [m] | Vertical RMSE [m] | Horizontal HMI |
| --- | ---: | ---: | ---: | ---: |
| Live INS baseline | 103.512 | 189.984 | 0.821 | 0.0% |
| Sequence + gradient observe-only matcher | 92.808 | 168.130 | n/a | n/a |
| Bounded-lag smoothed navigation output | 98.268 | 181.906 | 0.583 | 0.0% |

Acceptance runs completed with the same lag-smoother defaults:

| Scenario | Seed | Live INS RMSE [m] | Lag-smoothed RMSE [m] | Live CEP95 [m] | Lag-smoothed CEP95 [m] | Lag HMI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `maritime_baseline` | 42 | 103.512 | 98.268 | 189.984 | 181.906 | 0.0% |
| `maritime_baseline` | 123 | 85.733 | 80.183 | 153.248 | 150.279 | 0.0% |
| `maritime_baseline` | 777 | 249.184 | 234.013 | 425.268 | 413.821 | 0.0% |
| `uuv_long_endurance` | 42 | 1014.818 | 1005.036 | 2319.284 | 2295.491 | 0.0% |

What this means:

- this is the first delayed navigation path in the repo that consistently improves navigation metrics over the live INS baseline
- it does so without modifying the validated real-time INS path
- it is a first-class delayed-output estimator, not a replacement for the live filter

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

### Closed-Loop Delayed Sequence Feedback

The repo also contains an experimental delayed sequence-to-INS feedback controller.

Current outcome:

- two architectures now exist:
  - `bias_transfer`: the older current-state transfer path
  - `lag_replay`: a fixed-lag path that applies the delayed estimate at the delayed state and replays forward
- the default conservative settings for both remain a safe no-op on the maritime benchmark
- the best current active delayed-feedback result is a directional lag-replay policy:
  - `38` directional replay updates applied
  - INS horizontal RMSE `108.1 m`
  - INS CEP95 `211.5 m`
  - horizontal HMI `0%`
  - still worse than the observe-only baseline at `103.5 m`, so not yet promotable
- a relaxed fixed-lag replay benchmark does fire safely in the software sense, but still degrades performance:
  - `93` replayed sequence updates applied
  - INS horizontal RMSE worsened from `103.5 m` to `113.1 m`
  - horizontal HMI rose to about `38.5%`
- earlier relaxed current-state transfer variants were much worse, degrading INS horizontal RMSE to about `214.6 m`, `828.2 m`, and `7.2 km`

Conclusion:

- simple current-state transfer of delayed sequence bias is not a valid production feedback design
- fixed-lag replay plus directional delayed measurement is the best current closed-loop direction in the repo, but it is still not good enough to beat the observe-only baseline
- this entire area is currently useful as a documented negative result and benchmark harness, not as a production feedback path
- the next serious closed-loop sequence design would need replay plus retrodiction/smoother-aware delayed updates rather than direct horizontal pseudo-measurements alone

## System Coverage

The codebase currently covers these areas:

- Physics: WGS84 ellipsoid, normal gravity, ECEF/NED/ENU frames, kinematics, gravity maps, and gravity-reduction helpers
- Datasets: regional gravity-grid ingestion, processed-cache generation, and manifest sidecars
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

Config-defined regional scenario:

- `norwegian_margin_maritime`
- `norwegian_margin_public_maritime`

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

Run the bounded-lag sequence smoother:

```bash
python3 scripts/run_single_scenario.py \
  --scenario maritime_baseline \
  --seed 42 \
  --run-id sequence_lag \
  --output-dir /tmp/gravnav_sequence_lag \
  --map-matcher sequence \
  --use-gradiometer \
  --use-sequence-lag-smoother
```

Generate the saved validation report:

```bash
python3 scripts/generate_validation_report.py
```

Run the estimator comparison benchmark:

```bash
python3 scripts/benchmark_filters.py
```

Run the Norwegian-margin regional benchmark:

```bash
python3 scripts/benchmark_filters.py \
  --profile regional_core \
  --scenario norwegian_margin_maritime \
  --regional-map norwegian_margin \
  --output-dir data/outputs/reports/norwegian_margin_benchmark \
  --seed 42 \
  --dt-s 2.0 \
  --pf-particles 32
```

Prepare the public NAG-TEC Norwegian benchmark inputs:

```bash
python3 scripts/prepare_public_norwegian_benchmark.py \
  --raw-xyz-path data/gravity_maps/raw/public_nagtec/bouguer_anomaly_geo.xyz
```

Run the current public-product winning sequence configuration:

```bash
python3 scripts/run_single_scenario.py \
  --scenario configs/scenarios/norwegian_margin_public_maritime.json \
  --map-path data/gravity_maps/processed/norwegian_margin_public_bouguer_map.npz \
  --output-dir /tmp/gravnav_public_sequence \
  --run-id public_sequence_tuned \
  --seed 42 \
  --dt-s 2.0 \
  --map-matcher sequence \
  --use-gradiometer \
  --sequence-window-size 11 \
  --sequence-grid-half-span-north-m 100 \
  --sequence-grid-half-span-east-m 100 \
  --sequence-grid-spacing-m 20 20 \
  --sequence-transition-std-m 15 15 \
  --sequence-center-prior-std-m 50 50
```

Generate the Norwegian regional benchmark report:

```bash
python3 scripts/generate_regional_benchmark_report.py
```

The default CLI path uses JSON configs under `configs/` and does not require PyYAML.

## Tests

The current automated regression suite covers:

- Earth/geodesy and frame math
- regional gravity-map ingestion and manifest generation
- truth/scenario building
- config loading and CLI smoke execution
- regional benchmark smoke execution
- INS propagation and constrained fusion
- IMU truth generation
- gradiometer modeling
- PF behavior
- observability logic
- sequence matching

Current status:

- `python3 -m pytest -q`
- `36 passed`

## Reports And Artifacts

Important in-repo documentation and generated outputs:

- [NEXT_STEPS_REPORT.md](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/reports/NEXT_STEPS_REPORT.md)
- [validated_maritime_baseline_report.md](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/reports/validated_maritime_baseline_report.md)
- [norwegian_margin_regional_benchmark_report.md](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/reports/norwegian_margin_regional_benchmark_report.md)

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
- a productized end-to-end benchmark on the full downloaded public Norwegian regional product without relying on a locally prepared clipped/binned cache
- broader Monte Carlo and scenario sweeps as a standard documented workflow
- fuller plotting coverage for sensor-only diagnostics
- broader estimator and sensor test coverage beyond the current core suite
- YAML config population across the whole repo
- notebook and packaging layers

## Practical Interpretation

Today this repo should be understood as:

- a serious gravity-aided navigation simulator, not a toy notebook
- a validated maritime-style aided INS baseline with strong geodesy and frame conventions
- a useful benchmark platform for comparing scalar gravity, gradient likelihood, PF matching, sequence matching, and delayed lag-smoothed outputs
- a regional-benchmark-ready stack that can ingest both tracked fixtures and a public Norwegian-margin product into the existing map interface
- a place where negative results are preserved explicitly instead of being hidden behind retuning

It should not yet be described as:

- a fully validated closed-loop gravity-feedback navigation system
- a finished product for arbitrary routes or platforms
- a fully productized result on a cleaned full public Norwegian-margin survey product
- a proof that the current PF or delayed sequence feedback policies are ready for production use
