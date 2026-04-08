# Validated Maritime Baseline Report

## Run Scope

- Scenario: `maritime_baseline`
- Sample interval: `2.0 s`
- Baseline mode: nav-grade IMU + gravimeter + depth aid + velocity aid + PF observe-only
- PF feedback status: disabled in the validated baseline because closed-loop PF position injection remains tuning-sensitive

## Simulation Data

- Aided run archive: `data/outputs/runs/validated_maritime_baseline/maritime_baseline_aided.npz`
- Aided summary JSON: `data/outputs/runs/validated_maritime_baseline/maritime_baseline_aided_summary.json`
- Aided metrics JSON: `data/outputs/runs/validated_maritime_baseline/maritime_baseline_aided_metrics.json`
- IMU-only run archive: `data/outputs/runs/validated_maritime_baseline/maritime_baseline_imu_only.npz`
- IMU-only summary JSON: `data/outputs/runs/validated_maritime_baseline/maritime_baseline_imu_only_summary.json`
- IMU-only metrics JSON: `data/outputs/runs/validated_maritime_baseline/maritime_baseline_imu_only_metrics.json`

## Configuration Summary

- Truth duration: `2040.0 s` with `1021` truth samples
- IMU config: `{"accel_bias_random_walk_mps2_per_sqrt_s": [2.5e-06, 2.5e-06, 2.5e-06], "accel_noise_density_mps2_per_sqrt_hz": [0.00025, 0.00025, 0.00025], "accel_turn_on_bias_std_mps2": [0.00015, 0.00015, 0.00015], "gyro_bias_random_walk_radps_per_sqrt_s": [5e-07, 5e-07, 5e-07], "gyro_noise_density_radps_per_sqrt_hz": [8e-05, 8e-05, 8e-05], "gyro_turn_on_bias_std_radps": [2e-05, 2e-05, 2e-05], "name": "imu_nav_grade"}`
- Gravimeter config: `{"bandwidth_hz": 0.5, "bias_random_walk_mps2_per_sqrt_s": 2.5e-07, "fixed_bias_mps2": 0.0, "name": "gravimeter_proto", "noise_density_mps2_per_sqrt_hz": 1e-05, "scale_factor_error_ppm": 0.0, "supports_absolute_mode": false, "turn_on_bias_std_mps2": 5e-06}`
- Depth config: `{"bandwidth_hz": 2.0, "bias_random_walk_m_per_sqrt_s": 0.001, "fixed_bias_m": 0.0, "fluid_density_kgpm3": 1025.0, "gravity_mps2": 9.80665, "name": "depth_sensor", "noise_density_m_per_sqrt_hz": 0.02, "reference_pressure_pa": 101325.0, "reference_surface_height_m": 0.0, "scale_factor_error_ppm": 0.0, "turn_on_bias_std_m": 0.05}`
- Velocity-aid config: `{"bandwidth_hz": 2.0, "bias_random_walk_mps_per_sqrt_s": [0.002, 0.002, 0.002], "fixed_bias_mps": [0.0, 0.0, 0.0], "name": "velocity_aid", "noise_density_mps_per_sqrt_hz": [0.03, 0.03, 0.03], "scale_factor_error_ppm": [0.0, 0.0, 0.0], "turn_on_bias_std_mps": [0.02, 0.02, 0.02]}`
- Safe fusion policies in validated run:
  - interval-consistent IMU truth for propagation
  - velocity-aid updates constrained to the velocity state
  - depth updates constrained to the height state
  - PF enabled for diagnostics only, not for INS position feedback

## Key Metrics

| Metric | IMU-only | Validated aided baseline |
| --- | ---: | ---: |
| INS horizontal RMSE [m] | 13750.213 | 90.318 |
| INS CEP95 [m] | 21579.851 | 169.129 |
| INS vertical RMSE [m] | 2058.759 | 0.387 |
| Gravimeter RMSE [m/s²] | n/a | 9.423829e-06 |
| PF horizontal RMSE [m] | n/a | 128.574 |
| PF CEP95 [m] | n/a | 228.032 |

## Highlights

- The validated aided baseline reduces INS horizontal RMSE by `152.2x` relative to IMU-only.
- The validated aided baseline reduces INS CEP95 by `127.6x` relative to IMU-only.
- The default CLI baseline now runs stably at `90.3 m` horizontal RMSE instead of diverging to kilometer-scale error.
- The main simulation bug was the truth-to-IMU interface: the runner needed interval-consistent IMU truth rather than sample-centered kinematics.
- The main estimator instability was cross-covariance-driven overcorrection from scalar depth and vector velocity aids into bias states. The validated baseline now uses conservative constrained fusion paths.
- PF map matching is producing diagnostics in the baseline run, but PF position feedback is not yet part of the validated closed-loop navigation solution.

## Figures

- Navigation overview: `data/outputs/figures/validated_maritime_baseline/aided_navigation_overview.png`
- Local ground track: `data/outputs/figures/validated_maritime_baseline/aided_ground_track_local_ned.png`
- INS position error history: `data/outputs/figures/validated_maritime_baseline/aided_position_error_ned.png`
- Gravimeter disturbance history: `data/outputs/figures/validated_maritime_baseline/aided_gravimeter_history.png`
- PF diagnostics: `data/outputs/figures/validated_maritime_baseline/aided_pf_diagnostics.png`
- IMU-only vs aided horizontal error: `data/outputs/figures/validated_maritime_baseline/imu_only_vs_aided_horizontal_error.png`

## Conclusion

The simulation stack is now running correctly for the validated single-scenario baseline. The truth, IMU, fusion, and runner paths are numerically consistent, the default CLI run is stable, and the aided baseline is materially better than IMU-only. The remaining non-validated item is closed-loop PF position feedback: the gravity map-matching layer is usable in observe-only mode for diagnostics, but its direct feedback into the INS still needs separate tuning before it should be treated as part of the production baseline.

The practical next step is not more bug fixing in the physics stack. It is controlled development of a validated gravity-feedback policy, ideally with explicit acceptance gates, covariance inflation rules, and scenario-by-scenario evaluation against the now-stable aided baseline.
