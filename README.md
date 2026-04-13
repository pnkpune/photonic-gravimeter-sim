# photonic-gravimeter-sim

`photonic-gravimeter-sim` is a GPS-denied navigation simulation stack for passive, stealth-compatible missions. The design target is not a standalone gravimeter. It is an INS-centered navigation subsystem where gravity is the primary geophysical anchor and other passive data channels are allowed to reduce ambiguity without relying on GNSS.

The repo is maritime/UUV-first. The current best result is a realistic Norwegian-margin demo where a bounded-lag Earth-signature output built from a mission-mode photonic gravimeter, gravity gradient likelihood, and bathymetry beats the INS-only baseline across all acceptance seeds with zero horizontal HMI. That result is still region-specific: the first frozen second-region validation on Helgeland offshore failed, so the repo does not yet claim a two-region validated passive stack.

## Current Status

The stack is well past scaffolding. It now includes:

- WGS84 geodesy, frame transforms, and local-level kinematics
- trajectory and scenario generation
- IMU, scalar gravimeter, photonic gravimeter, gradiometer, depth, velocity-aid, and bathymetry sensor models
- local-level error-state INS propagation with constrained depth and velocity aiding
- particle-filter and sequence-based gravity map matching
- bounded-lag sequence smoothing as a separate delayed navigation output
- observability analysis, metrics, plots, reports, and regional dataset loaders
- Norwegian regional gravity and bathymetry demo-pack support
- second-region validation workflow for Helgeland offshore using the same estimator family

What is recommended today:

- keep the live INS as the real-time navigation backbone
- use gravity-led sequence matching as the global Earth-signature estimator
- use the bounded-lag Earth-signature output as the promoted passive correction product when it retains zero HMI
- treat the current promoted result as a strong single-region demo, not yet a generalized multi-region claim

What is not promoted:

- PF-to-INS closed-loop gravity feedback
- delayed sequence-to-INS replay feedback
- any path that silently treats a delayed geophysical estimate as an instantaneous live-state correction

## Headline Result

The promoted realistic demo is:

- theater: Norwegian margin
- scenario: `norwegian_margin_maritime`
- mission style: 1.25-hour off-grid multi-leg maritime survey
- speed: `4.0 m/s`
- coordinated-turn bank angle: `8 deg`
- sample period: `2.0 s`
- deterministic initial INS position offset: `[60.0, -30.0, 0.0] m` in local NED
- seeds: `42`, `123`, `777`

The promoted sensor and estimator stack is:

- live INS with depth and velocity aiding
- mission-mode photonic gravimeter
- gravity gradiometer likelihood
- bathymetry as supporting passive context
- observe-only sequence matcher
- bounded-lag sequence smoother as the reported Earth-signature output

Frozen demo sequence profile:

- `window_size = 11`
- `grid_half_span_m = [120.0, 120.0]`
- `grid_spacing_m = [20.0, 20.0]`
- `transition_std_m = [20.0, 20.0]`
- `center_prior_std_m = [80.0, 80.0]`
- `bathymetry_meas_std_m = 2.0`
- `bathymetry_weight = 3.5`
- `map_match_every_steps = 1`

Mission-mode photonic gravimeter defaults used in the promoted demo:

- `update_period_s = 1.0`
- `warmup_time_s = 45.0`
- `bandwidth_hz = 0.4`
- `noise_density = 4.0e-06 m/s^2/sqrt(Hz)`
- `turn_on_bias_std = 1.0e-06 m/s^2`
- motion-coupling residuals retained explicitly rather than assuming an ideal quiet-lab sensor

Supporting passive bathymetry defaults:

- measurement noise std: `2.0 m`
- turn-on bias std: `0.5 m`

## Promoted Demo Result

Median across seeds:

| Mode | Reported Earth-signature output | Live INS RMSE [m] | Earth-signature RMSE [m] | Live INS CEP95 [m] | Earth-signature CEP95 [m] | Horizontal HMI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `live_ins` | `live_ins` | 241.526 | 241.526 | 427.030 | 427.030 | 0.000 |
| `surrogate_gravity` | `sequence` | 241.526 | 219.616 | 427.030 | 391.954 | 0.000 |
| `photonic_gravity` | `sequence` | 241.526 | 216.118 | 427.030 | 388.292 | 0.000 |
| `photonic_gravity_bathymetry` | `sequence` | 241.526 | 197.839 | 427.030 | 375.991 | 0.000 |
| `photonic_gravity_bathymetry_lag` | `lag_smoothed` | 241.526 | 197.371 | 427.030 | 374.722 | 0.000 |

Per-seed result for the promoted output (`photonic_gravity_bathymetry_lag`):

| Seed | Live INS RMSE [m] | Lag-smoothed RMSE [m] | Improvement [m] | Horizontal HMI |
| --- | ---: | ---: | ---: | ---: |
| `42` | 241.526 | 197.371 | 44.155 | 0.000 |
| `123` | 135.255 | 133.763 | 1.492 | 0.000 |
| `777` | 364.901 | 306.314 | 58.587 | 0.000 |

This is the first robust in-repo result that satisfies the practical product bar:

- it beats INS-only across all acceptance seeds
- it uses realistic mission-style motion rather than a short local toy route
- it keeps GNSS out of the loop
- it preserves the live INS as the real-time backbone
- it reports the geophysical improvement as a bounded-lag passive navigation output, which is operationally credible

## Second-Region Validation

The second-region validation branch kept the stack semi-frozen:

- live INS + depth + velocity
- mission-mode photonic gravimeter
- gravity gradiometer likelihood
- bathymetry as supporting passive context
- observe-only sequence matcher
- bounded-lag Earth-signature output

The first region reran unchanged through the new demo-pack path, but the same architecture did not generalize to Helgeland offshore under the locked public-offshore profile.

Locked public-offshore profile:

- `window_size = 11`
- `grid_half_span_m = [100.0, 100.0]`
- `grid_spacing_m = [20.0, 20.0]`
- `transition_std_m = [15.0, 15.0]`
- `center_prior_std_m = [50.0, 50.0]`
- `bathymetry_meas_std_m = 2.0`
- `bathymetry_weight = 3.5`
- `map_match_every_steps = 1`

Two-region validation outcome:

| Region | Frozen profile | Live INS RMSE [m] | Promoted RMSE [m] | Live INS CEP95 [m] | Promoted CEP95 [m] | HMI zero all seeds | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `norwegian_margin` | `norwegian_margin_maritime_demo` | 241.526 | 197.371 | 427.030 | 374.722 | `True` | `PASS` |
| `helgeland_offshore` | `public_offshore_locked` | 161.608 | 199.273 | 278.771 | 331.707 | `False` | `FAIL` |

Why Helgeland failed:

- even `photonic_gravity` alone was worse than live INS on median RMSE: `209.549 m` vs `161.608 m`
- the promoted lag output had nonzero horizontal HMI on every seed, with median `0.790`
- seed `123` was about `79.2%` worse than live INS on horizontal RMSE

Current branch decision:

- do not promote a two-region passive-navigation claim yet
- do not add magnetic aiding next
- next branch should harden regional demo-pack realism and route-selection robustness first

## Why This Matters

The repo is no longer proving only that gravity can look good on a synthetic map. It now demonstrates a more realistic story:

- INS drifts during a denied mission
- gravity provides the core non-GNSS Earth-signature
- gradient likelihood reduces gravity ambiguity
- bathymetry adds complementary passive context
- the best current product is a delayed Earth-signature track that can stay passive and stealth-compatible

That is much closer to the actual use case than claiming “a gravimeter replaces GPS.”

## Other Important Benchmarks

The older milestones still matter because they explain the current architecture:

- Synthetic validated maritime baseline:
  live aided INS beats IMU-only by about `152x` on horizontal RMSE and `128x` on CEP95.
- Sequence matching beats PF:
  on the synthetic benchmark, `sequence_plus_gradient` remains the best current map matcher.
- Bounded-lag smoothing works better than closed-loop replay:
  direct PF or delayed sequence feedback into the live INS remained structurally unsafe, while the separate delayed-output path improved metrics without damaging the live solution.
- Public Norwegian gravity product path is in place:
  the repo can already ingest the NAG-TEC Bouguer anomaly export into the existing gravity runtime interface.

## What The Repo Simulates

At a high level, the end-to-end flow is:

1. generate a truth trajectory in geodetic, ECEF, and NED-consistent frames
2. load or prepare regional gravity and bathymetry products
3. simulate IMU and passive aiding sensors with realistic noise, drift, bandwidth, duty-cycle, and motion-coupling assumptions
4. propagate a local-level error-state INS
5. run observe-only Earth-signature matching using gravity as the mandatory channel
6. optionally publish a bounded-lag Earth-signature navigation output
7. compute navigation, integrity, and map-matching metrics
8. save reports, plots, JSON summaries, and NPZ archives

## Recommended Runs

Run the full promoted maritime demo:

```bash
python3 scripts/run_maritime_demo.py \
  --output-dir /tmp/gravnav_maritime_demo
```

Run the current regression suite:

```bash
python3 -m pytest -q
```

Run a single scenario directly:

```bash
python3 scripts/run_single_scenario.py \
  --scenario norwegian_margin_maritime \
  --sequence-profile configs/sequence_profiles/norwegian_margin_maritime_demo.json \
  --use-photonic-gravimeter \
  --use-gradiometer \
  --use-bathymetry
```

Benchmark estimator variants:

```bash
python3 scripts/benchmark_filters.py
```

## Validation

Current regression status:

- `49 passed`

The promoted maritime demo was validated on acceptance seeds:

- `42`
- `123`
- `777`

The winning promoted output is:

- `photonic_gravity_bathymetry_lag`

Validation boundary:

- the delayed Earth-signature output is promoted because it beats INS-only across the acceptance seeds with `0.0` horizontal HMI
- the live INS path is unchanged
- this is not a claim that closed-loop gravity feedback into the real-time INS is solved
- this is also not yet a claim that the same frozen passive stack generalizes across multiple offshore regions

## Realism Notes

The current demo is intentionally realistic in the ways that matter most for navigation:

- mission-duration denied run rather than a short toy route
- photonic gravimeter modeled as a mission-facing instrument with warm-up, cadence, bandwidth, drift, and residual motion coupling
- passive aids only: gravity, gradient, bathymetry, INS, depth, and velocity
- explicit initial INS offset so the Earth-signature estimator has to do real work

What this repo still does not claim:

- a universal gravity-only navigator
- a globally validated product on every map regime
- a full atom-interferometer optical or atomic-state simulator
- a solved real-time closed-loop gravity-feedback system

## Current Recommendation

If the goal is a credible GPS-denied passive-navigation demo, the current best story is:

- real-time navigation backbone: live INS with passive aiding
- global passive anchor: photonic-gravity-led Earth-signature sequence matcher
- supporting passive ambiguity reduction: gradient plus bathymetry
- reported navigation product: bounded-lag Earth-signature output

That is the first current repo path that is both technically defensible and better than INS-only under realistic denied-navigation conditions, but today it remains a single-region promoted result rather than a generalized offshore claim.

## Main Reference Docs

- [NEXT_STEPS_REPORT.md](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/reports/NEXT_STEPS_REPORT.md)
- [validated_maritime_baseline_report.md](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/reports/validated_maritime_baseline_report.md)
- [norwegian_margin_regional_benchmark_report.md](/Users/pranav/Downloads/photonic-gravimeter-sim/data/outputs/reports/norwegian_margin_regional_benchmark_report.md)
