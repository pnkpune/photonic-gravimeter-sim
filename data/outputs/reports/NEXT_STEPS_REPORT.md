# Next-Steps Report: From Gravity-Aided INS to Earth-Signature Navigator

**Merged roadmap synthesized from independent analyses by Claude (Opus) and ChatGPT (o1-pro)**
**Date: 2026-04-10**
**Repo state: validated maritime baseline, bounded-lag sequence smoother implemented, Norwegian-margin regional benchmark path added, PF feedback disabled, 35 tests passing**

---

## Current baseline (where we are)

The repo is a real, working gravity-aided navigation simulator. The validated maritime baseline proves:

| Metric | IMU-only | Aided (observe-only PF) | Improvement |
| --- | ---: | ---: | ---: |
| Horizontal RMSE [m] | 13,750 | 90.3 | 152× |
| CEP95 [m] | 21,580 | 169 | 128× |
| Vertical RMSE [m] | 2,059 | 0.387 | 5,319× |
| Gravimeter RMSE [m/s²] | — | 9.42e-06 | — |
| PF horizontal RMSE [m] | — | 128.6 | — |

The stack includes: WGS84 geodesy, nav-grade IMU model, scalar gravimeter with moving-base reduction, depth and velocity aiding, 15-state error-state INS, constrained fusion, particle-filter map matching (observe-only), integrity monitoring, metrics and reporting infrastructure.

PF feedback is intentionally disabled. Naive PF-to-INS pseudo-position injection destroys the INS (19 km RMSE). Covariance inflation makes it non-destructive but never better than observe-only. This is the central bottleneck.

New status beyond that baseline: the repo now also has a **separate bounded-lag smoothed navigation output** driven by `sequence_plus_gradient`. Unlike the closed-loop feedback experiments, this path leaves the live INS untouched and publishes a delayed navigation track. On `maritime_baseline`, seed `42`, it improves the live INS from **103.5 m** horizontal RMSE to **98.3 m** with **0.0%** horizontal HMI. That makes it the first delayed navigation path in the repo that improves navigation metrics without breaking the validated real-time solution.

---

## Regional benchmark update (2026-04-10)

The repo now has a **gravity-first Norwegian-margin regional benchmark path** that keeps the estimator stack unchanged and swaps only the map source:

- new `gravnav.datasets` layer that ingests a regular-grid CSV and produces a `GravityGridMap` cache plus manifest
- tracked in-repo Norwegian-margin fixture under `data/gravity_maps/raw/norwegian_margin/`
- processed cache and manifest under `data/gravity_maps/processed/`
- regional scenario `norwegian_margin_maritime`
- regional benchmark/report path comparing live INS, PF observe-only, sequence observe-only, and the bounded-lag smoother

Current result on `norwegian_margin_maritime`, seed `42`, `dt = 2.0 s`:

| Path | Horizontal RMSE [m] | CEP95 [m] | HMI horiz |
| --- | ---: | ---: | ---: |
| Live INS | 111.474 | 196.457 | 0.0 % |
| Best PF observe-only (`observe_plus_gradient`) | 131.762 | 253.578 | 0.0 % |
| Best sequence observe-only (`sequence_plus_gradient`) | 110.219 | 195.754 | 0.0 % |
| Bounded-lag smoother (`sequence_plus_gradient_lag_smoothed`) | 110.763 | 196.082 | 0.0 % |

What this means:

- the **ranking survives** on the regional benchmark: sequence matching still beats PF
- the bounded-lag smoother still beats the live INS, but only modestly
- the gains shrink sharply relative to the synthetic maritime benchmark, which means the earlier synthetic map materially overstated regional gravity distinctiveness
- the current regional path is a realism check, not yet a benchmark on a full public Norwegian-margin survey product

**Decision:** before adding more estimator complexity, the next highest-impact step is to replace the in-repo fixture with a full public regional product and rerun the exact same benchmark stack. The algorithm question is no longer “can sequence beat PF on synthetic maps?” It already can. The real question is how much of that survives once the map realism improves further.

---

## The core problem: why PF feedback fails

This is not a tuning problem. It is a **structural observability problem**.

Scalar gravity at a point gives a 1D measurement in 3D position space. The iso-gravity contours on any typical map are elongated curves. The PF posterior is therefore **ridge-shaped**: well-constrained perpendicular to the local gravity gradient, poorly constrained along it.

Feeding this posterior back as a pseudo-position measurement treats a ridge as if it were a point. The INS receives a confident but wrong constraint along the ridge direction. That is why it diverges.

**Evidence from the repo:** The PF's predicted disturbance spread across particles is only ~0.14 mGal versus ~1 mGal measurement noise. The ESS stays near maximum (~127/128). The particles are not being separated — the scalar measurement is too ambiguous to distinguish them in the current configuration.

---

## The plan: prioritized implementation backlog

### Priority 1 — Directional PF feedback (immediate, weeks 1-2)

**Goal:** Convert PF feedback from dangerous/off to information-aware/safe.

**What to build:**

1. **Eigendecompose the PF posterior covariance in local NED.** After each PF update, compute the 3×3 position covariance of the weighted particle cloud. Extract eigenvalues (λ₁ ≤ λ₂ ≤ λ₃) and corresponding eigenvectors (e₁, e₂, e₃).

2. **Inject a rank-1 directional pseudo-measurement along the well-constrained direction only.** The measurement model becomes:
   - H = e₁ᵀ (the unit vector along the smallest-eigenvalue direction)
   - z = e₁ᵀ · (PF posterior mean − INS position), projected scalar
   - R = λ₁ (the PF posterior variance in that direction)
   - This is a standard linear measurement update through the existing fusion interface

3. **Add acceptance gates based on observability indicators:**
   - Condition ratio: λ₃/λ₁ must exceed a threshold (ridge is well-defined)
   - ESS must be below a threshold (particles were actually discriminated)
   - Local map gradient norm must exceed minimum (there is information here)
   - Require N consecutive informative updates before first injection
   - Cap correction magnitude per update

4. **Add adaptive covariance inflation:** inflation = f(ESS, entropy, gradient norm, eigenvalue ratio). High ambiguity → huge inflation. Well-conditioned posterior → smaller inflation.

**Files to modify:**
- `src/gravnav/estimators/map_match_pf.py` — add posterior eigendecomposition and directional measurement extraction to `MapMatchPFUpdateResult`
- `src/gravnav/estimators/fusion.py` — add a directional pseudo-measurement constructor (rank-1 H matrix from eigenvector)
- `src/gravnav/simulation/runner.py` — wire directional feedback through the acceptance gate logic
- New: `src/gravnav/estimators/feedback_policy.py` — encapsulate gate conditions, adaptive inflation, persistence counters

**Success criterion:** Closed-loop PF feedback that beats the observe-only baseline (currently 90.3 m RMSE) without ever making it worse. Even a 5-10% improvement with guaranteed stability is a breakthrough relative to the current state.

**Outcome (2026-04-09 benchmark, seed 42):** Implemented as `DirectionalFeedbackController` in `src/gravnav/estimators/feedback_policy.py` with the full rank-1 + acceptance-gate machinery. Ran head-to-head against observe-only on `maritime_baseline`. Two failure modes found and fixed:

| Iteration | Horizontal RMSE | CEP95 | HMI horiz | Finding |
| --- | ---: | ---: | ---: | --- |
| Observe-only baseline | 103.5 m | 190.0 m | 0.0 % | — |
| Directional FB (full 3D eigendecomp) | 113.8 m | 220.5 m | 39.5 % | Smallest-eigenvalue direction was almost always `(0, 0, 1)` — the controller fires along vertical because depth aiding already makes that axis tight. Vertical bias accumulates, cross-couples through EKF into east. |
| Directional FB (`horizontal_only=True`) | 120.0 m | 248.6 m | — | Fires 20/1020 updates along real horizontal eigenvectors, but the PF mean is *biased relative to truth* (NE of INS when truth is SW of INS), so even small, inflated corrections pull INS the wrong way. |
| Gate sweep (4 configs) | 103.5–104.6 m | 190.0–192.0 m | 0.0–24.0 % | Any config that actually applies feedback ties or hurts RMSE and degrades integrity. Strict gates (`min_eigenvalue_ratio≥8`, `persistence_count≥3`, `max_correction_norm_m=2`, `base_inflation=15`) tie the baseline exactly at **103.5 m / 0 % HMI** because feedback never fires — a safe no-op. |

**Root cause:** The PF posterior itself is miscalibrated. The reported σ along the constrained direction is ~5 m but the true PF-to-truth distance is tens of metres — the scalar gravity likelihood collapses into a sharp but *biased* mode along a ridge. This is a **structural observability limit of scalar gravity on a smooth map**, not a gate-tuning problem. No re-weighting of a biased estimator recovers the lost information.

**Lock-in:** Defaults in `DirectionalFeedbackSpec` set to the conservative "safe no-op" config. Directional feedback code path is verified, tested, and ready to be re-enabled as soon as a second, orthogonal measurement channel makes the posterior honest.

**Decision:** Priority 1 is effectively complete in the *machinery* sense (code + gates + tests), but the *performance* success criterion is blocked on Priority 3. Proceed directly to Priority 3.

---

### Priority 2 — Observability analysis module (weeks 1-2, parallel with Priority 1)

**Goal:** Quantify when and where gravity navigation actually works.

**What to build:**

1. **Local gravity map Jacobian:** G = [∂δg/∂lat, ∂δg/∂lon, ∂δg/∂h] computed from finite differences of the map at each position.

2. **Trajectory-window observability Gramian:**
   ```
   O = Σ_k  Φ(t₀, t_k)ᵀ  G_kᵀ  R_k⁻¹  G_k  Φ(t₀, t_k)
   ```
   Accumulated over a sliding window. Eigenvalues of O indicate which position directions are observable.

3. **Navigation information density metric:** A scalar summary (e.g., log-det of O, or minimum eigenvalue) that indicates how much position information the gravity measurements provide at each trajectory segment.

4. **Feedback-allowed flag:** Binary policy — feedback is safe when the Gramian eigenvalues exceed thresholds. This feeds directly into the acceptance gates in Priority 1.

**Files to create:**
- `src/gravnav/analysis/observability.py` — Gramian computation, eigenvalue extraction, information density scoring
- `src/gravnav/analysis/__init__.py`

**Files to modify:**
- `src/gravnav/simulation/runner.py` — compute and log observability at each update
- `src/gravnav/plots/nav_plots.py` — add observability heatmap / eigenvalue timeline plot

---

### Priority 3 — Gravity gradient measurement model (weeks 2-3)

**Goal:** Break scalar ambiguity by adding gradient information.

**What to build:**

1. **Vertical gravity gradient (Γ_zz) from map finite differences:** Even without a physical gradiometer, simulate gradient measurements by evaluating the map at (lat, lon, h ± Δh) and computing the vertical derivative. Add measurement noise appropriate to a quantum gradiometer (~10 E = 10⁻⁸ s⁻²).

2. **Gradient likelihood in the PF:** The PF update becomes a 2D likelihood — gravity disturbance AND gradient must both match. This collapses the posterior ridge dramatically.

3. **Extended measurement model:** For the INS fusion path, gradient adds an independent measurement with a different H matrix (spatial second derivative of gravity), improving the observability rank from 1 to 2 at each point.

**Files to create:**
- `src/gravnav/sensors/gravity_gradiometer.py` — gradient measurement model, noise model, map-derived gradient computation
- `tests/test_gravity_gradiometer.py`

**Files to modify:**
- `src/gravnav/estimators/map_match_pf.py` — add optional gradient likelihood term
- `src/gravnav/simulation/runner.py` — wire gradient measurements into the update loop
- Configs: add gradiometer sensor specification

---

### Priority 4 — Sequence-based matching: Viterbi/HMM (weeks 3-5)

**Goal:** Exploit trajectory history instead of matching point-by-point.

**What to build:**

1. **Sliding-window HMM/Viterbi matcher:** Discretize candidate positions on a grid centered on the INS estimate. At each timestep, compute transition probabilities from INS velocity and emission probabilities from gravity (and gradient) likelihoods. Run Viterbi over the window to find the most-likely trajectory.

2. **Delayed pseudo-measurement injection:** The sequence matcher produces a trajectory estimate, not a point estimate. Inject the correction at the center of the window (delayed but more accurate).

3. **Comparison benchmark vs. PF:** Same scenario, same sensors, compare PF vs. Viterbi on accuracy, computational cost, and stability.

**Files to create:**
- `src/gravnav/estimators/gravity_sequence_match.py` — HMM/Viterbi implementation
- `tests/test_sequence_match.py`

**Files to modify:**
- `src/gravnav/simulation/runner.py` — add sequence matcher as an alternative estimator path

**Outcome (2026-04-09 benchmark, seed 42):** Implemented as `GravitySequenceMatcher` in `src/gravnav/estimators/gravity_sequence_match.py` and wired into `src/gravnav/simulation/runner.py`, `src/gravnav/simulation/results.py`, `src/gravnav/simulation/metrics.py`, `scripts/run_single_scenario.py`, and `scripts/benchmark_filters.py`. The first implementation is deliberately **observe-only**: it emits delayed trajectory estimates and metrics, but does not yet feed delayed corrections into the live INS.

Benchmark on `maritime_baseline`:

| Configuration | INS horizontal RMSE | Map-matcher horizontal RMSE | Map-matcher CEP95 | Finding |
| --- | ---: | ---: | ---: | --- |
| PF observe-only | 103.5 m | 98.94 m | 180.59 m | Reference PF result |
| PF + gradient | 103.5 m | 96.90 m | 178.54 m | Best current PF result |
| Sequence only | 103.5 m | 99.04 m | 179.50 m | Roughly ties scalar PF |
| Sequence + gradient | 103.5 m | **92.81 m** | **168.13 m** | Best current map-matching estimator |

**Interpretation:** Sequence matching is now justified as a first-class estimator path. History helps enough to beat the current PF on the map-matcher estimate itself, especially once gradient likelihood is included. But because the current implementation is observe-only, the INS metrics stay flat at the validated baseline. That means the next architectural step is **not** more pointwise PF tuning; it is designing a delayed sequence-to-INS feedback path that respects the delayed-estimate timing.

**Decision:** Priority 4 is implemented and benchmarked successfully in observe-only mode. The next work should focus on delayed feedback / smoothing-aware fusion for sequence outputs, or move in parallel on Priority 5 if sensor-physics fidelity becomes the bottleneck.

**Follow-up (2026-04-09 delayed-feedback experiment):** Implemented an initial `SequenceFeedbackController` that transfers the delayed sequence-estimated horizontal INS bias to the current state with covariance inflation proportional to delay. This path is wired into `src/gravnav/estimators/feedback_policy.py`, `src/gravnav/simulation/runner.py`, `src/gravnav/estimators/fusion.py`, and the CLI/benchmark scripts.

Result:

| Configuration | INS horizontal RMSE | Finding |
| --- | ---: | --- |
| Default conservative gate | 103.5 m | Safe no-op, zero applied updates |
| Active tune A (`peak>=0.05`, inflation 3.0, transfer RW 0.6) | 7.2 km | Catastrophic divergence |
| Active tune B (`peak>=0.05`, inflation 6.0, transfer RW 1.0) | 828 m | Still far worse than baseline |
| Active tune C (`peak>=0.05`, inflation 10.0, transfer RW 1.5, max corr 50 m) | 214.6 m | Less bad, still clearly worse |

**Conclusion:** A simple current-state transfer of delayed sequence bias is structurally inadequate. The delayed sequence estimate is useful, but not in the form “apply this old horizontal offset to the current INS state.” The next serious closed-loop step is a replay/smoother-aware delayed-update design, not more tuning of this transfer controller.

**Follow-up (2026-04-09 fixed-lag replay experiment):** Implemented a replay-aware delayed sequence feedback path. In this mode, the delayed sequence estimate is fused into the INS state at the delayed time, then the runner replays the stored IMU, depth, and velocity-aid measurements forward to the present. The sequence matcher is reset after an applied replay update so subsequent windows use consistent INS centers. This path is wired into `src/gravnav/estimators/feedback_policy.py`, `src/gravnav/simulation/runner.py`, `scripts/run_single_scenario.py`, and `scripts/benchmark_filters.py`.

Result on `maritime_baseline`, seed 42:

| Configuration | INS horizontal RMSE | CEP95 | HMI horiz | Applied updates | Finding |
| --- | ---: | ---: | ---: | ---: | --- |
| Sequence + gradient observe-only | 103.5 m | 190.0 m | 0.0 % | 0 | Reference |
| Replay feedback, conservative default | 103.5 m | 190.0 m | 0.0 % | 0 | Safe no-op |
| Replay feedback, relaxed (`peak>=0.03`, `max_std<=120 m`, `max_corr<=50 m`, inflation `10.0`) | 113.1 m | 220.1 m | 38.5 % | 93 | Much less destructive than bias-transfer, but still worse than observe-only |

**Interpretation:** The replay architecture is the correct structural move because it applies the estimate at the time it actually describes. But that alone is not enough. Even with lag replay, the current horizontal pseudo-measurement remains biased enough that active updates degrade both INS accuracy and integrity.

**Follow-up (2026-04-09 directional delayed-measurement experiment):** Upgraded the replay controller to support a richer delayed measurement model: instead of always injecting the full 2D horizontal posterior mean offset, the controller can eigendecompose the horizontal posterior covariance and inject only the best-constrained horizontal component as a rank-1 directional measurement at the delayed state. This is the sequence-side analogue of the earlier PF directional feedback idea, but now combined with lag replay.

Result on `maritime_baseline`, seed 42:

| Configuration | INS horizontal RMSE | CEP95 | HMI horiz | Applied updates | Finding |
| --- | ---: | ---: | ---: | ---: | --- |
| Replay, full-horizontal relaxed (`peak>=0.03`, `max_std<=120 m`, `max_corr<=50 m`, inflation `10.0`) | 113.1 m | 220.1 m | 38.5 % | 93 | Better than bias-transfer, but still unsafe |
| Replay, directional horizontal safe (`ratio>=1.15`, `peak>=0.03`, `max_std<=120 m`, `max_corr<=30 m`, inflation `10.0`) | 108.1 m | 211.5 m | **0.0 %** | 38 | Best current active delayed-feedback result; safer, but still worse than observe-only |

**Interpretation:** The richer delayed measurement model helps. Directional replay removes the integrity collapse seen in relaxed full-horizontal replay and gets much closer to the observe-only baseline. But it still does not actually beat the baseline. That means the remaining issue is not just “full 2D correction is too aggressive”; it is that the delayed posterior itself is still not accurate enough for direct pseudo-measurement injection.

**Decision:** Priority 4 remains complete only in the observe-only sense. The repo now contains:
- a clearly bad current-state transfer controller
- a better-structured full-horizontal lag-replay controller
- an even better directional lag-replay controller

None is yet acceptable as a production feedback path. The next serious step is no longer “add replay” or “add directional geometry”; it is “add replay plus retrodiction/smoother-aware delayed-update logic or a richer delayed state model.”

**Follow-up (2026-04-09 bounded-lag sequence smoother output):** Implemented the planned **fixed-lag smoothed navigation track as a first-class delayed estimator output**, without feeding it back into the live INS. The core pieces are:

- `SequenceAnchorEstimate` and anchor export from `GravitySequenceMatcher`
- `SequenceLagSmootherSpec` / `SequenceLagSmootherController`
- lag-buffered replay over stored IMU/depth/velocity history
- `SimulationEstimatorLog.lag_smoothed_states`
- lag-smoothed metrics and benchmark reporting

Two implementation bugs had to be fixed before this path became meaningful:

1. sequence outputs were initially keyed by the matcher's internal update index instead of truth-step time, so the lag smoother silently fell back to replay for most of the run
2. the default lag was initially interpreted in raw truth steps instead of **measurement-update lag**, which was wrong whenever gravity updates were downsampled relative to the truth grid

After fixing those, the lag-smoothed path became a real delayed navigation estimator instead of a no-op.

Primary benchmark on `maritime_baseline`, seed `42`:

| Path | Horizontal RMSE | CEP95 | Vertical RMSE | HMI horiz |
| --- | ---: | ---: | ---: | ---: |
| Live INS baseline | 103.5 m | 190.0 m | 0.821 m | 0.0 % |
| Sequence + gradient observe-only matcher | 92.8 m | 168.1 m | — | — |
| **Bounded-lag smoothed navigation output** | **98.3 m** | **181.9 m** | **0.583 m** | **0.0 %** |

Acceptance checks from the plan:

| Scenario | Seed | Live INS RMSE | Lag-smoothed RMSE | Live CEP95 | Lag-smoothed CEP95 | HMI horiz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `maritime_baseline` | 42 | 103.5 m | 98.3 m | 190.0 m | 181.9 m | 0.0 % |
| `maritime_baseline` | 123 | 85.7 m | 80.2 m | 153.2 m | 150.3 m | 0.0 % |
| `maritime_baseline` | 777 | 249.2 m | 234.0 m | 425.3 m | 413.8 m | 0.0 % |
| `uuv_long_endurance` | 42 | 1014.8 m | 1005.0 m | 2319.3 m | 2295.5 m | 0.0 % |

**Interpretation:** This is the first delayed navigation path in the repo that actually clears the success criterion of this phase:

- it beats the live INS on the primary maritime benchmark
- it also improves CEP95
- the gain survives the small multi-seed maritime check
- it also survives one additional scenario
- horizontal HMI remains zero in all tested runs

**Decision:** The bounded-lag sequence smoother is now a successful, validated **delayed-output estimator**. It should remain separate from the real-time INS in this phase. That is the right boundary: the repo now has a credible delayed navigation product, but it still does **not** have a validated closed-loop sequence-feedback solution.

---

### Priority 5 — Photonic gravimeter physics model (weeks 3-5, parallel with Priority 4)

**Goal:** Model the instrument you actually want to build, not a generic scalar sensor.

**What to build:**

1. **Atom interferometer phase response:** φ = k_eff · g · T², where T is tunable interrogation time and k_eff depends on laser wavelength and momentum transfer order.

2. **Quantum-noise-limited sensitivity:** Δg = 1/(k_eff · T² · √N), derivable from first principles given atom number N, interrogation time T, and effective wavevector k_eff.

3. **Systematic effects:** AC Stark (light) shifts, wavefront aberrations, Coriolis coupling in the interferometer (depends on platform rotation rate — couples to IMU), two-photon light shift.

4. **Vibration rejection model:** Simultaneous classical accelerometer used to post-correct atom interferometer fringes. The residual after correction is the actual measurement noise floor.

5. **Optional multi-axis / gradiometer configuration:** Same atoms, different interrogation geometry → gravity vector components or gradient.

**Files to create:**
- `src/gravnav/sensors/photonic_gravimeter.py` — physics-based atom interferometer model
- `configs/sensors/photonic_gravimeter_coldatom.json`
- `tests/test_photonic_gravimeter.py`

---

### Priority 6 — Fill scaffold files and harden infrastructure (weeks 2-4, ongoing)

**What to build:**

1. `scripts/run_monte_carlo.py` — batch scenario execution with parameter sweeps
2. `scripts/make_synthetic_map.py` — CLI for generating gravity maps with controlled anomaly spectra
3. `scripts/benchmark_filters.py` — automated comparison: INS-only / INS+depth+vel / INS+gravity observe-only / INS+directional PF feedback / INS+Viterbi
4. `tests/test_gravimeter.py` — moving-base recovery, noise model, bandwidth filtering
5. `tests/test_map_match_pf.py` — posterior convergence, ESS behavior, eigenvalue extraction
6. `pyproject.toml` — full metadata, dependencies, console entry points
7. YAML config population (mirror all JSON configs)
8. Wire residuals/NIS into integrity snapshots (currently zeroed out)

---

### Priority 7 — Real Earth data, regional scope (weeks 5-8)

**Goal:** Move from synthetic maps to real geophysical products. Start with one high-contrast maritime theater, not global.

**Data sources to integrate:**

| Product | Coverage | Resolution | Use |
| --- | --- | --- | --- |
| ICGEM (EGM2008 / EIGEN-6C4) | Global | ~5 arc-min (degree 2190) | Static gravity field |
| GRACE / GRACE-FO (via ICGEM) | Global | ~300 km | Temporal gravity variations |
| GEBCO_2025 | Global | 15 arc-sec | Bathymetry/topography |
| WMM2025 / IGRF-14 | Global | Degree 133/13 | Magnetic field |
| ERA5 | Global | 0.25° hourly | Atmospheric state |
| HYCOM GOFS 3.1 | Global ocean | 1/12° | Ocean currents, T/S |
| FES2022b | Global ocean | 1/16° | Tides and tide loading |

**Files to create:**
- `src/gravnav/datasets/gravity_loader.py` — ICGEM spherical harmonic evaluation or grid interpolation
- `src/gravnav/datasets/bathymetry_loader.py` — GEBCO NetCDF reader
- `src/gravnav/datasets/magnetic_loader.py` — WMM/IGRF evaluation
- `src/gravnav/datasets/tidal_model.py` — FES2022b tidal gravity prediction
- `src/gravnav/datasets/__init__.py`
- `data/gravity_maps/README.md` — map format specification (coordinate convention, units, reference height, interpolation method, temporal validity, source provenance, uncertainty layers, land/ocean masking, version metadata)

**First test region:** Pick a region with strong gravity gradients and good survey coverage (e.g., Mid-Atlantic Ridge, Mariana Trench approach, or Norwegian continental margin). Prove that the full pipeline works with real data before going global.

---

### Priority 8 — ML Layer 1: residual correction network (weeks 6-10)

**Goal:** Learn to predict and remove nuisance terms from gravimeter measurements.

**Architecture:** Temporal convolutional network (TCN) with dilated convolutions.

**Input features (per timestep):**
- Raw gravimeter output
- IMU specific force (3-axis) and angular rate (3-axis)
- Platform velocity and attitude estimates
- Vertical acceleration spectral features (sea-state proxy)
- Temperature, depth/pressure

**Output:** Predicted residual (scalar) + heteroscedastic uncertainty (scalar)

**Loss:** Negative log-likelihood with learned variance:
```
L = 0.5 · log(σ²_pred) + 0.5 · (y_true − y_pred)² / σ²_pred
```

**Training data:** Simulated trajectories with known truth (compute actual residuals), later augmented with real survey data where post-processed GNSS truth is available.

**Files to create:**
- `src/gravnav/ml/residual_correction.py` — TCN architecture, training loop, inference wrapper
- `src/gravnav/ml/__init__.py`
- `scripts/train_residual_model.py`
- `configs/ml/residual_tcn.yaml`

**Integration:** The residual model sits between raw gravimeter output and the PF/Viterbi measurement model. It produces a corrected gravity disturbance with calibrated uncertainty.

---

### Priority 9 — Multi-modal fusion expansion (months 2-3)

**Goal:** Gravity is one channel, not the whole system.

**Measurement channels to add:**
- Magnetic field vector (3-component, from WMM/IGRF map + anomaly)
- Bathymetry / echo-sounder (map matching against GEBCO)
- Tidal gravity signature (time-varying, predictable from FES2022b)
- Ocean current profile (from HYCOM, as nuisance correction or weak position constraint)

**Each channel needs:**
- Sensor model (measurement + noise + bias)
- Map/model interface (lookup expected value at position)
- Likelihood function for PF/Viterbi
- Observability contribution to the Gramian

**Architectural upgrade:** Move toward a layered estimator:
- Layer A: high-rate INS mechanization + bias states
- Layer B: physics observation models (gravity, gradient, magnetic, bathymetry, tidal)
- Layer C: global inference (PF/Viterbi/factor-graph hybrid, integrity, route planning)

---

### Priority 10 — Monte Carlo studies and benchmarking (months 2-3)

**Benchmark matrix:**

| Configuration | What it tests |
| --- | --- |
| INS only | Pure dead-reckoning drift baseline |
| INS + depth + velocity | Sensor aiding without map matching |
| INS + scalar gravity (observe-only PF) | Current validated baseline |
| INS + scalar gravity (directional feedback) | Priority 1 result |
| INS + scalar + gradient | Priority 3 addition |
| INS + Viterbi sequence matcher | Priority 4 addition |
| INS + multi-modal (gravity + magnetic + bathymetry) | Priority 9 result |
| INS + ML residual correction + multi-modal | Priority 8 + 9 result |

**Scenario variations:**
- High-gradient region vs. flat/ambiguous region
- Short mission (30 min) vs. long mission (24 hr) vs. very long (7 days)
- Surface vessel vs. deep UUV
- Calm seas vs. rough weather
- Good initial position vs. large initial error (adversarial)

---

## Longer-horizon research directions (months 3-6+)

These are promising but speculative. Pursue after Priorities 1-10 are solid.

### Neural Earth field (SIREN / Fourier feature network)
Train an implicit neural representation F(lat, lon, h, t) → multi-modal signature vector. Compression target: global gravity field at navigation-useful resolution in ~100 MB. Use SIREN architecture (sinusoidal activations match spherical harmonic structure). Progressive training from low to high spherical harmonic degree.

### Tidal gravity exploitation
Lunar/solar tidal signatures (~100 μGal) are predictable functions of (position, time). Over 24-hour missions, tidal phase constrains time independently; amplitude pattern constrains latitude; ocean loading signature constrains coastal proximity. Requires ~1 μGal sensitivity (achievable with cold-atom instruments).

### Spectral fingerprinting
Match gravity spectrograms (short-time Fourier transform over sliding windows) instead of point values. Different regions have different spectral textures. A 10-minute spectrogram is far more globally distinctive than a single scalar value. Use VGGish-style CNN to produce compact embeddings for fast nearest-neighbor matching.

### Information-theoretic route planning
Optimize trajectory for navigation information gain: I(w) = expected posterior entropy reduction at waypoint w. Fly perpendicular to iso-gravity contours. Slow down in flat regions to probe vertical gradient. Seek high-gradient corridors when possible.

### Learned proposal PF (ML Layer 3)
Use a sequence model (set transformer with temporal attention) to propose particles in high-likelihood regions. Training via contrastive learning: discriminate true trajectory segments from hard negatives displaced by 1-10 km. This replaces the INS-only proposal with a learned proposal that dramatically reduces required particle count.

### Sagnac rotation sensing
If the photonic architecture supports it: ring interferometer or matter-wave Sagnac gyroscope gives absolute rotation rate → absolute latitude from Ω·sin(φ) and absolute heading reference independent of magnetics.

### Synthetic aperture gravimetry
Combine gravity measurements along a trajectory to solve for local mass density at finer resolution than individual measurement footprint. Computationally expensive but could dramatically improve map-matching distinctiveness.

### Collaborative fleet SLAM
Multiple vehicles share gravity-signature-correlated relative constraints (not absolute positions) via distributed factor graph. Gravity-based loop closure across a fleet.

---

## Key references

### Gravity-aided navigation algorithms
- Li, Greentree, Moran — "Gravity-aided navigation using Viterbi map matching algorithm," Journal of Navigation (2024). Sequence-based HMM/Viterbi formulation; explicitly highlights sensor noise, spatial uncertainty, and map ambiguity.

### Quantum / moving-platform gravimetry
- Jensen et al. — "Airborne gravimetry with quantum technology: observations from Iceland and Greenland," ESSD (2025). Moving-platform quantum gravimetry data products and hybrid processing.
- Bidel et al. — "Absolute marine gravimetry with matter-wave interferometry," Nature Communications (2018). Moving-platform atom gravimetry ambiguity and INS/motion correction requirements.
- Nature — "Quantum sensing for gravity cartography" (2022). Gravity gradient sensing direction.

### Neural implicit representations
- Sitzmann et al. — "Implicit Neural Representations with Periodic Activation Functions" (SIREN), NeurIPS 2020. Foundation architecture for continuous fields with sinusoidal activations.

### Earth data products
- GEBCO_2025 — Global terrain/bathymetry, 15 arc-second grid (August 2025)
- WMM2025 — World Magnetic Model for navigation (current standard)
- IGRF-14 — Geomagnetic reference model, 2025-2030 predictive interval
- ICGEM — Global gravity models + GRACE/GRACE-FO temporal solutions
- ERA5 — Hourly global atmospheric reanalysis, 1940-present
- HYCOM GOFS 3.1 — Operational 1/12° global ocean prediction
- FES2022b — Global ocean tide model and atlas

### Multi-modal underwater navigation
- Reviews of terrain-aided navigation for underwater vehicles (bathymetric SLAM, observability-aware planning, gravity + geomagnetic combined aiding)

### Observability and information-theoretic planning
- Cadena et al. — "Past, Present, and Future of SLAM," IEEE T-RO 2016
- DiFrancesco et al. — "Gravity Gradiometer Systems," Geophysical Prospecting 2009
- GOCE mission — Spaceborne gravity gradiometry demonstration

---

## One-sentence summary

**The immediate next move is to swap the Norwegian fixture for a real public regional gravity product and rerun the same benchmark stack before adding more estimator complexity; physics realism is now the bottleneck, not missing filter variants.**
