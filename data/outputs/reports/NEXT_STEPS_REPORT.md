# Next-Steps Report: From Gravity-Aided INS to Earth-Signature Navigator

**Merged roadmap synthesized from independent analyses by Claude (Opus) and ChatGPT (o1-pro)**
**Date: 2026-04-09**
**Repo state: validated maritime baseline, PF feedback disabled, 14 tests passing**

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

**Build observability-aware directional PF feedback first, add gradient and sequence matching next, bring in real Earth data regionally, then layer ML on top — always inside a Bayesian framework where physics is the backbone and ML makes each component sharper.**
