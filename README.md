# photonic-gravimeter-sim

`photonic-gravimeter-sim` is a gravity-aided navigation simulation repository built around a real geodesy/navigation foundation instead of a toy Earth model.

The current codebase starts from:

- a WGS84 normal-gravity and geodesy layer
- a consistent ECEF/NED frame and rotation layer
- truth-trajectory and vehicle-motion builders
- sensor models for IMU, scalar gravimeter, depth aiding, and velocity aiding
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
- kinematics helpers
- IMU, gravimeter, depth, and velocity-aid sensor models
- truth trajectories, vehicle profiles, and named scenarios
- RNG, units, and config utilities

Still scaffold-only:

- `src/gravnav/physics/gravity_map.py`
- `src/gravnav/physics/corrections.py`
- `src/gravnav/estimators/*`
- `src/gravnav/simulation/*`
- `src/gravnav/plots/*`
- `scripts/*`
- `tests/*`
- `configs/*`
- `notebooks/*`
- `pyproject.toml`

So the repository currently contains the foundational physics/truth/sensor layers, but not yet the runnable end-to-end estimator/simulation package.

## Built-In Truth Scenarios

The current truth layer includes named scenario presets in `src/gravnav/truth/scenarios.py`:

- `maritime_baseline`
- `uuv_long_endurance`
- `uav_stress_case`

These are intended as baseline motion libraries for later simulation and estimator work.

## Near-Term Next Steps

The next meaningful layers to implement are:

1. gravity map representation and synthetic-map generation
2. INS / error-state propagation layer
3. gravity map matching and fusion
4. simulation runner, metrics, and plots
5. tests and real config files

## Notes

- YAML config support in `src/gravnav/utils/config.py` requires `PyYAML`.
- The repository is still in a foundation-building phase; packaging and test wiring are not finished yet.
