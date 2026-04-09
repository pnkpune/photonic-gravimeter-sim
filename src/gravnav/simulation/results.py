"""
results.py

Typed simulation-result containers and lightweight persistence helpers for the
gravity-aided navigation simulator.

Why this file exists
--------------------
The repository now has:
- truth trajectories
- sensor models that emit typed measurement objects
- INS / fusion / PF / integrity estimator layers

What the simulation package still needs is a clean place to collect all of that
into one run result that later code can:
- inspect
- summarize
- save to disk
- reload for plotting / benchmarking / Monte Carlo analysis

This module therefore provides:
1) metadata and summary containers
2) sensor-log and estimator-log containers
3) one top-level scenario/run result object
4) extraction helpers that convert measurement/state histories into NumPy arrays
5) compact JSON + NPZ persistence helpers

Conventions
-----------
- Angles are radians.
- Distances are metres.
- Velocities are m/s.
- Gravity and acceleration are m/s^2.
- Timestamps are seconds since scenario start.
- This module stores raw histories exactly as emitted by the simulator; it does
  not try to reinterpret the underlying physics.

Design notes
------------
- This file is intentionally lightweight and avoids introducing pandas/xarray as
  a hard dependency.
- The top-level save format is a compressed `.npz` archive for numeric arrays
  plus a few JSON blobs for metadata and summary structures.
- The loader returns a plain mapping rather than reconstructing every rich object,
  because the immediate downstream use cases are plotting, benchmarking, and
  ad hoc inspection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..estimators.error_state_ins import ErrorStateINS, ErrorStateINSState
from ..estimators.gravity_sequence_match import SequenceMatchUpdateResult
from ..estimators.integrity import IntegritySnapshot
from ..estimators.map_match_pf import MapMatchPFUpdateResult
from ..truth.trajectory import TruthTrajectory

FloatArray = NDArray[np.float64]


# -----------------------------------------------------------------------------
# Low-level helpers
# -----------------------------------------------------------------------------


def _as_float_array(x: ArrayLike) -> FloatArray:
    """Convert input to a NumPy float64 array."""
    return np.asarray(x, dtype=np.float64)


def _time_array(x: ArrayLike, *, name: str) -> FloatArray:
    """
    Validate and return a one-dimensional time array.
    """
    arr = _as_float_array(x).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return arr


def _vec3(x: ArrayLike, *, name: str) -> FloatArray:
    """
    Validate and return a 3-vector.
    """
    arr = _as_float_array(x).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


def _mat3(x: ArrayLike, *, name: str) -> FloatArray:
    """
    Validate and return a 3x3 matrix.
    """
    arr = _as_float_array(x)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {arr.shape}.")
    return arr


def _ensure_output_path(path: str | Path) -> Path:
    """
    Resolve an output path and create its parent directory if needed.
    """
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _jsonable(obj: Any) -> Any:
    """
    Recursively convert an object into JSON-serializable form.

    Rules
    -----
    - dataclasses -> dict via `asdict`
    - numpy scalars -> Python scalars
    - numpy arrays -> nested lists
    - Path -> string
    - mappings / sequences -> recurse
    - unknown objects -> repr(obj)

    Notes
    -----
    This helper is intentionally conservative and readability-first.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if is_dataclass(obj):
        return _jsonable(asdict(obj))

    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]

    return repr(obj)


def _state_from_filter_or_state(
    ins_or_state: ErrorStateINS | ErrorStateINSState,
) -> ErrorStateINSState:
    """
    Return `ErrorStateINSState` regardless of whether the caller passed a filter
    object or the state directly.
    """
    if isinstance(ins_or_state, ErrorStateINS):
        return ins_or_state.state
    if isinstance(ins_or_state, ErrorStateINSState):
        return ins_or_state
    raise TypeError(
        "Expected ErrorStateINS or ErrorStateINSState, "
        f"got {type(ins_or_state).__name__}."
    )


def _get_required_attr(obj: Any, attr: str) -> Any:
    """
    Return a required attribute or raise a readable error.
    """
    if not hasattr(obj, attr):
        raise AttributeError(
            f"Object of type {type(obj).__name__} does not define required "
            f"attribute {attr!r}."
        )
    return getattr(obj, attr)


def _safe_float_attr(obj: Any, attr: str, default: float = np.nan) -> float:
    """
    Return a float-valued attribute if present and not None, else `default`.
    """
    value = getattr(obj, attr, default)
    if value is None:
        return float(default)
    return float(value)


def _safe_bool_attr(obj: Any, attr: str, default: bool = False) -> bool:
    """
    Return a bool-valued attribute if present and not None, else `default`.
    """
    value = getattr(obj, attr, default)
    if value is None:
        return bool(default)
    return bool(value)


# -----------------------------------------------------------------------------
# Generic extraction helpers
# -----------------------------------------------------------------------------


def extract_scalar_measurement_history(
    measurements: Sequence[Any],
    *,
    value_attr: str,
    time_attr: str = "time_s",
    ideal_attr: Optional[str] = None,
) -> dict[str, FloatArray]:
    """
    Extract time and scalar values from a sequence of scalar measurement objects.

    Parameters
    ----------
    measurements : sequence
        Sequence of arbitrary objects.
    value_attr : str
        Name of the scalar value attribute to extract.
    time_attr : str, default="time_s"
        Name of the time attribute.
    ideal_attr : str, optional
        Optional ideal-value attribute to extract.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping containing:
        - `"time_s"`
        - `"value"`
        - optionally `"ideal_value"`

    Notes
    -----
    Missing timestamps are recorded as `np.nan`.
    """
    n = len(measurements)
    times = np.full(n, np.nan, dtype=np.float64)
    values = np.empty(n, dtype=np.float64)
    ideals = None if ideal_attr is None else np.full(n, np.nan, dtype=np.float64)

    for k, meas in enumerate(measurements):
        times[k] = _safe_float_attr(meas, time_attr, np.nan)
        values[k] = float(_get_required_attr(meas, value_attr))
        if ideals is not None:
            value = getattr(meas, ideal_attr, np.nan)
            ideals[k] = np.nan if value is None else float(value)

    out = {
        "time_s": times,
        "value": values,
    }
    if ideals is not None:
        out["ideal_value"] = ideals
    return out


def extract_vector_measurement_history(
    measurements: Sequence[Any],
    *,
    value_attr: str,
    time_attr: str = "time_s",
    ideal_attr: Optional[str] = None,
) -> dict[str, FloatArray]:
    """
    Extract time and 3-vector values from a sequence of vector measurement objects.

    Parameters
    ----------
    measurements : sequence
        Sequence of arbitrary objects.
    value_attr : str
        Name of the vector value attribute.
    time_attr : str, default="time_s"
        Name of the time attribute.
    ideal_attr : str, optional
        Optional ideal-value attribute to extract.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping containing:
        - `"time_s"`
        - `"value"`
        - optionally `"ideal_value"`

    Notes
    -----
    Missing timestamps are recorded as `np.nan`.
    """
    n = len(measurements)
    times = np.full(n, np.nan, dtype=np.float64)
    values = np.empty((n, 3), dtype=np.float64)
    ideals = None if ideal_attr is None else np.full((n, 3), np.nan, dtype=np.float64)

    for k, meas in enumerate(measurements):
        times[k] = _safe_float_attr(meas, time_attr, np.nan)
        values[k] = _vec3(_get_required_attr(meas, value_attr), name=value_attr)
        if ideals is not None:
            value = getattr(meas, ideal_attr, None)
            if value is not None:
                ideals[k] = _vec3(value, name=ideal_attr)

    out = {
        "time_s": times,
        "value": values,
    }
    if ideals is not None:
        out["ideal_value"] = ideals
    return out


# -----------------------------------------------------------------------------
# Metadata and summary containers
# -----------------------------------------------------------------------------


@dataclass
class SimulationMetadata:
    """
    Metadata describing one simulation run.

    Attributes
    ----------
    scenario_name : str or None
        Human-readable scenario name.
    run_id : str or None
        Optional run identifier.
    description : str or None
        Optional free-text description.
    config : dict
        Optional configuration mapping used to generate the run.
    rng_state : dict or None
        Optional serialized RNG state.
    extra : dict
        Additional user-defined metadata.
    """

    scenario_name: Optional[str] = None
    run_id: Optional[str] = None
    description: Optional[str] = None
    config: dict[str, Any] = field(default_factory=dict)
    rng_state: Optional[dict[str, Any]] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "SimulationMetadata":
        """Deep-ish copy of the metadata container."""
        return SimulationMetadata(
            scenario_name=self.scenario_name,
            run_id=self.run_id,
            description=self.description,
            config=dict(self.config),
            rng_state=None if self.rng_state is None else dict(self.rng_state),
            extra=dict(self.extra),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return _jsonable(asdict(self))


@dataclass
class ScenarioSimulationSummary:
    """
    Compact summary of one scenario simulation.

    Attributes
    ----------
    scenario_name : str or None
        Scenario name.
    run_id : str or None
        Optional run identifier.
    duration_s : float
        Truth duration [s].
    num_truth_samples : int
        Number of truth samples.
    num_imu_samples : int
        Number of IMU samples.
    num_gravimeter_samples : int
        Number of gravimeter samples.
    num_depth_samples : int
        Number of depth samples.
    num_velocity_aid_samples : int
        Number of velocity-aid samples.
    num_ins_states : int
        Number of INS states logged.
    num_pf_updates : int
        Number of PF updates logged.
    num_sequence_updates : int
        Number of sequence-matcher updates logged.
    num_integrity_snapshots : int
        Number of integrity snapshots logged.
    """

    scenario_name: Optional[str]
    run_id: Optional[str]
    duration_s: float
    num_truth_samples: int
    num_imu_samples: int
    num_gravimeter_samples: int
    num_depth_samples: int
    num_velocity_aid_samples: int
    num_ins_states: int
    num_pf_updates: int
    num_sequence_updates: int
    num_integrity_snapshots: int

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return _jsonable(asdict(self))


# -----------------------------------------------------------------------------
# Sensor log container
# -----------------------------------------------------------------------------


@dataclass
class SimulationSensorLog:
    """
    Logged sensor outputs for one simulation run.

    Attributes
    ----------
    imu_samples : list
        IMU measurement objects.
    gravimeter_samples : list
        Gravimeter measurement objects.
    depth_samples : list
        Depth measurement objects.
    velocity_aid_samples : list
        Velocity-aid measurement objects.
    custom_streams : dict[str, list]
        Optional user-defined sensor-like streams.
    """

    imu_samples: list[Any] = field(default_factory=list)
    gravimeter_samples: list[Any] = field(default_factory=list)
    depth_samples: list[Any] = field(default_factory=list)
    velocity_aid_samples: list[Any] = field(default_factory=list)
    custom_streams: dict[str, list[Any]] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        """
        Return sample counts by stream.
        """
        out = {
            "imu": len(self.imu_samples),
            "gravimeter": len(self.gravimeter_samples),
            "depth": len(self.depth_samples),
            "velocity_aid": len(self.velocity_aid_samples),
        }
        for key, value in self.custom_streams.items():
            out[f"custom:{key}"] = len(value)
        return out

    def add_custom_sample(self, name: str, sample: Any) -> None:
        """
        Append one sample to a named custom sensor stream.
        """
        key = str(name)
        self.custom_streams.setdefault(key, []).append(sample)

    def imu_history_arrays(self) -> dict[str, np.ndarray]:
        """
        Extract IMU history arrays.

        Returns
        -------
        dict[str, np.ndarray]
            Keys include:
            - `imu_time_s`
            - `imu_omega_ib_b_radps`
            - `imu_f_ib_b_mps2`
            - ideal, bias, noise, and saturation arrays
        """
        n = len(self.imu_samples)
        out: dict[str, np.ndarray] = {
            "imu_time_s": np.full(n, np.nan, dtype=np.float64),
            "imu_omega_ib_b_radps": np.empty((n, 3), dtype=np.float64),
            "imu_f_ib_b_mps2": np.empty((n, 3), dtype=np.float64),
            "imu_ideal_omega_ib_b_radps": np.empty((n, 3), dtype=np.float64),
            "imu_ideal_f_ib_b_mps2": np.empty((n, 3), dtype=np.float64),
            "imu_gyro_bias_used_radps": np.empty((n, 3), dtype=np.float64),
            "imu_accel_bias_used_mps2": np.empty((n, 3), dtype=np.float64),
            "imu_gyro_white_noise_radps": np.empty((n, 3), dtype=np.float64),
            "imu_accel_white_noise_mps2": np.empty((n, 3), dtype=np.float64),
            "imu_gyro_saturated": np.empty(n, dtype=bool),
            "imu_accel_saturated": np.empty(n, dtype=bool),
        }

        for k, meas in enumerate(self.imu_samples):
            out["imu_time_s"][k] = _safe_float_attr(meas, "time_s", np.nan)
            out["imu_omega_ib_b_radps"][k] = _vec3(
                _get_required_attr(meas, "omega_ib_b_radps"),
                name="omega_ib_b_radps",
            )
            out["imu_f_ib_b_mps2"][k] = _vec3(
                _get_required_attr(meas, "f_ib_b_mps2"),
                name="f_ib_b_mps2",
            )
            out["imu_ideal_omega_ib_b_radps"][k] = _vec3(
                _get_required_attr(meas, "ideal_omega_ib_b_radps"),
                name="ideal_omega_ib_b_radps",
            )
            out["imu_ideal_f_ib_b_mps2"][k] = _vec3(
                _get_required_attr(meas, "ideal_f_ib_b_mps2"),
                name="ideal_f_ib_b_mps2",
            )
            out["imu_gyro_bias_used_radps"][k] = _vec3(
                _get_required_attr(meas, "gyro_bias_used_radps"),
                name="gyro_bias_used_radps",
            )
            out["imu_accel_bias_used_mps2"][k] = _vec3(
                _get_required_attr(meas, "accel_bias_used_mps2"),
                name="accel_bias_used_mps2",
            )
            out["imu_gyro_white_noise_radps"][k] = _vec3(
                _get_required_attr(meas, "gyro_white_noise_radps"),
                name="gyro_white_noise_radps",
            )
            out["imu_accel_white_noise_mps2"][k] = _vec3(
                _get_required_attr(meas, "accel_white_noise_mps2"),
                name="accel_white_noise_mps2",
            )
            out["imu_gyro_saturated"][k] = _safe_bool_attr(
                meas, "gyro_saturated", False
            )
            out["imu_accel_saturated"][k] = _safe_bool_attr(
                meas, "accel_saturated", False
            )
        return out

    def gravimeter_history_arrays(self) -> dict[str, np.ndarray]:
        """
        Extract gravimeter history arrays.
        """
        n = len(self.gravimeter_samples)
        out: dict[str, np.ndarray] = {
            "gravimeter_time_s": np.full(n, np.nan, dtype=np.float64),
            "gravimeter_kind": np.empty(n, dtype=object),
            "gravimeter_value_mps2": np.empty(n, dtype=np.float64),
            "gravimeter_ideal_value_mps2": np.empty(n, dtype=np.float64),
            "gravimeter_motion_residual_mps2": np.empty(n, dtype=np.float64),
            "gravimeter_filtered_input_mps2": np.empty(n, dtype=np.float64),
            "gravimeter_bias_used_mps2": np.empty(n, dtype=np.float64),
            "gravimeter_white_noise_mps2": np.empty(n, dtype=np.float64),
            "gravimeter_saturated": np.empty(n, dtype=bool),
        }

        for k, meas in enumerate(self.gravimeter_samples):
            out["gravimeter_time_s"][k] = _safe_float_attr(meas, "time_s", np.nan)
            out["gravimeter_kind"][k] = str(_get_required_attr(meas, "kind"))
            out["gravimeter_value_mps2"][k] = float(
                _get_required_attr(meas, "value_mps2")
            )
            out["gravimeter_ideal_value_mps2"][k] = float(
                _get_required_attr(meas, "ideal_value_mps2")
            )
            out["gravimeter_motion_residual_mps2"][k] = float(
                _get_required_attr(meas, "motion_residual_mps2")
            )
            out["gravimeter_filtered_input_mps2"][k] = float(
                _get_required_attr(meas, "filtered_input_mps2")
            )
            out["gravimeter_bias_used_mps2"][k] = float(
                _get_required_attr(meas, "bias_used_mps2")
            )
            out["gravimeter_white_noise_mps2"][k] = float(
                _get_required_attr(meas, "white_noise_mps2")
            )
            out["gravimeter_saturated"][k] = _safe_bool_attr(
                meas, "saturated", False
            )

        # Convert object dtype to a fixed-width Unicode array for NPZ friendliness.
        out["gravimeter_kind"] = np.asarray(out["gravimeter_kind"], dtype=str)
        return out

    def depth_history_arrays(self) -> dict[str, np.ndarray]:
        """
        Extract depth-sensor history arrays.
        """
        n = len(self.depth_samples)
        out: dict[str, np.ndarray] = {
            "depth_time_s": np.full(n, np.nan, dtype=np.float64),
            "depth_value_m": np.empty(n, dtype=np.float64),
            "depth_ideal_depth_m": np.empty(n, dtype=np.float64),
            "depth_filtered_depth_m": np.empty(n, dtype=np.float64),
            "depth_bias_used_m": np.empty(n, dtype=np.float64),
            "depth_white_noise_m": np.empty(n, dtype=np.float64),
            "depth_saturated": np.empty(n, dtype=bool),
            "depth_reference_surface_height_m": np.empty(n, dtype=np.float64),
        }

        for k, meas in enumerate(self.depth_samples):
            out["depth_time_s"][k] = _safe_float_attr(meas, "time_s", np.nan)
            out["depth_value_m"][k] = float(_get_required_attr(meas, "value_m"))
            out["depth_ideal_depth_m"][k] = float(
                _get_required_attr(meas, "ideal_depth_m")
            )
            out["depth_filtered_depth_m"][k] = float(
                _get_required_attr(meas, "filtered_depth_m")
            )
            out["depth_bias_used_m"][k] = float(
                _get_required_attr(meas, "bias_used_m")
            )
            out["depth_white_noise_m"][k] = float(
                _get_required_attr(meas, "white_noise_m")
            )
            out["depth_saturated"][k] = _safe_bool_attr(meas, "saturated", False)
            out["depth_reference_surface_height_m"][k] = float(
                _get_required_attr(meas, "reference_surface_height_m")
            )
        return out

    def velocity_aid_history_arrays(self) -> dict[str, np.ndarray]:
        """
        Extract velocity-aid history arrays.
        """
        n = len(self.velocity_aid_samples)
        out: dict[str, np.ndarray] = {
            "velocity_aid_time_s": np.full(n, np.nan, dtype=np.float64),
            "velocity_aid_kind": np.empty(n, dtype=object),
            "velocity_aid_frame": np.empty(n, dtype=object),
            "velocity_aid_value_mps": np.empty((n, 3), dtype=np.float64),
            "velocity_aid_ideal_value_mps": np.empty((n, 3), dtype=np.float64),
            "velocity_aid_filtered_input_mps": np.empty((n, 3), dtype=np.float64),
            "velocity_aid_bias_used_mps": np.empty((n, 3), dtype=np.float64),
            "velocity_aid_white_noise_mps": np.empty((n, 3), dtype=np.float64),
            "velocity_aid_saturated": np.empty(n, dtype=bool),
        }

        for k, meas in enumerate(self.velocity_aid_samples):
            out["velocity_aid_time_s"][k] = _safe_float_attr(meas, "time_s", np.nan)
            out["velocity_aid_kind"][k] = str(_get_required_attr(meas, "kind"))
            out["velocity_aid_frame"][k] = str(_get_required_attr(meas, "frame"))
            out["velocity_aid_value_mps"][k] = _vec3(
                _get_required_attr(meas, "value_mps"),
                name="value_mps",
            )
            out["velocity_aid_ideal_value_mps"][k] = _vec3(
                _get_required_attr(meas, "ideal_value_mps"),
                name="ideal_value_mps",
            )
            out["velocity_aid_filtered_input_mps"][k] = _vec3(
                _get_required_attr(meas, "filtered_input_mps"),
                name="filtered_input_mps",
            )
            out["velocity_aid_bias_used_mps"][k] = _vec3(
                _get_required_attr(meas, "bias_used_mps"),
                name="bias_used_mps",
            )
            out["velocity_aid_white_noise_mps"][k] = _vec3(
                _get_required_attr(meas, "white_noise_mps"),
                name="white_noise_mps",
            )
            out["velocity_aid_saturated"][k] = _safe_bool_attr(
                meas, "saturated", False
            )

        out["velocity_aid_kind"] = np.asarray(out["velocity_aid_kind"], dtype=str)
        out["velocity_aid_frame"] = np.asarray(out["velocity_aid_frame"], dtype=str)
        return out


# -----------------------------------------------------------------------------
# Estimator log container
# -----------------------------------------------------------------------------


@dataclass
class SimulationEstimatorLog:
    """
    Logged estimator outputs for one simulation run.

    Attributes
    ----------
    ins_states : list[ErrorStateINSState]
        Logged INS states.
    pf_updates : list[MapMatchPFUpdateResult]
        Logged gravity-map PF updates.
    sequence_updates : list[SequenceMatchUpdateResult]
        Logged delayed sequence-matcher updates.
    integrity_snapshots : list[IntegritySnapshot]
        Logged integrity snapshots.
    custom_streams : dict[str, list]
        Optional user-defined estimator streams.
    """

    ins_states: list[ErrorStateINSState] = field(default_factory=list)
    pf_updates: list[MapMatchPFUpdateResult] = field(default_factory=list)
    sequence_updates: list[SequenceMatchUpdateResult] = field(default_factory=list)
    integrity_snapshots: list[IntegritySnapshot] = field(default_factory=list)
    custom_streams: dict[str, list[Any]] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        """
        Return sample counts by estimator stream.
        """
        out = {
            "ins_states": len(self.ins_states),
            "pf_updates": len(self.pf_updates),
            "sequence_updates": len(self.sequence_updates),
            "integrity_snapshots": len(self.integrity_snapshots),
        }
        for key, value in self.custom_streams.items():
            out[f"custom:{key}"] = len(value)
        return out

    def append_ins_state(
        self,
        ins_or_state: ErrorStateINS | ErrorStateINSState,
    ) -> None:
        """
        Append a deep copy of an INS state to the log.
        """
        self.ins_states.append(_state_from_filter_or_state(ins_or_state).copy())

    def add_custom_sample(self, name: str, sample: Any) -> None:
        """
        Append one sample to a named custom estimator stream.
        """
        key = str(name)
        self.custom_streams.setdefault(key, []).append(sample)

    def ins_history_arrays(self) -> dict[str, np.ndarray]:
        """
        Extract INS history arrays from logged states.

        Returns
        -------
        dict[str, np.ndarray]
            Keys include:
            - `ins_time_s`
            - `ins_lat_rad`, `ins_lon_rad`, `ins_height_m`
            - `ins_v_ned_mps`
            - `ins_C_n_b`
            - `ins_gyro_bias_radps`
            - `ins_accel_bias_mps2`
            - `ins_P`
        """
        n = len(self.ins_states)
        out: dict[str, np.ndarray] = {
            "ins_time_s": np.full(n, np.nan, dtype=np.float64),
            "ins_lat_rad": np.empty(n, dtype=np.float64),
            "ins_lon_rad": np.empty(n, dtype=np.float64),
            "ins_height_m": np.empty(n, dtype=np.float64),
            "ins_v_ned_mps": np.empty((n, 3), dtype=np.float64),
            "ins_C_n_b": np.empty((n, 3, 3), dtype=np.float64),
            "ins_gyro_bias_radps": np.empty((n, 3), dtype=np.float64),
            "ins_accel_bias_mps2": np.empty((n, 3), dtype=np.float64),
            "ins_P": np.empty((n, 15, 15), dtype=np.float64),
        }

        for k, state in enumerate(self.ins_states):
            nom = state.nominal
            out["ins_time_s"][k] = _safe_float_attr(nom, "time_s", np.nan)
            out["ins_lat_rad"][k] = float(_get_required_attr(nom, "lat_rad"))
            out["ins_lon_rad"][k] = float(_get_required_attr(nom, "lon_rad"))
            out["ins_height_m"][k] = float(_get_required_attr(nom, "height_m"))
            out["ins_v_ned_mps"][k] = _vec3(
                _get_required_attr(nom, "v_ned_mps"),
                name="v_ned_mps",
            )
            out["ins_C_n_b"][k] = _mat3(
                _get_required_attr(nom, "C_n_b"),
                name="C_n_b",
            )
            out["ins_gyro_bias_radps"][k] = _vec3(
                _get_required_attr(nom, "gyro_bias_radps"),
                name="gyro_bias_radps",
            )
            out["ins_accel_bias_mps2"][k] = _vec3(
                _get_required_attr(nom, "accel_bias_mps2"),
                name="accel_bias_mps2",
            )
            P = _as_float_array(state.P)
            if P.shape != (15, 15):
                raise ValueError(f"INS covariance must be (15, 15), got {P.shape}.")
            out["ins_P"][k] = P

        return out

    def pf_history_arrays(self) -> dict[str, np.ndarray]:
        """
        Extract PF update history arrays.
        """
        n = len(self.pf_updates)
        out: dict[str, np.ndarray] = {
            "pf_time_s": np.full(n, np.nan, dtype=np.float64),
            "pf_lat_rad": np.empty(n, dtype=np.float64),
            "pf_lon_rad": np.empty(n, dtype=np.float64),
            "pf_height_m": np.empty(n, dtype=np.float64),
            "pf_covariance_ned_m2": np.empty((n, 3, 3), dtype=np.float64),
            "pf_covariance_geodetic": np.empty((n, 3, 3), dtype=np.float64),
            "pf_effective_sample_size": np.empty(n, dtype=np.float64),
            "pf_effective_sample_size_before": np.empty(n, dtype=np.float64),
            "pf_effective_sample_size_after": np.empty(n, dtype=np.float64),
            "pf_predicted_disturbance_mps2": np.full(n, np.nan, dtype=np.float64),
            "pf_predicted_disturbance_mean_mps2": np.empty(n, dtype=np.float64),
            "pf_predicted_disturbance_std_mps2": np.empty(n, dtype=np.float64),
            "pf_resampled": np.empty(n, dtype=bool),
        }

        for k, upd in enumerate(self.pf_updates):
            est = upd.estimate
            out["pf_time_s"][k] = _safe_float_attr(upd, "time_s", np.nan)
            out["pf_lat_rad"][k] = float(_get_required_attr(est, "lat_rad"))
            out["pf_lon_rad"][k] = float(_get_required_attr(est, "lon_rad"))
            out["pf_height_m"][k] = float(_get_required_attr(est, "height_m"))
            out["pf_covariance_ned_m2"][k] = _mat3(
                _get_required_attr(est, "covariance_ned_m2"),
                name="covariance_ned_m2",
            )
            out["pf_covariance_geodetic"][k] = _mat3(
                _get_required_attr(est, "covariance_geodetic"),
                name="covariance_geodetic",
            )
            out["pf_effective_sample_size"][k] = float(
                _get_required_attr(est, "effective_sample_size")
            )
            out["pf_effective_sample_size_before"][k] = float(
                _get_required_attr(upd, "effective_sample_size_before")
            )
            out["pf_effective_sample_size_after"][k] = float(
                _get_required_attr(upd, "effective_sample_size_after")
            )

            pred = getattr(est, "predicted_disturbance_mps2", None)
            out["pf_predicted_disturbance_mps2"][k] = (
                np.nan if pred is None else float(pred)
            )
            out["pf_predicted_disturbance_mean_mps2"][k] = float(
                _get_required_attr(upd, "predicted_disturbance_mean_mps2")
            )
            out["pf_predicted_disturbance_std_mps2"][k] = float(
                _get_required_attr(upd, "predicted_disturbance_std_mps2")
            )
            out["pf_resampled"][k] = bool(_get_required_attr(upd, "resampled"))

        return out

    def sequence_history_arrays(self) -> dict[str, np.ndarray]:
        """
        Extract sequence-matcher update history arrays.
        """
        n = len(self.sequence_updates)
        out: dict[str, np.ndarray] = {
            "sequence_time_s": np.full(n, np.nan, dtype=np.float64),
            "sequence_lat_rad": np.empty(n, dtype=np.float64),
            "sequence_lon_rad": np.empty(n, dtype=np.float64),
            "sequence_height_m": np.empty(n, dtype=np.float64),
            "sequence_covariance_ned_m2": np.empty((n, 3, 3), dtype=np.float64),
            "sequence_covariance_geodetic": np.empty((n, 3, 3), dtype=np.float64),
            "sequence_window_size_used": np.empty(n, dtype=np.int64),
            "sequence_delayed_by_steps": np.empty(n, dtype=np.int64),
            "sequence_num_candidates": np.empty(n, dtype=np.int64),
            "sequence_posterior_entropy_nats": np.empty(n, dtype=np.float64),
            "sequence_marginal_peak_probability": np.empty(n, dtype=np.float64),
            "sequence_predicted_disturbance_mean_mps2": np.empty(n, dtype=np.float64),
            "sequence_predicted_disturbance_std_mps2": np.empty(n, dtype=np.float64),
            "sequence_used_gradient": np.empty(n, dtype=bool),
            "sequence_viterbi_log_score": np.empty(n, dtype=np.float64),
            "sequence_viterbi_offset_ned_m": np.empty((n, 3), dtype=np.float64),
        }

        for k, upd in enumerate(self.sequence_updates):
            est = upd.estimate
            out["sequence_time_s"][k] = float(_get_required_attr(upd, "time_s"))
            out["sequence_lat_rad"][k] = float(_get_required_attr(est, "lat_rad"))
            out["sequence_lon_rad"][k] = float(_get_required_attr(est, "lon_rad"))
            out["sequence_height_m"][k] = float(_get_required_attr(est, "height_m"))
            out["sequence_covariance_ned_m2"][k] = _mat3(
                _get_required_attr(est, "covariance_ned_m2"),
                name="covariance_ned_m2",
            )
            out["sequence_covariance_geodetic"][k] = _mat3(
                _get_required_attr(est, "covariance_geodetic"),
                name="covariance_geodetic",
            )
            out["sequence_window_size_used"][k] = int(
                _get_required_attr(upd, "window_size_used")
            )
            out["sequence_delayed_by_steps"][k] = int(
                _get_required_attr(upd, "delayed_by_steps")
            )
            out["sequence_num_candidates"][k] = int(
                _get_required_attr(upd, "num_candidates")
            )
            out["sequence_posterior_entropy_nats"][k] = float(
                _get_required_attr(upd, "posterior_entropy_nats")
            )
            out["sequence_marginal_peak_probability"][k] = float(
                _get_required_attr(upd, "marginal_peak_probability")
            )
            out["sequence_predicted_disturbance_mean_mps2"][k] = float(
                _get_required_attr(upd, "predicted_disturbance_mean_mps2")
            )
            out["sequence_predicted_disturbance_std_mps2"][k] = float(
                _get_required_attr(upd, "predicted_disturbance_std_mps2")
            )
            out["sequence_used_gradient"][k] = bool(
                _get_required_attr(upd, "used_gradient")
            )
            out["sequence_viterbi_log_score"][k] = float(
                _get_required_attr(upd, "viterbi_log_score")
            )
            out["sequence_viterbi_offset_ned_m"][k] = _vec3(
                _get_required_attr(upd, "viterbi_offset_ned_m"),
                name="viterbi_offset_ned_m",
            )

        return out

    def integrity_history_arrays(self) -> dict[str, np.ndarray]:
        """
        Extract integrity-monitor history arrays.

        Notes
        -----
        This method only extracts the most important scalar diagnostics so it can
        remain robust even if the integrity structures grow later.
        """
        n = len(self.integrity_snapshots)
        out: dict[str, np.ndarray] = {
            "integrity_time_s": np.full(n, np.nan, dtype=np.float64),
            "integrity_horizontal_protection_m": np.full(n, np.nan, dtype=np.float64),
            "integrity_vertical_protection_m": np.full(n, np.nan, dtype=np.float64),
            "integrity_radial_protection_m": np.full(n, np.nan, dtype=np.float64),
            "integrity_horizontal_error_m": np.full(n, np.nan, dtype=np.float64),
            "integrity_vertical_error_m": np.full(n, np.nan, dtype=np.float64),
            "integrity_hmi_horizontal": np.empty(n, dtype=bool),
            "integrity_hmi_vertical": np.empty(n, dtype=bool),
        }

        for k, snap in enumerate(self.integrity_snapshots):
            out["integrity_time_s"][k] = _safe_float_attr(snap, "time_s", np.nan)
            pls = _get_required_attr(snap, "protection_levels")
            out["integrity_horizontal_protection_m"][k] = float(
                _get_required_attr(pls, "horizontal_m")
            )
            out["integrity_vertical_protection_m"][k] = float(
                _get_required_attr(pls, "vertical_m")
            )
            out["integrity_radial_protection_m"][k] = float(
                _get_required_attr(pls, "radial_3d_m")
            )
            out["integrity_horizontal_error_m"][k] = _safe_float_attr(
                snap, "horizontal_error_m", np.nan
            )
            out["integrity_vertical_error_m"][k] = _safe_float_attr(
                snap, "vertical_error_m", np.nan
            )
            out["integrity_hmi_horizontal"][k] = _safe_bool_attr(
                snap,
                "hazardously_misleading_horizontal",
                False,
            )
            out["integrity_hmi_vertical"][k] = _safe_bool_attr(
                snap,
                "hazardously_misleading_vertical",
                False,
            )

        return out


# -----------------------------------------------------------------------------
# Top-level scenario/run result container
# -----------------------------------------------------------------------------


@dataclass
class ScenarioSimulationResult:
    """
    Complete result of one scenario simulation run.

    Attributes
    ----------
    truth : TruthTrajectory
        Canonical truth trajectory.
    sensors : SimulationSensorLog
        Logged sensor outputs.
    estimators : SimulationEstimatorLog
        Logged estimator outputs.
    metadata : SimulationMetadata
        Run metadata.

    Notes
    -----
    This is the central payload that the runner should return once the simulation
    package is fully wired up.
    """

    truth: TruthTrajectory
    sensors: SimulationSensorLog = field(default_factory=SimulationSensorLog)
    estimators: SimulationEstimatorLog = field(default_factory=SimulationEstimatorLog)
    metadata: SimulationMetadata = field(default_factory=SimulationMetadata)

    def __len__(self) -> int:
        """Number of truth samples."""
        return len(self.truth)

    @property
    def duration_s(self) -> float:
        """
        Duration of the truth trajectory [s].

        Returns
        -------
        float
            `time_s[-1] - time_s[0]`, or 0 for degenerate inputs.
        """
        if len(self.truth) == 0:
            return 0.0
        return float(self.truth.time_s[-1] - self.truth.time_s[0])

    def truth_time_s(self) -> FloatArray:
        """Truth time history [s]."""
        return _time_array(self.truth.time_s, name="truth.time_s").copy()

    def truth_geodetic(self) -> FloatArray:
        """
        Truth geodetic history as `[lat, lon, h]`.
        """
        return np.column_stack(
            [
                self.truth.lat_rad,
                self.truth.lon_rad,
                self.truth.height_m,
            ]
        ).astype(np.float64)

    def truth_velocity_ned(self) -> FloatArray:
        """Truth NED velocity history [m/s]."""
        return np.asarray(self.truth.v_ned_mps, dtype=np.float64).copy()

    def truth_velocity_derivative_ned(self) -> FloatArray:
        """Truth NED velocity derivative history [m/s^2]."""
        return np.asarray(self.truth.v_dot_ned_mps2, dtype=np.float64).copy()

    def truth_attitude_dcm(self) -> FloatArray:
        """Truth body->NED DCM history."""
        return np.asarray(self.truth.C_n_b, dtype=np.float64).copy()

    def truth_body_rate(self) -> FloatArray:
        """Truth body rate with respect to NED, resolved in body [rad/s]."""
        return np.asarray(self.truth.omega_nb_b_radps, dtype=np.float64).copy()

    def summary(self) -> ScenarioSimulationSummary:
        """
        Build a compact summary of the run.
        """
        sensor_counts = self.sensors.counts()
        estimator_counts = self.estimators.counts()

        return ScenarioSimulationSummary(
            scenario_name=self.metadata.scenario_name,
            run_id=self.metadata.run_id,
            duration_s=self.duration_s,
            num_truth_samples=len(self.truth),
            num_imu_samples=int(sensor_counts.get("imu", 0)),
            num_gravimeter_samples=int(sensor_counts.get("gravimeter", 0)),
            num_depth_samples=int(sensor_counts.get("depth", 0)),
            num_velocity_aid_samples=int(sensor_counts.get("velocity_aid", 0)),
            num_ins_states=int(estimator_counts.get("ins_states", 0)),
            num_pf_updates=int(estimator_counts.get("pf_updates", 0)),
            num_sequence_updates=int(estimator_counts.get("sequence_updates", 0)),
            num_integrity_snapshots=int(
                estimator_counts.get("integrity_snapshots", 0)
            ),
        )

    def summary_mapping(self) -> dict[str, Any]:
        """Return the run summary as a JSON-friendly mapping."""
        return self.summary().to_mapping()

    def archive_arrays(self) -> dict[str, np.ndarray]:
        """
        Return a flat mapping of NumPy-friendly arrays suitable for NPZ export.

        Included groups
        ---------------
        - truth arrays
        - sensor history arrays
        - estimator history arrays
        """
        out: dict[str, np.ndarray] = {
            "truth_time_s": np.asarray(self.truth.time_s, dtype=np.float64),
            "truth_lat_rad": np.asarray(self.truth.lat_rad, dtype=np.float64),
            "truth_lon_rad": np.asarray(self.truth.lon_rad, dtype=np.float64),
            "truth_height_m": np.asarray(self.truth.height_m, dtype=np.float64),
            "truth_v_ned_mps": np.asarray(self.truth.v_ned_mps, dtype=np.float64),
            "truth_v_dot_ned_mps2": np.asarray(
                self.truth.v_dot_ned_mps2, dtype=np.float64
            ),
            "truth_C_n_b": np.asarray(self.truth.C_n_b, dtype=np.float64),
            "truth_omega_nb_b_radps": np.asarray(
                self.truth.omega_nb_b_radps, dtype=np.float64
            ),
        }

        if len(self.sensors.imu_samples) > 0:
            out.update(self.sensors.imu_history_arrays())
        if len(self.sensors.gravimeter_samples) > 0:
            out.update(self.sensors.gravimeter_history_arrays())
        if len(self.sensors.depth_samples) > 0:
            out.update(self.sensors.depth_history_arrays())
        if len(self.sensors.velocity_aid_samples) > 0:
            out.update(self.sensors.velocity_aid_history_arrays())

        if len(self.estimators.ins_states) > 0:
            out.update(self.estimators.ins_history_arrays())
        if len(self.estimators.pf_updates) > 0:
            out.update(self.estimators.pf_history_arrays())
        if len(self.estimators.sequence_updates) > 0:
            out.update(self.estimators.sequence_history_arrays())
        if len(self.estimators.integrity_snapshots) > 0:
            out.update(self.estimators.integrity_history_arrays())

        return out

    def save_summary_json(
        self,
        path: str | Path,
        *,
        indent: int = 2,
    ) -> Path:
        """
        Save metadata + summary as a JSON file.

        Parameters
        ----------
        path : str or pathlib.Path
            Output JSON path.
        indent : int, default=2
            Pretty-print indent level.

        Returns
        -------
        pathlib.Path
            Resolved output path.
        """
        p = _ensure_output_path(path)
        payload = {
            "metadata": self.metadata.to_mapping(),
            "summary": self.summary_mapping(),
            "sensor_counts": _jsonable(self.sensors.counts()),
            "estimator_counts": _jsonable(self.estimators.counts()),
        }
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent, sort_keys=False)
        return p

    def save_npz(
        self,
        path: str | Path,
    ) -> Path:
        """
        Save the run to a compressed `.npz` archive.

        Parameters
        ----------
        path : str or pathlib.Path
            Output archive path.

        Returns
        -------
        pathlib.Path
            Resolved output path.

        Notes
        -----
        The archive contains:
        - numeric/string arrays from `archive_arrays()`
        - JSON blobs for metadata, summary, and custom streams
        """
        p = _ensure_output_path(path)
        arrays = self.archive_arrays()

        metadata_json = json.dumps(self.metadata.to_mapping(), sort_keys=False)
        summary_json = json.dumps(self.summary_mapping(), sort_keys=False)
        sensor_counts_json = json.dumps(_jsonable(self.sensors.counts()), sort_keys=False)
        estimator_counts_json = json.dumps(
            _jsonable(self.estimators.counts()),
            sort_keys=False,
        )
        sensor_custom_streams_json = json.dumps(
            _jsonable(self.sensors.custom_streams),
            sort_keys=False,
        )
        estimator_custom_streams_json = json.dumps(
            _jsonable(self.estimators.custom_streams),
            sort_keys=False,
        )

        np.savez_compressed(
            p,
            **arrays,
            metadata_json=np.array(metadata_json),
            summary_json=np.array(summary_json),
            sensor_counts_json=np.array(sensor_counts_json),
            estimator_counts_json=np.array(estimator_counts_json),
            sensor_custom_streams_json=np.array(sensor_custom_streams_json),
            estimator_custom_streams_json=np.array(estimator_custom_streams_json),
        )
        return p


# -----------------------------------------------------------------------------
# Loader
# -----------------------------------------------------------------------------


def load_npz_archive(path: str | Path) -> dict[str, Any]:
    """
    Load an archive produced by :meth:`ScenarioSimulationResult.save_npz`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the `.npz` archive.

    Returns
    -------
    dict[str, Any]
        Plain mapping containing arrays and parsed JSON blobs.

    Notes
    -----
    This function intentionally returns a lightweight mapping rather than trying
    to reconstruct every rich dataclass in the simulator. That keeps the loader
    stable even if the result schema grows over time.
    """
    p = Path(path).expanduser().resolve()
    out: dict[str, Any] = {}

    with np.load(p, allow_pickle=False) as data:
        for key in data.files:
            value = np.asarray(data[key])
            if key.endswith("_json"):
                continue
            if value.ndim == 0:
                try:
                    out[key] = value.item()
                except ValueError:
                    out[key] = value
            else:
                out[key] = value.copy()

        if "metadata_json" in data:
            out["metadata"] = json.loads(str(np.asarray(data["metadata_json"]).item()))
        if "summary_json" in data:
            out["summary"] = json.loads(str(np.asarray(data["summary_json"]).item()))
        if "sensor_counts_json" in data:
            out["sensor_counts"] = json.loads(
                str(np.asarray(data["sensor_counts_json"]).item())
            )
        if "estimator_counts_json" in data:
            out["estimator_counts"] = json.loads(
                str(np.asarray(data["estimator_counts_json"]).item())
            )
        if "sensor_custom_streams_json" in data:
            out["sensor_custom_streams"] = json.loads(
                str(np.asarray(data["sensor_custom_streams_json"]).item())
            )
        if "estimator_custom_streams_json" in data:
            out["estimator_custom_streams"] = json.loads(
                str(np.asarray(data["estimator_custom_streams_json"]).item())
            )

    return out


__all__ = [
    "FloatArray",
    "ScenarioSimulationResult",
    "ScenarioSimulationSummary",
    "SimulationEstimatorLog",
    "SimulationMetadata",
    "SimulationSensorLog",
    "extract_scalar_measurement_history",
    "extract_vector_measurement_history",
    "load_npz_archive",
]
