"""
base.py

Shared sensor-model abstractions and reusable helper utilities for the
gravity-aided navigation simulator.

This module is intentionally the common foundation for concrete sensor models
such as:
- IMU
- scalar gravimeter
- depth sensor
- velocity aiding sensor

Why this file exists
--------------------
The repository already contains stateful concrete sensor models that follow the
same high-level pattern:
- a specification object
- an owned random-number generator
- an internal state / bias state
- - a `current_bias_state()`-style accessor
- optional saturation and filtering
- optional timestamps on returned measurements

That shared shape is visible in the existing IMU and gravimeter modules, while
`src/gravnav/sensors/base.py` is still empty. This file turns that repeated
structure into reusable building blocks so the remaining sensor modules do not
have to re-implement the same glue code. :contentReference[oaicite:3]{index=3}

Primary references used here
----------------------------
1) Python official `abc` documentation
   https://docs.python.org/3/library/abc.html

   Used for:
   - defining abstract base classes
   - using `@abstractmethod` to require subclasses to implement core methods

2) Python official `dataclasses` documentation
   https://docs.python.org/3/library/dataclasses.html

   Used for:
   - lightweight immutable-ish containers for specs and measurements
   - explicit typed data containers rather than ad hoc dictionaries

3) NumPy Random Generator documentation
   https://numpy.org/doc/stable/reference/random/generator.html

   Used for:
   - the modern `numpy.random.Generator` interface
   - consistency with the repository's `utils.rng` policy and the rest of the
     sensor stack

Design notes
------------
- This file is intentionally generic: it contains no IMU-specific or
  gravimeter-specific physics.
- It centralizes the boring but important parts:
  validation, saturation, discrete-noise scaling, first-order filtering, and
  RNG ownership.
- The concrete sensor models remain responsible for their own measurement
  physics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..utils.rng import (
    RNGStateRecord,
    generator_state_record,
    make_rng,
    restore_generator_from_state,
)

FloatArray = NDArray[np.float64]

SpecT = TypeVar("SpecT")
StateT = TypeVar("StateT")


# -----------------------------------------------------------------------------
# Low-level numeric helpers
# -----------------------------------------------------------------------------


def _as_float_array(x: ArrayLike) -> FloatArray:
    """Convert input to a NumPy float64 array."""
    return np.asarray(x, dtype=np.float64)


def scalar(x: ArrayLike, *, name: str) -> float:
    """
    Validate and return a scalar float.

    Parameters
    ----------
    x : array-like or scalar
        Input to validate.
    name : str
        Name used in error messages.

    Returns
    -------
    float
        Scalar value.

    Raises
    ------
    ValueError
        If the input is not scalar-like.
    """
    arr = _as_float_array(x)
    if arr.ndim != 0:
        raise ValueError(f"{name} must be scalar-like, got shape {arr.shape}.")
    return float(arr)


def vec3(x: ArrayLike, *, name: str) -> FloatArray:
    """
    Validate and return a 3-vector of shape `(3,)`.
    """
    arr = _as_float_array(x).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be shape (3,), got {arr.shape}.")
    return arr


def axis3(x: ArrayLike | float, *, name: str) -> FloatArray:
    """
    Convert a scalar or shape-(3,) input into a shape-(3,) vector.

    Parameters
    ----------
    x : scalar or array-like
        Either one scalar value to be broadcast to all three axes, or an
        explicit 3-vector.
    name : str
        Name used in error messages.

    Returns
    -------
    np.ndarray, shape (3,)
        Per-axis vector.

    Notes
    -----
    This is especially convenient for sensor specs where a datasheet may give:
    - one scalar value used identically on all axes, or
    - a separate value for each axis.
    """
    arr = _as_float_array(x)
    if arr.ndim == 0:
        return np.full(3, float(arr), dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be scalar or shape (3,), got {arr.shape}.")
    return arr


def nonnegative_axis3(x: ArrayLike | float, *, name: str) -> FloatArray:
    """
    Like `axis3(...)`, but require all entries to be nonnegative.
    """
    arr = axis3(x, name=name)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative, got {arr}.")
    return arr


def positive_or_inf_axis3(x: ArrayLike | float, *, name: str) -> FloatArray:
    """
    Convert a scalar or 3-vector into a per-axis limit vector and require each
    entry to be positive or +inf.

    Useful for saturation limits.
    """
    arr = axis3(x, name=name)
    if np.any((arr <= 0.0) & ~np.isinf(arr)):
        raise ValueError(f"{name} must contain positive values or inf, got {arr}.")
    return arr


def clip_scalar(value: float, max_abs: float) -> tuple[float, bool]:
    """
    Clip a scalar symmetrically to `[-max_abs, +max_abs]`.

    Parameters
    ----------
    value : float
        Input scalar.
    max_abs : float
        Maximum absolute allowed magnitude. Must be positive or `np.inf`.

    Returns
    -------
    tuple[float, bool]
        `(clipped_value, saturated_flag)`.
    """
    limit = float(max_abs)
    if limit <= 0.0 and not np.isinf(limit):
        raise ValueError(f"max_abs must be positive or inf, got {limit}.")
    clipped = float(np.clip(float(value), -limit, limit))
    saturated = not np.isclose(clipped, float(value))
    return clipped, saturated


def clip_vector_per_axis(
    value: ArrayLike,
    max_abs: ArrayLike | float,
) -> tuple[FloatArray, bool]:
    """
    Clip a 3-vector independently on each axis.

    Parameters
    ----------
    value : array-like, shape (3,)
        Input vector.
    max_abs : scalar or array-like, shape (3,)
        Per-axis maximum absolute magnitudes.

    Returns
    -------
    tuple[np.ndarray, bool]
        `(clipped_vector, saturated_flag)`.

    Notes
    -----
    The saturation flag is True if clipping occurred on any axis.
    """
    v = vec3(value, name="value")
    lim = positive_or_inf_axis3(max_abs, name="max_abs")
    clipped = np.clip(v, -lim, lim)
    saturated = not np.allclose(clipped, v)
    return clipped.astype(np.float64), bool(saturated)


def standard_normal3(rng: np.random.Generator) -> FloatArray:
    """
    Return a standard-normal 3-vector.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source generator.

    Returns
    -------
    np.ndarray, shape (3,)
        One draw from N(0, I_3).
    """
    return rng.standard_normal(3, dtype=np.float64)


# -----------------------------------------------------------------------------
# Generic stochastic scaling helpers
# -----------------------------------------------------------------------------


def discrete_white_noise_std_from_density(
    noise_density_per_sqrt_hz: ArrayLike | float,
    dt_s: float,
) -> FloatArray:
    r"""
    Convert continuous-time white-noise density to discrete-time sample standard
    deviation.

    Parameters
    ----------
    noise_density_per_sqrt_hz : scalar or shape (3,)
        White-noise density in `[units]/sqrt(Hz)`.
    dt_s : float
        Sample interval [s].

    Returns
    -------
    np.ndarray, shape (3,)
        Discrete-time per-sample standard deviation in the same physical units.

    Formula
    -------
    The common engineering discretization is:

        sigma_d = sigma / sqrt(dt)

    assuming ideal anti-alias filtering / ideal decimation prior to sampling.

    Notes
    -----
    This helper is generic and unit-agnostic; examples include:
    - gyro noise density in rad/s/sqrt(Hz)
    - accelerometer noise density in m/s^2/sqrt(Hz)
    - velocity-aid noise density in m/s/sqrt(Hz)
    - depth noise density in m/sqrt(Hz)

    The concrete unit interpretation belongs to the calling sensor model.
    """
    dt = float(dt_s)
    if dt <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt}.")
    sigma = nonnegative_axis3(
        noise_density_per_sqrt_hz,
        name="noise_density_per_sqrt_hz",
    )
    return sigma / np.sqrt(dt)


def discrete_random_walk_step_std(
    random_walk_per_sqrt_s: ArrayLike | float,
    dt_s: float,
) -> FloatArray:
    r"""
    Convert a random-walk coefficient into the standard deviation of the
    one-step bias increment.

    Parameters
    ----------
    random_walk_per_sqrt_s : scalar or shape (3,)
        Random-walk coefficient in `[bias units]/sqrt(s)`.
    dt_s : float
        Sample interval [s].

    Returns
    -------
    np.ndarray, shape (3,)
        Standard deviation of the one-step increment.

    Formula
    -------
    Under the standard Brownian / Wiener discretization:

        b_k = b_{k-1} + sigma_rw * sqrt(dt) * w_k
        w_k ~ N(0, I)

    therefore:

        sigma_step = sigma_rw * sqrt(dt)

    Notes
    -----
    This helper is again unit-agnostic and can be used by any sensor whose bias
    or drift state is modeled as a random walk.
    """
    dt = float(dt_s)
    if dt <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt}.")
    sigma = nonnegative_axis3(
        random_walk_per_sqrt_s,
        name="random_walk_per_sqrt_s",
    )
    return sigma * np.sqrt(dt)


# -----------------------------------------------------------------------------
# First-order low-pass helpers
# -----------------------------------------------------------------------------


def first_order_lpf_alpha(
    cutoff_hz: float,
    dt_s: float,
) -> float:
    r"""
    Return the exact zero-order-hold first-order low-pass smoothing coefficient.

    Parameters
    ----------
    cutoff_hz : float
        3 dB cutoff frequency [Hz].
    dt_s : float
        Sample interval [s].

    Returns
    -------
    float
        Smoothing coefficient α.

    Formula
    -------
    For a first-order low-pass with time constant:

        tau = 1 / (2 pi f_c)

    and zero-order-hold discretization over `dt`, the exact update coefficient is:

        α = 1 - exp(-dt / tau)
          = 1 - exp(-2 pi f_c dt)

    so the recursive update is:

        y_k = y_{k-1} + α (x_k - y_{k-1})

    Notes
    -----
    This is the same simple bandwidth model already useful for the gravimeter
    and likely to be useful for future low-rate aiding sensors.
    """
    fc = float(cutoff_hz)
    dt = float(dt_s)
    if fc <= 0.0:
        raise ValueError(f"cutoff_hz must be positive, got {fc}.")
    if dt <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt}.")
    return float(1.0 - np.exp(-2.0 * np.pi * fc * dt))


def first_order_lpf_step(
    previous_state: ArrayLike | float | None,
    input_value: ArrayLike | float,
    *,
    dt_s: float,
    cutoff_hz: float | None,
):
    r"""
    Advance a first-order low-pass filter by one sample.

    Parameters
    ----------
    previous_state : scalar, array-like, or None
        Previous filter output. If None, the input is passed through directly and
        becomes the initialized state.
    input_value : scalar or array-like
        Current raw input value.
    dt_s : float
        Sample interval [s].
    cutoff_hz : float or None
        Cutoff frequency [Hz]. If None, the function behaves as a passthrough.

    Returns
    -------
    same shape as input_value
        Updated filter output / new state.

    Update rule
    -----------
    If `cutoff_hz is None`:
        y_k = x_k

    Else:
        y_k = y_{k-1} + α (x_k - y_{k-1})
        α   = 1 - exp(-2 pi f_c dt)

    Notes
    -----
    The caller can simply store the returned value as the next state.
    """
    x = _as_float_array(input_value)

    if cutoff_hz is None:
        y = x
    elif previous_state is None:
        y = x
    else:
        prev = _as_float_array(previous_state)
        if prev.shape != x.shape:
            raise ValueError(
                f"previous_state must match input_value shape {x.shape}, got {prev.shape}."
            )
        alpha = first_order_lpf_alpha(float(cutoff_hz), float(dt_s))
        y = prev + alpha * (x - prev)

    if y.ndim == 0:
        return float(y)
    return np.asarray(y, dtype=np.float64)


# -----------------------------------------------------------------------------
# Generic data containers
# -----------------------------------------------------------------------------


@dataclass
class SensorSpecBase:
    """
    Minimal base specification shared by all sensors.

    Attributes
    ----------
    name : str
        Human-readable sensor identifier.

    Notes
    -----
    Concrete spec classes are free to add any additional fields they need.
    """
    name: str = "sensor"


@dataclass
class SensorMeasurementBase:
    """
    Minimal base measurement shared by all sensors.

    Attributes
    ----------
    time_s : float or None
        Optional timestamp [s].
    saturated : bool
        True if saturation occurred while producing this measurement.
    """
    time_s: Optional[float]
    saturated: bool = False


@dataclass
class ScalarSensorMeasurementBase(SensorMeasurementBase):
    """
    Base class for scalar sensor measurements.

    Attributes
    ----------
    value : float
        Final scalar measurement value.
    ideal_value : float or None
        Optional ideal noiseless value before corruption.
    """
    value: float = 0.0
    ideal_value: Optional[float] = None


@dataclass
class VectorSensorMeasurementBase(SensorMeasurementBase):
    """
    Base class for 3-axis vector sensor measurements.

    Attributes
    ----------
    value : np.ndarray, shape (3,)
        Final vector measurement.
    ideal_value : np.ndarray, shape (3,), optional
        Optional ideal noiseless value before corruption.
    """
    value: FloatArray = None  # type: ignore[assignment]
    ideal_value: Optional[FloatArray] = None

    def __post_init__(self) -> None:
        self.value = vec3(self.value, name="value")
        if self.ideal_value is not None:
            self.ideal_value = vec3(self.ideal_value, name="ideal_value")


# -----------------------------------------------------------------------------
# Abstract stateful sensor base
# -----------------------------------------------------------------------------


class StatefulSensorBase(ABC, Generic[SpecT, StateT]):
    """
    Abstract base class for stateful simulated sensors.

    Parameters
    ----------
    spec : SpecT
        Sensor specification object.
    rng : numpy.random.Generator, optional
        Optional externally supplied random-number generator. If omitted, a new
        generator is created using the repository RNG policy.

    State owned by this base class
    ------------------------------
    - `spec`
    - `rng`

    Required subclass API
    ---------------------
    Subc
    - `reset(...)`
    - `current_bias_state()`

    Why this base class exists
    --------------------------
    The IMU and gravimeter modules already follow this same pattern: a stateful
    sensor object owns a specification, an RNG, and an internal bias/filter state,
    and exposes reset/state-inspection methods. This ABC makes that contract
    explicit for the rest of the sensor:contentReference[oaicite:6]{index=6}
    """

    def __init__(
        self,
        spec: SpecT,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.spec = spec
        self.rng = make_rng(rng)

    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any) -> None:
        """
        Reset the internal sensor state.

        Concrete sensors decide what that means, for example:
        - sampling a new turn-on bias
        - zeroing a bias state
        - resetting a low-pass filter state
        """

    @abstractmethod
    def current_bias_state(self) -> StateT:
        """
        Return a copy of the current active bias/state relevant to the sensor.
        """

    def rng_state_record(self) -> RNGStateRecord:
        """
        Return a serializable snapshot of the sensor RNG state.

        This is useful for exact replay of stochastic simulations.
        """
        return generator_state_record(self.rng)

    def restore_rng_state(self, state: RNGStateRecord | dict[str, Any]) -> None:
        """
        Restore the sensor RNG from a previously saved state record.
        """
        self.rng = restore_generator_from_state(state)

    def reseed(
        self,
        rng: Optional[np.random.Generator | int] = None,
    ) -> None:
        """
        Replace the sensor RNG with a new generator.

        Parameters
        ----------
        rng : numpy.random.Generator, int, or None
            New RNG source. If None, a fresh default generator is created.

        Notes
        -----
        This helper is convenient in experiments where you want to:
        - reuse the same spec/state logic
        - but restart the stochastic stream
        """
        self.rng = make_rng(rng)


__all__ = [
    "FloatArray",
    "SensorSpecBase",
    "SensorMeasurementBase",
    "ScalarSensorMeasurementBase",
    "VectorSensorMeasurementBase",
    "StatefulSensorBase",
    "axis3",
    "clip_scalar",
    "clip_vector_per_axis",
    "discrete_random_walk_step_std",
    "discrete_white_noise_std_from_density",
    "first_order_lpf_alpha",
    "first_order_lpf_step",
    "nonnegative_axis3",
    "positive_or_inf_axis3",
    "scalar",
    "standard_normal3",
    "vec3",
]