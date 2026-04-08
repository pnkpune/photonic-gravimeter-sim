"""
scenarios.py

Declarative scenario specifications and named mission presets for the
gravity-aided navigation simulator.

This module is the orchestration layer above:
- `truth.vehicle_models`      : motion primitives and profile builders
- `truth.trajectory`          : validated truth-trajectory containers

Its job is to let the rest of the repository talk in terms of:
- "maritime baseline"
- "UUV long-endurance survey"
- "UAV stress case"
- or a custom list of straight legs and turns

instead of manually stitching together motion primitives every time.

Why this file exists
--------------------
The repository structure already reserves:
- `configs/scenarios/*.yaml`
- `scripts/run_single_scenario.py`
- `scripts/run_monte_carlo.py`

and the truth stack below this layer already exists conceptually:
- `trajectory.py` owns the canonical truth-state container
- `vehicle_models.py` owns the kinematic motion primitives

Therefore `scenarios.py` should be the single place that:
1. defines a compact declarative scenario schema,
2. translates segment specs into vehicle-model profiles,
3. builds full `TruthTrajectory` objects from those profiles,
4. exposes a small registry of named default scenarios.

Primary references used here
----------------------------
1) NOAA Ocean Exploration, "Tracks Showing a Lawnmower Pattern"
   URL:
   https://oceanexplorer.noaa.gov/multimedia/okeanos-explorations-seascape-alaska-ex2302-features-logan-updates-media-lawnmower-at/

   Used for:
   - the survey-style "lawnmower" / back-and-forth track concept
   - motivating the maritime/UUV baseline presets as repeated straight survey
     legs connected by turns rather than arbitrary random motion

2) NOAA Ocean Exploration, "Live From the Field: Updates from Logan"
   URL:
   https://oceanexplorer.noaa.gov/expedition-feature/okeanos-seascape-alaska-ex2302-features-logan-updates/

   Used for the concise description that a back-and-forth "lawnmower transect"
   is a common mapping technique for collecting high-resolution coverage data.

3) NOAA / UNH Center for Coastal and Ocean Mapping,
   "Hydrographic Survey with Autonomous Surface Vehicles"
   URL:
   https://repository.library.noaa.gov/view/noaa/29172/noaa_29172_DS1.pdf

   Used for the statement that hydrographic survey commonly follows systematic
   lawnmower-style lines rather than general transit navigation.

4) NASA Glenn Research Center, "Banking Turns"
   URL:
   https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/banking-turns/

   Used to motivate the scenario turn primitives being expressed through banked
   heading changes.

5) MIT OpenCourseWare 16.333, Lecture 12
   URL:
   https://ocw.mit.edu/courses/16-333-aircraft-stability-and-control-fall-2004/03fb92311b62c652e222e14efb49573b_lecture_12.pdf

   Used for the coordinated-turn relation already implemented in
   `vehicle_models.py` and therefore reused here through the vehicle-model API.

Design notes
------------
- This file intentionally introduces *no new navigation physics*.
  It is an orchestration/configuration layer.
- The actual kinematics remain in `vehicle_models.py`.
- The geodetic propagation remains in `trajectory.py`.
- The scenario presets are deliberately plausible and useful for testing:
  they are not tuned to a specific real vehicle unless and until later config
  files say so explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..physics.earth import normal_gravity
from .trajectory import TruthTrajectory
from .vehicle_models import (
    VehicleKinematicProfile,
    build_truth_trajectory_from_profile,
    concatenate_profiles,
    make_constant_rate_turn_profile,
    make_coordinated_turn_profile,
    make_smooth_coordinated_turn_profile,
    make_straight_profile,
)

FloatArray = NDArray[np.float64]


def _as_float_array(x: Any) -> FloatArray:
    """Convert input to a NumPy float64 array."""
    return np.asarray(x, dtype=np.float64)


def _wrap_angle_pi(angle_rad: float | FloatArray) -> float | FloatArray:
    """Wrap angle(s) to [-pi, pi)."""
    ang = _as_float_array(angle_rad)
    wrapped = (ang + np.pi) % (2.0 * np.pi) - np.pi
    if wrapped.ndim == 0:
        return float(wrapped)
    return np.asarray(wrapped, dtype=np.float64)


def _require_positive(name: str, value: float) -> float:
    """Require a scalar to be strictly positive."""
    val = float(value)
    if val <= 0.0:
        raise ValueError(f"{name} must be positive, got {val}.")
    return val


def _require_nonnegative(name: str, value: float) -> float:
    """Require a scalar to be nonnegative."""
    val = float(value)
    if val < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {val}.")
    return val


# -----------------------------------------------------------------------------
# Scenario segment specifications
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class StraightSegmentSpec:
    """
    Straight constant-speed segment.

    Parameters
    ----------
    duration_s : float
        Segment duration [s].
    speed_mps : float
        Constant speed [m/s].
    flight_path_angle_rad : float, default=0.0
        Constant flight-path angle [rad]. Positive means climbing.
    roll_rad : float, default=0.0
        Constant roll angle [rad].
    label : str, default="straight"
        Optional human-readable label.

    Notes
    -----
    Heading is inherited from the scenario state at the start of the segment and
    remains constant throughout the segment.
    """

    duration_s: float
    speed_mps: float
    flight_path_angle_rad: float = 0.0
    roll_rad: float = 0.0
    label: str = "straight"

    def __post_init__(self) -> None:
        _require_positive("duration_s", self.duration_s)
        _require_nonnegative("speed_mps", self.speed_mps)


@dataclass(frozen=True)
class ConstantRateTurnSegmentSpec:
    """
    Constant-speed turn specified directly by heading rate.

    Parameters
    ----------
    duration_s : float
        Segment duration [s].
    speed_mps : float
        Constant speed [m/s].
    heading_rate_radps : float
        Constant heading rate [rad/s]. Positive means right turn.
    flight_path_angle_rad : float, default=0.0
        Constant flight-path angle [rad].
    roll_rad : float or None, default=None
        Optional explicit roll angle [rad]. If None, the corresponding
        coordinated-turn bank angle is computed by `vehicle_models.py`.
    label : str, default="constant_rate_turn"
        Optional human-readable label.
    """

    duration_s: float
    speed_mps: float
    heading_rate_radps: float
    flight_path_angle_rad: float = 0.0
    roll_rad: float | None = None
    label: str = "constant_rate_turn"

    def __post_init__(self) -> None:
        _require_positive("duration_s", self.duration_s)
        _require_positive("speed_mps", self.speed_mps)


@dataclass(frozen=True)
class CoordinatedTurnSegmentSpec:
    """
    Coordinated turn specified by bank angle, with optional smooth roll-in/out.

    Parameters
    ----------
    duration_s : float
        Total segment duration [s].
    speed_mps : float
        Constant speed [m/s].
    bank_angle_rad : float
        Target steady bank angle [rad]. Positive means right turn.
    flight_path_angle_rad : float, default=0.0
        Constant flight-path angle [rad].
    roll_in_duration_s : float, default=0.0
        Smooth roll-in duration [s].
    roll_out_duration_s : float, default=0.0
        Smooth roll-out duration [s].
    smooth : bool, default=True
        If True, use `make_smooth_coordinated_turn_profile(...)`. If False, use
        an instantaneous-entry steady turn profile.
    gravity_mps2 : float or None, default=None
        Optional reference gravity to use for the coordinated-turn relation. If
        None, the scenario builder uses the initial-point normal gravity.
    label : str, default="coordinated_turn"
        Optional human-readable label.
    """

    duration_s: float
    speed_mps: float
    bank_angle_rad: float
    flight_path_angle_rad: float = 0.0
    roll_in_duration_s: float = 0.0
    roll_out_duration_s: float = 0.0
    smooth: bool = True
    gravity_mps2: float | None = None
    label: str = "coordinated_turn"

    def __post_init__(self) -> None:
        _require_positive("duration_s", self.duration_s)
        _require_positive("speed_mps", self.speed_mps)
        _require_nonnegative("roll_in_duration_s", self.roll_in_duration_s)
        _require_nonnegative("roll_out_duration_s", self.roll_out_duration_s)
        if self.roll_in_duration_s + self.roll_out_duration_s > self.duration_s + 1e-12:
            raise ValueError(
                "roll_in_duration_s + roll_out_duration_s must not exceed duration_s."
            )
        if self.gravity_mps2 is not None:
            _require_positive("gravity_mps2", self.gravity_mps2)


SegmentSpec: TypeAlias = (
    StraightSegmentSpec
    | ConstantRateTurnSegmentSpec
    | CoordinatedTurnSegmentSpec
)


# -----------------------------------------------------------------------------
# Scenario specification container
# -----------------------------------------------------------------------------


@dataclass
class ScenarioSpec:
    """
    Declarative scenario specification.

    Parameters
    ----------
    name : str
        Scenario identifier.
    initial_lat_rad : float
        Initial geodetic latitude [rad].
    initial_lon_rad : float
        Initial longitude [rad].
    initial_height_m : float
        Initial ellipsoidal height [m].
    initial_heading_rad : float
        Initial heading [rad], clockwise from North toward East.
    segments : sequence of SegmentSpec
        Ordered segment list.
    default_dt_s : float, default=1.0
        Default sample interval [s].
    description : str, default=""
        Optional human-readable description.
    metadata : dict, default_factory=dict
        Optional free-form metadata.
    """

    name: str
    initial_lat_rad: float
    initial_lon_rad: float
    initial_height_m: float
    initial_heading_rad: float
    segments: Sequence[SegmentSpec]
    default_dt_s: float = 1.0
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.initial_lat_rad = float(self.initial_lat_rad)
        self.initial_lon_rad = float(_wrap_angle_pi(self.initial_lon_rad))
        self.initial_height_m = float(self.initial_height_m)
        self.initial_heading_rad = float(_wrap_angle_pi(self.initial_heading_rad))
        self.default_dt_s = _require_positive("default_dt_s", self.default_dt_s)

        if abs(self.initial_lat_rad) > 0.5 * np.pi + 1e-12:
            raise ValueError("initial_lat_rad must lie within [-pi/2, pi/2].")
        if not self.segments:
            raise ValueError("segments must contain at least one segment.")

        validated_segments: list[SegmentSpec] = []
        for seg in self.segments:
            if not isinstance(
                seg,
                (
                    StraightSegmentSpec,
                    ConstantRateTurnSegmentSpec,
                    CoordinatedTurnSegmentSpec,
                ),
            ):
                raise TypeError(
                    "Each segment must be a StraightSegmentSpec, "
                    "ConstantRateTurnSegmentSpec, or CoordinatedTurnSegmentSpec."
                )
            validated_segments.append(seg)

        self.segments = tuple(validated_segments)

    @property
    def initial_gravity_mps2(self) -> float:
        """
        WGS 84 normal gravity at the scenario's initial point [m/s^2].

        This is a useful default for coordinated-turn bank/rate conversions in
        later builders.
        """
        return float(normal_gravity(self.initial_lat_rad, self.initial_height_m))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ScenarioSpec":
        """
        Build a `ScenarioSpec` from a mapping that mirrors a future YAML/JSON schema.

        Expected top-level keys
        -----------------------
        - name
        - initial_lat_rad or initial_lat_deg
        - initial_lon_rad or initial_lon_deg
        - initial_height_m
        - initial_heading_rad or initial_heading_deg
        - default_dt_s (optional)
        - description (optional)
        - metadata (optional)
        - segments: list[dict]

        Segment mapping formats
        -----------------------
        Each segment dict must contain a `type` key in:
        - "straight"
        - "constant_rate_turn"
        - "coordinated_turn"

        Angles may be supplied either in radians (`*_rad`) or degrees (`*_deg`).
        """
        def read_angle_rad(data: Mapping[str, Any], key_base: str, default: float | None = None) -> float:
            rad_key = f"{key_base}_rad"
            deg_key = f"{key_base}_deg"
            if rad_key in data:
                return float(data[rad_key])
            if deg_key in data:
                return float(np.deg2rad(float(data[deg_key])))
            if default is not None:
                return float(default)
            raise KeyError(f"Missing {rad_key} or {deg_key} in mapping.")

        segments_data = mapping.get("segments")
        if segments_data is None:
            raise KeyError("Scenario mapping must contain a 'segments' field.")
        if not isinstance(segments_data, Sequence) or isinstance(segments_data, (str, bytes)):
            raise TypeError("'segments' must be a sequence of mappings.")

        segments = tuple(_segment_from_mapping(seg) for seg in segments_data)

        return cls(
            name=str(mapping["name"]),
            initial_lat_rad=read_angle_rad(mapping, "initial_lat"),
            initial_lon_rad=read_angle_rad(mapping, "initial_lon"),
            initial_height_m=float(mapping.get("initial_height_m", 0.0)),
            initial_heading_rad=read_angle_rad(mapping, "initial_heading", default=0.0),
            segments=segments,
            default_dt_s=float(mapping.get("default_dt_s", 1.0)),
            description=str(mapping.get("description", "")),
            metadata=dict(mapping.get("metadata", {})),
        )

    def to_mapping(self) -> dict[str, Any]:
        """
        Export the scenario spec to a plain-Python mapping.

        This is useful when you later want a clean bridge to YAML dumps.
        """
        return {
            "name": self.name,
            "initial_lat_rad": self.initial_lat_rad,
            "initial_lon_rad": self.initial_lon_rad,
            "initial_height_m": self.initial_height_m,
            "initial_heading_rad": self.initial_heading_rad,
            "default_dt_s": self.default_dt_s,
            "description": self.description,
            "metadata": dict(self.metadata),
            "segments": [_segment_to_mapping(seg) for seg in self.segments],
        }


def _segment_from_mapping(mapping: Mapping[str, Any]) -> SegmentSpec:
    """
    Parse one segment mapping.

    Supported `type` values:
    - straight
    - constant_rate_turn
    - coordinated_turn
    """
    if not isinstance(mapping, Mapping):
        raise TypeError("Each segment entry must be a mapping.")

    seg_type = str(mapping.get("type", "")).strip().lower()
    if not seg_type:
        raise KeyError("Segment mapping must contain a non-empty 'type' field.")

    def read_angle_rad(key_base: str, default: float | None = None) -> float:
        rad_key = f"{key_base}_rad"
        deg_key = f"{key_base}_deg"
        if rad_key in mapping:
            return float(mapping[rad_key])
        if deg_key in mapping:
            return float(np.deg2rad(float(mapping[deg_key])))
        if default is not None:
            return float(default)
        raise KeyError(f"Missing {rad_key} or {deg_key} for segment type {seg_type!r}.")

    def read_angle_rate_radps(key_base: str) -> float:
        radps_key = f"{key_base}_radps"
        degps_key = f"{key_base}_degps"
        if radps_key in mapping:
            return float(mapping[radps_key])
        if degps_key in mapping:
            return float(np.deg2rad(float(mapping[degps_key])))
        raise KeyError(
            f"Missing {radps_key} or {degps_key} for segment type {seg_type!r}."
        )

    common = {
        "duration_s": float(mapping["duration_s"]),
        "speed_mps": float(mapping["speed_mps"]),
        "flight_path_angle_rad": read_angle_rad("flight_path_angle", default=0.0),
        "label": str(mapping.get("label", seg_type)),
    }

    if seg_type == "straight":
        return StraightSegmentSpec(
            **common,
            roll_rad=read_angle_rad("roll", default=0.0),
        )

    if seg_type == "constant_rate_turn":
        roll_val: float | None
        if "roll_rad" in mapping or "roll_deg" in mapping:
            roll_val = read_angle_rad("roll")
        else:
            roll_val = None

        return ConstantRateTurnSegmentSpec(
            **common,
            heading_rate_radps=read_angle_rate_radps("heading_rate"),
            roll_rad=roll_val,
        )

    if seg_type == "coordinated_turn":
        return CoordinatedTurnSegmentSpec(
            **common,
            bank_angle_rad=read_angle_rad("bank_angle"),
            roll_in_duration_s=float(mapping.get("roll_in_duration_s", 0.0)),
            roll_out_duration_s=float(mapping.get("roll_out_duration_s", 0.0)),
            smooth=bool(mapping.get("smooth", True)),
            gravity_mps2=(
                None if mapping.get("gravity_mps2") is None else float(mapping["gravity_mps2"])
            ),
        )

    raise ValueError(
        f"Unknown segment type {seg_type!r}. Expected one of: "
        "'straight', 'constant_rate_turn', 'coordinated_turn'."
    )


def _segment_to_mapping(segment: SegmentSpec) -> dict[str, Any]:
    """Convert a segment spec back to a plain-Python mapping."""
    if isinstance(segment, StraightSegmentSpec):
        return {
            "type": "straight",
            "duration_s": segment.duration_s,
            "speed_mps": segment.speed_mps,
            "flight_path_angle_rad": segment.flight_path_angle_rad,
            "roll_rad": segment.roll_rad,
            "label": segment.label,
        }
    if isinstance(segment, ConstantRateTurnSegmentSpec):
        return {
            "type": "constant_rate_turn",
            "duration_s": segment.duration_s,
            "speed_mps": segment.speed_mps,
            "heading_rate_radps": segment.heading_rate_radps,
            "flight_path_angle_rad": segment.flight_path_angle_rad,
            "roll_rad": segment.roll_rad,
            "label": segment.label,
        }
    if isinstance(segment, CoordinatedTurnSegmentSpec):
        return {
            "type": "coordinated_turn",
            "duration_s": segment.duration_s,
            "speed_mps": segment.speed_mps,
            "bank_angle_rad": segment.bank_angle_rad,
            "flight_path_angle_rad": segment.flight_path_angle_rad,
            "roll_in_duration_s": segment.roll_in_duration_s,
            "roll_out_duration_s": segment.roll_out_duration_s,
            "smooth": segment.smooth,
            "gravity_mps2": segment.gravity_mps2,
            "label": segment.label,
        }
    raise TypeError(f"Unsupported segment type: {type(segment)!r}")


# -----------------------------------------------------------------------------
# Scenario -> profile / trajectory builders
# -----------------------------------------------------------------------------


def build_profile_for_segment(
    segment: SegmentSpec,
    *,
    dt_s: float,
    initial_heading_rad: float,
    gravity_mps2: float,
) -> VehicleKinematicProfile:
    """
    Build a `VehicleKinematicProfile` for one segment.

    Parameters
    ----------
    segment : SegmentSpec
        Segment specification.
    dt_s : float
        Sample interval [s].
    initial_heading_rad : float
        Heading at the first sample of the segment [rad].
    gravity_mps2 : float
        Reference gravity [m/s^2] used by coordinated-turn segments unless they
        provide their own `gravity_mps2`.

    Returns
    -------
    VehicleKinematicProfile
        Motion profile for the segment.

    Notes
    -----
    This function is intentionally the only place in `scenarios.py` that knows
    how segment-spec types map onto the lower-level vehicle-model builders.
    """
    dt = _require_positive("dt_s", dt_s)
    heading0 = float(_wrap_angle_pi(initial_heading_rad))
    g_ref = _require_positive("gravity_mps2", gravity_mps2)

    if isinstance(segment, StraightSegmentSpec):
        return make_straight_profile(
            duration_s=segment.duration_s,
            dt_s=dt,
            speed_mps=segment.speed_mps,
            heading_rad=heading0,
            flight_path_angle_rad=segment.flight_path_angle_rad,
            roll_rad=segment.roll_rad,
        )

    if isinstance(segment, ConstantRateTurnSegmentSpec):
        return make_constant_rate_turn_profile(
            duration_s=segment.duration_s,
            dt_s=dt,
            speed_mps=segment.speed_mps,
            initial_heading_rad=heading0,
            heading_rate_radps=segment.heading_rate_radps,
            flight_path_angle_rad=segment.flight_path_angle_rad,
            roll_rad=segment.roll_rad,
            gravity_mps2=g_ref,
        )

    if isinstance(segment, CoordinatedTurnSegmentSpec):
        seg_g = g_ref if segment.gravity_mps2 is None else float(segment.gravity_mps2)
        if segment.smooth:
            return make_smooth_coordinated_turn_profile(
                duration_s=segment.duration_s,
                dt_s=dt,
                speed_mps=segment.speed_mps,
                initial_heading_rad=heading0,
                target_bank_angle_rad=segment.bank_angle_rad,
                flight_path_angle_rad=segment.flight_path_angle_rad,
                gravity_mps2=seg_g,
                roll_in_duration_s=segment.roll_in_duration_s,
                roll_out_duration_s=segment.roll_out_duration_s,
            )
        return make_coordinated_turn_profile(
            duration_s=segment.duration_s,
            dt_s=dt,
            speed_mps=segment.speed_mps,
            initial_heading_rad=heading0,
            bank_angle_rad=segment.bank_angle_rad,
            flight_path_angle_rad=segment.flight_path_angle_rad,
            gravity_mps2=seg_g,
        )

    raise TypeError(f"Unsupported segment type: {type(segment)!r}")


def build_profile_from_scenario(
    scenario: ScenarioSpec,
    *,
    dt_s: float | None = None,
    gravity_mps2: float | None = None,
) -> VehicleKinematicProfile:
    """
    Build a full vehicle-kinematic profile from a `ScenarioSpec`.

    Parameters
    ----------
    scenario : ScenarioSpec
        Scenario specification.
    dt_s : float, optional
        Sample interval [s]. If omitted, `scenario.default_dt_s` is used.
    gravity_mps2 : float, optional
        Reference gravity for coordinated-turn bank/rate conversion. If omitted,
        WGS 84 normal gravity at the scenario initial point is used.

    Returns
    -------
    VehicleKinematicProfile
        Concatenated profile across all scenario segments.
    """
    dt = scenario.default_dt_s if dt_s is None else _require_positive("dt_s", dt_s)
    g_ref = scenario.initial_gravity_mps2 if gravity_mps2 is None else _require_positive("gravity_mps2", gravity_mps2)

    profiles: list[VehicleKinematicProfile] = []
    heading = float(scenario.initial_heading_rad)

    for segment in scenario.segments:
        prof = build_profile_for_segment(
            segment,
            dt_s=dt,
            initial_heading_rad=heading,
            gravity_mps2=g_ref,
        )
        profiles.append(prof)
        heading = float(prof.heading_rad[-1])

    return concatenate_profiles(profiles)


def build_truth_trajectory_from_scenario(
    scenario: ScenarioSpec,
    *,
    dt_s: float | None = None,
    gravity_mps2: float | None = None,
) -> TruthTrajectory:
    """
    Build a `TruthTrajectory` from a `ScenarioSpec`.

    Parameters
    ----------
    scenario : ScenarioSpec
        Scenario specification.
    dt_s : float, optional
        Sample interval [s]. If omitted, `scenario.default_dt_s` is used.
    gravity_mps2 : float, optional
        Optional reference gravity for coordinated-turn conversion.

    Returns
    -------
    TruthTrajectory
        Full truth trajectory for the scenario.
    """
    profile = build_profile_from_scenario(
        scenario,
        dt_s=dt_s,
        gravity_mps2=gravity_mps2,
    )
    return build_truth_trajectory_from_profile(
        profile,
        lat0_rad=scenario.initial_lat_rad,
        lon0_rad=scenario.initial_lon_rad,
        height0_m=scenario.initial_height_m,
    )


# -----------------------------------------------------------------------------
# Named default scenarios
# -----------------------------------------------------------------------------


def make_maritime_baseline_scenario() -> ScenarioSpec:
    """
    Return a default maritime survey-style baseline scenario.

    Rationale
    ---------
    NOAA describes hydrographic and mapping operations as commonly following
    systematic lawnmower / back-and-forth coverage patterns rather than generic
    transit motion. This preset captures that idea with:
    - long straight survey legs
    - shallow, smooth 180-degree turns
    - nearly level motion at constant modest speed

    Intended use
    ------------
    - baseline moving-base gravimetry tests
    - early INS + gravimeter debugging
    - map-matching sanity checks in gentle dynamics
    """
    return ScenarioSpec(
        name="maritime_baseline",
        description=(
            "Baseline surface-vessel survey pattern: repeated straight mapping legs "
            "with shallow smooth turns."
        ),
        initial_lat_rad=np.deg2rad(18.250),
        initial_lon_rad=np.deg2rad(72.750),
        initial_height_m=0.0,
        initial_heading_rad=np.deg2rad(90.0),
        default_dt_s=1.0,
        metadata={
            "platform_class": "surface_vessel",
            "mission_style": "coverage_survey",
        },
        segments=(
            StraightSegmentSpec(
                duration_s=600.0,
                speed_mps=4.0,
                label="survey_leg_1",
            ),
            CoordinatedTurnSegmentSpec(
                duration_s=120.0,
                speed_mps=4.0,
                bank_angle_rad=np.deg2rad(8.0),
                roll_in_duration_s=20.0,
                roll_out_duration_s=20.0,
                smooth=True,
                label="turn_1",
            ),
            StraightSegmentSpec(
                duration_s=600.0,
                speed_mps=4.0,
                label="survey_leg_2",
            ),
            CoordinatedTurnSegmentSpec(
                duration_s=120.0,
                speed_mps=4.0,
                bank_angle_rad=np.deg2rad(8.0),
                roll_in_duration_s=20.0,
                roll_out_duration_s=20.0,
                smooth=True,
                label="turn_2",
            ),
            StraightSegmentSpec(
                duration_s=600.0,
                speed_mps=4.0,
                label="survey_leg_3",
            ),
        ),
    )


def make_uuv_long_endurance_scenario() -> ScenarioSpec:
    """
    Return a long-endurance underwater survey scenario.

    Rationale
    ---------
    AUV/UUV mapping and coverage operations are commonly executed in repeated
    lawnmower-style passes. This preset keeps the motion relatively gentle and
    low-speed, which is useful for:
    - long-duration drift studies
    - gravity-map observability checks
    - filter tuning under modest maneuvering stress

    Modeling choice
    ---------------
    The initial ellipsoidal height is set negative to represent operation below
    mean sea level in a simple ellipsoidal-height sense. The rest of the stack
    treats height geometrically; it does not impose any hydrodynamic model here.
    """
    return ScenarioSpec(
        name="uuv_long_endurance",
        description=(
            "Gentle underwater long-endurance coverage survey with repeated "
            "straight legs and smooth shallow turns."
        ),
        initial_lat_rad=np.deg2rad(14.500),
        initial_lon_rad=np.deg2rad(74.000),
        initial_height_m=-50.0,
        initial_heading_rad=np.deg2rad(0.0),
        default_dt_s=1.0,
        metadata={
            "platform_class": "uuv",
            "mission_style": "coverage_survey",
        },
        segments=(
            StraightSegmentSpec(
                duration_s=1800.0,
                speed_mps=2.5,
                label="survey_leg_1",
            ),
            CoordinatedTurnSegmentSpec(
                duration_s=200.0,
                speed_mps=2.5,
                bank_angle_rad=np.deg2rad(6.0),
                roll_in_duration_s=30.0,
                roll_out_duration_s=30.0,
                smooth=True,
                label="turn_1",
            ),
            StraightSegmentSpec(
                duration_s=1800.0,
                speed_mps=2.5,
                label="survey_leg_2",
            ),
            CoordinatedTurnSegmentSpec(
                duration_s=200.0,
                speed_mps=2.5,
                bank_angle_rad=np.deg2rad(6.0),
                roll_in_duration_s=30.0,
                roll_out_duration_s=30.0,
                smooth=True,
                label="turn_2",
            ),
            StraightSegmentSpec(
                duration_s=1800.0,
                speed_mps=2.5,
                label="survey_leg_3",
            ),
        ),
    )


def make_uav_stress_case_scenario() -> ScenarioSpec:
    """
    Return a UAV dynamics-stress scenario.

    Rationale
    ---------
    This preset is intentionally more aggressive than the maritime/UUV cases.
    It mixes:
    - faster airspeed
    - steeper coordinated turns
    - climb and descent straight segments
    - repeated direction reversals

    Intended use
    ------------
    - IMU / gravimeter motion-coupling stress tests
    - truth-generation regression tests
    - validating that estimators do not silently assume "gentle ship motion"
    """
    return ScenarioSpec(
        name="uav_stress_case",
        description=(
            "High-dynamic UAV scenario with steeper turns and alternating climb/"
            "descent legs to stress inertial and gravimetry processing."
        ),
        initial_lat_rad=np.deg2rad(28.500),
        initial_lon_rad=np.deg2rad(77.000),
        initial_height_m=1200.0,
        initial_heading_rad=np.deg2rad(45.0),
        default_dt_s=0.2,
        metadata={
            "platform_class": "uav",
            "mission_style": "dynamic_stress_test",
        },
        segments=(
            StraightSegmentSpec(
                duration_s=60.0,
                speed_mps=35.0,
                flight_path_angle_rad=np.deg2rad(4.0),
                label="climb_leg_1",
            ),
            CoordinatedTurnSegmentSpec(
                duration_s=20.0,
                speed_mps=35.0,
                bank_angle_rad=np.deg2rad(25.0),
                roll_in_duration_s=3.0,
                roll_out_duration_s=3.0,
                smooth=True,
                label="right_turn_1",
            ),
            StraightSegmentSpec(
                duration_s=50.0,
                speed_mps=35.0,
                flight_path_angle_rad=np.deg2rad(-5.0),
                label="descent_leg_1",
            ),
            CoordinatedTurnSegmentSpec(
                duration_s=18.0,
                speed_mps=35.0,
                bank_angle_rad=np.deg2rad(-30.0),
                roll_in_duration_s=2.0,
                roll_out_duration_s=2.0,
                smooth=True,
                label="left_turn_1",
            ),
            StraightSegmentSpec(
                duration_s=40.0,
                speed_mps=32.0,
                flight_path_angle_rad=np.deg2rad(0.0),
                label="level_leg",
            ),
            CoordinatedTurnSegmentSpec(
                duration_s=22.0,
                speed_mps=32.0,
                bank_angle_rad=np.deg2rad(28.0),
                roll_in_duration_s=2.5,
                roll_out_duration_s=2.5,
                smooth=True,
                label="right_turn_2",
            ),
            StraightSegmentSpec(
                duration_s=55.0,
                speed_mps=38.0,
                flight_path_angle_rad=np.deg2rad(3.0),
                label="climb_leg_2",
            ),
        ),
    )


NAMED_SCENARIO_BUILDERS: dict[str, Callable[[], ScenarioSpec]] = {
    "maritime_baseline": make_maritime_baseline_scenario,
    "uuv_long_endurance": make_uuv_long_endurance_scenario,
    "uav_stress_case": make_uav_stress_case_scenario,
}


def available_scenario_names() -> tuple[str, ...]:
    """
    Return the names of built-in scenarios.
    """
    return tuple(NAMED_SCENARIO_BUILDERS.keys())


def get_named_scenario(name: str) -> ScenarioSpec:
    """
    Return one built-in scenario by name.

    Parameters
    ----------
    name : str
        Scenario name.

    Returns
    -------
    ScenarioSpec
        Requested built-in scenario.

    Raises
    ------
    KeyError
        If the scenario name is unknown.
    """
    key = str(name)
    try:
        return NAMED_SCENARIO_BUILDERS[key]()
    except KeyError as exc:
        raise KeyError(
            f"Unknown scenario {key!r}. Available scenarios: {available_scenario_names()}."
        ) from exc


__all__ = [
    "FloatArray",
    "ConstantRateTurnSegmentSpec",
    "CoordinatedTurnSegmentSpec",
    "NAMED_SCENARIO_BUILDERS",
    "ScenarioSpec",
    "SegmentSpec",
    "StraightSegmentSpec",
    "available_scenario_names",
    "build_profile_for_segment",
    "build_profile_from_scenario",
    "build_truth_trajectory_from_scenario",
    "get_named_scenario",
    "make_maritime_baseline_scenario",
    "make_uav_stress_case_scenario",
    "make_uuv_long_endurance_scenario",
]
