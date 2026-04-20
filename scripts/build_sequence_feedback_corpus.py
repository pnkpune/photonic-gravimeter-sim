#!/usr/bin/env python3
"""
Build a hybrid sequence-feedback trust corpus from real multimodal demo packs.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, TypeVar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from gravnav.datasets import resolve_regional_demo_pack
from gravnav.estimators.error_state_ins import (
    ErrorStateINS,
    ErrorStateINSNominalState,
    ErrorStateINSProcessNoise,
    ErrorStateINSState,
)
from gravnav.estimators.feedback_policy import SequenceFeedbackController, SequenceFeedbackSpec
from gravnav.estimators.gravity_sequence_match import (
    SequenceAmbiguityDiagnostics,
    SequenceAnchorEstimate,
    SequenceMatchEstimate,
    SequenceMatchUpdateResult,
)
from gravnav.estimators.integrity import integrity_snapshot_from_ins
from gravnav.estimators.map_match_pf import geodetic_offsets_to_local_ned
from gravnav.ml.feedback_trust import (
    SEQUENCE_FEEDBACK_FEATURE_NAMES,
    SequenceFeedbackEventCorpus,
    extract_sequence_feedback_features,
)
from gravnav.physics.tides import TideCorrectionSpec
from gravnav.sensors.bathymetry import BathymetrySensorSpec
from gravnav.sensors.current_profile import CurrentProfileSensorSpec
from gravnav.sensors.depth import DepthMeasurement, DepthSensorSpec
from gravnav.sensors.gravimeter import GravimeterSpec
from gravnav.sensors.gravity_gradiometer import GravityGradiometerSpec
from gravnav.sensors.imu import IMUMeasurement, IMUSpec
from gravnav.sensors.magnetometer import MagnetometerSensorSpec
from gravnav.sensors.velocity_aid import VelocityAidMeasurement, VelocityAidSpec
from gravnav.simulation.runner import (
    GravitySequenceMatcherSpec,
    ScenarioSimulationRunner,
    SimulationRunnerConfig,
)
from gravnav.truth.scenarios import ScenarioSpec
from gravnav.truth.trajectory import TruthTrajectory
from gravnav.utils.config import load_config_mapping

SpecT = TypeVar("SpecT")

DEFAULT_IMU_CONFIG = PROJECT_ROOT / "configs/sensors/imu_nav_grade.json"
DEFAULT_GRAVIMETER_CONFIG = PROJECT_ROOT / "configs/sensors/gravimeter_proto.json"
DEFAULT_DEPTH_CONFIG = PROJECT_ROOT / "configs/sensors/depth_sensor.json"
DEFAULT_VELOCITY_CONFIG = PROJECT_ROOT / "configs/sensors/velocity_aid.json"
DEFAULT_GRADIOMETER_CONFIG = PROJECT_ROOT / "configs/sensors/gravity_gradiometer_proto.json"
DEFAULT_BATHYMETRY_CONFIG = PROJECT_ROOT / "configs/sensors/bathymetry_sensor.json"
DEFAULT_MAGNETOMETER_CONFIG = PROJECT_ROOT / "configs/sensors/magnetometer_scalar.json"
DEFAULT_CURRENT_PROFILE_CONFIG = PROJECT_ROOT / "configs/sensors/current_profile_sensor.json"
DEFAULT_GAIN_ALPHA_CANDIDATES = (0.1, 0.25, 0.4, 0.5, 0.65, 0.8, 1.0)


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _resolve_path(path_like: str | Path) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower())
    return slug.strip("_") or "cache"


def _array_dict(
    prefix: str,
    value: Any,
) -> dict[str, np.ndarray]:
    return {prefix: np.asarray(value, dtype=np.float64)}


def _serialize_truth_arrays(truth: TruthTrajectory) -> dict[str, np.ndarray]:
    return {
        "truth_time_s": np.asarray(truth.time_s, dtype=np.float64),
        "truth_lat_rad": np.asarray(truth.lat_rad, dtype=np.float64),
        "truth_lon_rad": np.asarray(truth.lon_rad, dtype=np.float64),
        "truth_height_m": np.asarray(truth.height_m, dtype=np.float64),
        "truth_v_ned_mps": np.asarray(truth.v_ned_mps, dtype=np.float64),
        "truth_v_dot_ned_mps2": np.asarray(truth.v_dot_ned_mps2, dtype=np.float64),
        "truth_C_n_b": np.asarray(truth.C_n_b, dtype=np.float64),
        "truth_omega_nb_b_radps": np.asarray(
            truth.omega_nb_b_radps,
            dtype=np.float64,
        ),
    }


def _deserialize_truth_arrays(
    data: Mapping[str, Any],
) -> TruthTrajectory:
    return TruthTrajectory(
        time_s=np.asarray(data["truth_time_s"], dtype=np.float64),
        lat_rad=np.asarray(data["truth_lat_rad"], dtype=np.float64),
        lon_rad=np.asarray(data["truth_lon_rad"], dtype=np.float64),
        height_m=np.asarray(data["truth_height_m"], dtype=np.float64),
        v_ned_mps=np.asarray(data["truth_v_ned_mps"], dtype=np.float64),
        v_dot_ned_mps2=np.asarray(data["truth_v_dot_ned_mps2"], dtype=np.float64),
        C_n_b=np.asarray(data["truth_C_n_b"], dtype=np.float64),
        omega_nb_b_radps=np.asarray(
            data["truth_omega_nb_b_radps"],
            dtype=np.float64,
        ),
    )


def _serialize_ins_state_arrays(
    states: list[ErrorStateINSState],
) -> dict[str, np.ndarray]:
    if not states:
        raise ValueError("Expected at least one INS state to serialize.")
    process_noise = states[0].process_noise
    time_s = np.full(len(states), np.nan, dtype=np.float64)
    for idx, state in enumerate(states):
        if state.nominal.time_s is not None:
            time_s[idx] = float(state.nominal.time_s)
    return {
        "ins_nominal_time_s": time_s,
        "ins_nominal_lat_rad": np.asarray(
            [state.nominal.lat_rad for state in states],
            dtype=np.float64,
        ),
        "ins_nominal_lon_rad": np.asarray(
            [state.nominal.lon_rad for state in states],
            dtype=np.float64,
        ),
        "ins_nominal_height_m": np.asarray(
            [state.nominal.height_m for state in states],
            dtype=np.float64,
        ),
        "ins_nominal_v_ned_mps": np.asarray(
            [state.nominal.v_ned_mps for state in states],
            dtype=np.float64,
        ),
        "ins_nominal_C_n_b": np.asarray(
            [state.nominal.C_n_b for state in states],
            dtype=np.float64,
        ),
        "ins_nominal_gyro_bias_radps": np.asarray(
            [state.nominal.gyro_bias_radps for state in states],
            dtype=np.float64,
        ),
        "ins_nominal_accel_bias_mps2": np.asarray(
            [state.nominal.accel_bias_mps2 for state in states],
            dtype=np.float64,
        ),
        "ins_P": np.asarray([state.P for state in states], dtype=np.float64),
        "ins_process_noise_gyro_white_noise_radps_per_sqrt_hz": np.asarray(
            process_noise.gyro_white_noise_radps_per_sqrt_hz,
            dtype=np.float64,
        ),
        "ins_process_noise_accel_white_noise_mps2_per_sqrt_hz": np.asarray(
            process_noise.accel_white_noise_mps2_per_sqrt_hz,
            dtype=np.float64,
        ),
        "ins_process_noise_gyro_bias_random_walk_radps_per_sqrt_s": np.asarray(
            process_noise.gyro_bias_random_walk_radps_per_sqrt_s,
            dtype=np.float64,
        ),
        "ins_process_noise_accel_bias_random_walk_mps2_per_sqrt_s": np.asarray(
            process_noise.accel_bias_random_walk_mps2_per_sqrt_s,
            dtype=np.float64,
        ),
        "ins_process_noise_name": np.asarray(process_noise.name),
    }


def _deserialize_ins_state_arrays(
    data: Mapping[str, Any],
) -> list[ErrorStateINSState]:
    process_noise = ErrorStateINSProcessNoise(
        gyro_white_noise_radps_per_sqrt_hz=np.asarray(
            data["ins_process_noise_gyro_white_noise_radps_per_sqrt_hz"],
            dtype=np.float64,
        ),
        accel_white_noise_mps2_per_sqrt_hz=np.asarray(
            data["ins_process_noise_accel_white_noise_mps2_per_sqrt_hz"],
            dtype=np.float64,
        ),
        gyro_bias_random_walk_radps_per_sqrt_s=np.asarray(
            data["ins_process_noise_gyro_bias_random_walk_radps_per_sqrt_s"],
            dtype=np.float64,
        ),
        accel_bias_random_walk_mps2_per_sqrt_s=np.asarray(
            data["ins_process_noise_accel_bias_random_walk_mps2_per_sqrt_s"],
            dtype=np.float64,
        ),
        name=str(np.asarray(data["ins_process_noise_name"]).item()),
    )
    states: list[ErrorStateINSState] = []
    time_s = np.asarray(data["ins_nominal_time_s"], dtype=np.float64)
    lat_rad = np.asarray(data["ins_nominal_lat_rad"], dtype=np.float64)
    lon_rad = np.asarray(data["ins_nominal_lon_rad"], dtype=np.float64)
    height_m = np.asarray(data["ins_nominal_height_m"], dtype=np.float64)
    v_ned_mps = np.asarray(data["ins_nominal_v_ned_mps"], dtype=np.float64)
    C_n_b = np.asarray(data["ins_nominal_C_n_b"], dtype=np.float64)
    gyro_bias = np.asarray(data["ins_nominal_gyro_bias_radps"], dtype=np.float64)
    accel_bias = np.asarray(data["ins_nominal_accel_bias_mps2"], dtype=np.float64)
    P = np.asarray(data["ins_P"], dtype=np.float64)
    for idx in range(lat_rad.shape[0]):
        nominal_time_s = None if not np.isfinite(time_s[idx]) else float(time_s[idx])
        states.append(
            ErrorStateINSState(
                nominal=ErrorStateINSNominalState(
                    time_s=nominal_time_s,
                    lat_rad=float(lat_rad[idx]),
                    lon_rad=float(lon_rad[idx]),
                    height_m=float(height_m[idx]),
                    v_ned_mps=v_ned_mps[idx],
                    C_n_b=C_n_b[idx],
                    gyro_bias_radps=gyro_bias[idx],
                    accel_bias_mps2=accel_bias[idx],
                ),
                P=P[idx],
                process_noise=process_noise,
            )
        )
    return states


def _serialize_imu_arrays(
    samples: list[IMUMeasurement],
) -> dict[str, np.ndarray]:
    time_s = np.full(len(samples), np.nan, dtype=np.float64)
    for idx, sample in enumerate(samples):
        if sample.time_s is not None:
            time_s[idx] = float(sample.time_s)
    return {
        "imu_time_s": time_s,
        "imu_omega_ib_b_radps": np.asarray(
            [sample.omega_ib_b_radps for sample in samples],
            dtype=np.float64,
        ),
        "imu_f_ib_b_mps2": np.asarray(
            [sample.f_ib_b_mps2 for sample in samples],
            dtype=np.float64,
        ),
        "imu_ideal_omega_ib_b_radps": np.asarray(
            [sample.ideal_omega_ib_b_radps for sample in samples],
            dtype=np.float64,
        ),
        "imu_ideal_f_ib_b_mps2": np.asarray(
            [sample.ideal_f_ib_b_mps2 for sample in samples],
            dtype=np.float64,
        ),
        "imu_gyro_bias_used_radps": np.asarray(
            [sample.gyro_bias_used_radps for sample in samples],
            dtype=np.float64,
        ),
        "imu_accel_bias_used_mps2": np.asarray(
            [sample.accel_bias_used_mps2 for sample in samples],
            dtype=np.float64,
        ),
        "imu_gyro_white_noise_radps": np.asarray(
            [sample.gyro_white_noise_radps for sample in samples],
            dtype=np.float64,
        ),
        "imu_accel_white_noise_mps2": np.asarray(
            [sample.accel_white_noise_mps2 for sample in samples],
            dtype=np.float64,
        ),
        "imu_gyro_saturated": np.asarray(
            [sample.gyro_saturated for sample in samples],
            dtype=bool,
        ),
        "imu_accel_saturated": np.asarray(
            [sample.accel_saturated for sample in samples],
            dtype=bool,
        ),
    }


def _deserialize_imu_arrays(
    data: Mapping[str, Any],
) -> list[IMUMeasurement]:
    time_s = np.asarray(data["imu_time_s"], dtype=np.float64)
    omega = np.asarray(data["imu_omega_ib_b_radps"], dtype=np.float64)
    specific_force = np.asarray(data["imu_f_ib_b_mps2"], dtype=np.float64)
    ideal_omega = np.asarray(data["imu_ideal_omega_ib_b_radps"], dtype=np.float64)
    ideal_force = np.asarray(data["imu_ideal_f_ib_b_mps2"], dtype=np.float64)
    gyro_bias = np.asarray(data["imu_gyro_bias_used_radps"], dtype=np.float64)
    accel_bias = np.asarray(data["imu_accel_bias_used_mps2"], dtype=np.float64)
    gyro_noise = np.asarray(data["imu_gyro_white_noise_radps"], dtype=np.float64)
    accel_noise = np.asarray(data["imu_accel_white_noise_mps2"], dtype=np.float64)
    gyro_saturated = np.asarray(data["imu_gyro_saturated"], dtype=bool)
    accel_saturated = np.asarray(data["imu_accel_saturated"], dtype=bool)
    out: list[IMUMeasurement] = []
    for idx in range(omega.shape[0]):
        out.append(
            IMUMeasurement(
                time_s=None if not np.isfinite(time_s[idx]) else float(time_s[idx]),
                omega_ib_b_radps=omega[idx],
                f_ib_b_mps2=specific_force[idx],
                ideal_omega_ib_b_radps=ideal_omega[idx],
                ideal_f_ib_b_mps2=ideal_force[idx],
                gyro_bias_used_radps=gyro_bias[idx],
                accel_bias_used_mps2=accel_bias[idx],
                gyro_white_noise_radps=gyro_noise[idx],
                accel_white_noise_mps2=accel_noise[idx],
                gyro_saturated=bool(gyro_saturated[idx]),
                accel_saturated=bool(accel_saturated[idx]),
            )
        )
    return out


def _serialize_depth_arrays(
    measurements: list[DepthMeasurement | None],
) -> dict[str, np.ndarray]:
    count = len(measurements)
    valid = np.asarray([meas is not None for meas in measurements], dtype=bool)
    time_s = np.full(count, np.nan, dtype=np.float64)
    value_m = np.zeros(count, dtype=np.float64)
    ideal_depth_m = np.zeros(count, dtype=np.float64)
    filtered_depth_m = np.zeros(count, dtype=np.float64)
    bias_used_m = np.zeros(count, dtype=np.float64)
    white_noise_m = np.zeros(count, dtype=np.float64)
    reference_surface_height_m = np.zeros(count, dtype=np.float64)
    saturated = np.zeros(count, dtype=bool)
    for idx, meas in enumerate(measurements):
        if meas is None:
            continue
        if meas.time_s is not None:
            time_s[idx] = float(meas.time_s)
        value_m[idx] = float(meas.value_m)
        ideal_depth_m[idx] = float(meas.ideal_depth_m)
        filtered_depth_m[idx] = float(meas.filtered_depth_m)
        bias_used_m[idx] = float(meas.bias_used_m)
        white_noise_m[idx] = float(meas.white_noise_m)
        saturated[idx] = bool(meas.saturated)
        reference_surface_height_m[idx] = float(meas.reference_surface_height_m)
    return {
        "depth_valid": valid,
        "depth_time_s": time_s,
        "depth_value_m": value_m,
        "depth_ideal_depth_m": ideal_depth_m,
        "depth_filtered_depth_m": filtered_depth_m,
        "depth_bias_used_m": bias_used_m,
        "depth_white_noise_m": white_noise_m,
        "depth_saturated": saturated,
        "depth_reference_surface_height_m": reference_surface_height_m,
    }


def _deserialize_depth_arrays(
    data: Mapping[str, Any],
) -> list[DepthMeasurement | None]:
    valid = np.asarray(data["depth_valid"], dtype=bool)
    time_s = np.asarray(data["depth_time_s"], dtype=np.float64)
    value_m = np.asarray(data["depth_value_m"], dtype=np.float64)
    ideal_depth_m = np.asarray(data["depth_ideal_depth_m"], dtype=np.float64)
    filtered_depth_m = np.asarray(data["depth_filtered_depth_m"], dtype=np.float64)
    bias_used_m = np.asarray(data["depth_bias_used_m"], dtype=np.float64)
    white_noise_m = np.asarray(data["depth_white_noise_m"], dtype=np.float64)
    saturated = np.asarray(data["depth_saturated"], dtype=bool)
    reference_surface_height_m = np.asarray(
        data["depth_reference_surface_height_m"],
        dtype=np.float64,
    )
    out: list[DepthMeasurement | None] = []
    for idx, is_valid in enumerate(valid.tolist()):
        if not is_valid:
            out.append(None)
            continue
        out.append(
            DepthMeasurement(
                time_s=None if not np.isfinite(time_s[idx]) else float(time_s[idx]),
                value_m=float(value_m[idx]),
                ideal_depth_m=float(ideal_depth_m[idx]),
                filtered_depth_m=float(filtered_depth_m[idx]),
                bias_used_m=float(bias_used_m[idx]),
                white_noise_m=float(white_noise_m[idx]),
                saturated=bool(saturated[idx]),
                reference_surface_height_m=float(reference_surface_height_m[idx]),
            )
        )
    return out


def _serialize_velocity_arrays(
    measurements: list[VelocityAidMeasurement | None],
) -> dict[str, np.ndarray]:
    count = len(measurements)
    valid = np.asarray([meas is not None for meas in measurements], dtype=bool)
    time_s = np.full(count, np.nan, dtype=np.float64)
    value_mps = np.zeros((count, 3), dtype=np.float64)
    ideal_value_mps = np.zeros((count, 3), dtype=np.float64)
    filtered_input_mps = np.zeros((count, 3), dtype=np.float64)
    bias_used_mps = np.zeros((count, 3), dtype=np.float64)
    white_noise_mps = np.zeros((count, 3), dtype=np.float64)
    saturated = np.zeros(count, dtype=bool)
    kind = np.full(count, "", dtype=np.str_)
    frame = np.full(count, "", dtype=np.str_)
    for idx, meas in enumerate(measurements):
        if meas is None:
            continue
        if meas.time_s is not None:
            time_s[idx] = float(meas.time_s)
        value_mps[idx] = np.asarray(meas.value_mps, dtype=np.float64)
        ideal_value_mps[idx] = np.asarray(meas.ideal_value_mps, dtype=np.float64)
        filtered_input_mps[idx] = np.asarray(meas.filtered_input_mps, dtype=np.float64)
        bias_used_mps[idx] = np.asarray(meas.bias_used_mps, dtype=np.float64)
        white_noise_mps[idx] = np.asarray(meas.white_noise_mps, dtype=np.float64)
        saturated[idx] = bool(meas.saturated)
        kind[idx] = str(meas.kind)
        frame[idx] = str(meas.frame)
    return {
        "velocity_valid": valid,
        "velocity_time_s": time_s,
        "velocity_value_mps": value_mps,
        "velocity_ideal_value_mps": ideal_value_mps,
        "velocity_filtered_input_mps": filtered_input_mps,
        "velocity_bias_used_mps": bias_used_mps,
        "velocity_white_noise_mps": white_noise_mps,
        "velocity_saturated": saturated,
        "velocity_kind": kind,
        "velocity_frame": frame,
    }


def _deserialize_velocity_arrays(
    data: Mapping[str, Any],
) -> list[VelocityAidMeasurement | None]:
    valid = np.asarray(data["velocity_valid"], dtype=bool)
    time_s = np.asarray(data["velocity_time_s"], dtype=np.float64)
    value_mps = np.asarray(data["velocity_value_mps"], dtype=np.float64)
    ideal_value_mps = np.asarray(data["velocity_ideal_value_mps"], dtype=np.float64)
    filtered_input_mps = np.asarray(
        data["velocity_filtered_input_mps"],
        dtype=np.float64,
    )
    bias_used_mps = np.asarray(data["velocity_bias_used_mps"], dtype=np.float64)
    white_noise_mps = np.asarray(data["velocity_white_noise_mps"], dtype=np.float64)
    saturated = np.asarray(data["velocity_saturated"], dtype=bool)
    kind = np.asarray(data["velocity_kind"], dtype=np.str_)
    frame = np.asarray(data["velocity_frame"], dtype=np.str_)
    out: list[VelocityAidMeasurement | None] = []
    for idx, is_valid in enumerate(valid.tolist()):
        if not is_valid:
            out.append(None)
            continue
        out.append(
            VelocityAidMeasurement(
                kind=str(kind[idx]),
                frame=str(frame[idx]),
                time_s=None if not np.isfinite(time_s[idx]) else float(time_s[idx]),
                value_mps=value_mps[idx],
                ideal_value_mps=ideal_value_mps[idx],
                filtered_input_mps=filtered_input_mps[idx],
                bias_used_mps=bias_used_mps[idx],
                white_noise_mps=white_noise_mps[idx],
                saturated=bool(saturated[idx]),
            )
        )
    return out


def _serialize_sequence_update(
    update: SequenceMatchUpdateResult,
) -> dict[str, Any]:
    return {
        "estimate": {
            "lat_rad": float(update.estimate.lat_rad),
            "lon_rad": float(update.estimate.lon_rad),
            "height_m": float(update.estimate.height_m),
            "covariance_ned_m2": np.asarray(
                update.estimate.covariance_ned_m2,
                dtype=np.float64,
            ).tolist(),
            "covariance_geodetic": np.asarray(
                update.estimate.covariance_geodetic,
                dtype=np.float64,
            ).tolist(),
            "predicted_disturbance_mps2": update.estimate.predicted_disturbance_mps2,
            "marginal_peak_probability": update.estimate.marginal_peak_probability,
            "predicted_bathymetry_m": update.estimate.predicted_bathymetry_m,
            "predicted_magnetic_total_nt": update.estimate.predicted_magnetic_total_nt,
        },
        "global_index": int(update.global_index),
        "time_s": float(update.time_s),
        "window_size_used": int(update.window_size_used),
        "delayed_by_steps": int(update.delayed_by_steps),
        "num_candidates": int(update.num_candidates),
        "posterior_entropy_nats": float(update.posterior_entropy_nats),
        "marginal_peak_probability": float(update.marginal_peak_probability),
        "predicted_disturbance_mean_mps2": float(update.predicted_disturbance_mean_mps2),
        "predicted_disturbance_std_mps2": float(update.predicted_disturbance_std_mps2),
        "used_gradient": bool(update.used_gradient),
        "used_bathymetry": bool(update.used_bathymetry),
        "used_magnetics": bool(update.used_magnetics),
        "viterbi_log_score": float(update.viterbi_log_score),
        "viterbi_offset_ned_m": np.asarray(
            update.viterbi_offset_ned_m,
            dtype=np.float64,
        ).tolist(),
        "posterior_mean_offset_ned_m": np.asarray(
            update.posterior_mean_offset_ned_m,
            dtype=np.float64,
        ).tolist(),
        "ambiguity_diagnostics": {
            "posterior_candidate_ess": float(
                update.ambiguity_diagnostics.posterior_candidate_ess
            ),
            "posterior_candidate_ess_fraction": float(
                update.ambiguity_diagnostics.posterior_candidate_ess_fraction
            ),
            "edge_mass_fraction": float(update.ambiguity_diagnostics.edge_mass_fraction),
            "support_radius_n_m": float(update.ambiguity_diagnostics.support_radius_n_m),
            "support_radius_e_m": float(update.ambiguity_diagnostics.support_radius_e_m),
            "horizontal_covariance_eigenvalue_ratio": float(
                update.ambiguity_diagnostics.horizontal_covariance_eigenvalue_ratio
            ),
            "grid_saturated_north": bool(
                update.ambiguity_diagnostics.grid_saturated_north
            ),
            "grid_saturated_east": bool(
                update.ambiguity_diagnostics.grid_saturated_east
            ),
            "grid_saturated_any": bool(update.ambiguity_diagnostics.grid_saturated_any),
            "gravity_predicted_spread_mps2": float(
                update.ambiguity_diagnostics.gravity_predicted_spread_mps2
            ),
            "gravity_information_ratio": float(
                update.ambiguity_diagnostics.gravity_information_ratio
            ),
            "bathymetry_predicted_spread_m": update.ambiguity_diagnostics.bathymetry_predicted_spread_m,
            "bathymetry_information_ratio": update.ambiguity_diagnostics.bathymetry_information_ratio,
            "magnetic_predicted_spread_nt": update.ambiguity_diagnostics.magnetic_predicted_spread_nt,
            "magnetic_information_ratio": update.ambiguity_diagnostics.magnetic_information_ratio,
            "dominant_failure_mode": str(update.ambiguity_diagnostics.dominant_failure_mode),
            "grid_mode": str(update.ambiguity_diagnostics.grid_mode),
            "grid_half_span_m": np.asarray(
                update.ambiguity_diagnostics.grid_half_span_m,
                dtype=np.float64,
            ).tolist(),
            "grid_spacing_m": np.asarray(
                update.ambiguity_diagnostics.grid_spacing_m,
                dtype=np.float64,
            ).tolist(),
        },
        "anchor_estimates": [
            {
                "time_s": float(anchor.time_s),
                "lat_rad": float(anchor.lat_rad),
                "lon_rad": float(anchor.lon_rad),
                "height_m": float(anchor.height_m),
                "covariance_ned_m2": np.asarray(
                    anchor.covariance_ned_m2,
                    dtype=np.float64,
                ).tolist(),
                "covariance_geodetic": np.asarray(
                    anchor.covariance_geodetic,
                    dtype=np.float64,
                ).tolist(),
                "posterior_mean_offset_ned_m": np.asarray(
                    anchor.posterior_mean_offset_ned_m,
                    dtype=np.float64,
                ).tolist(),
                "viterbi_offset_ned_m": np.asarray(
                    anchor.viterbi_offset_ned_m,
                    dtype=np.float64,
                ).tolist(),
                "marginal_peak_probability": float(anchor.marginal_peak_probability),
                "posterior_entropy_nats": float(anchor.posterior_entropy_nats),
                "global_index": int(anchor.global_index),
                "delayed_by_steps": int(anchor.delayed_by_steps),
            }
            for anchor in update.anchor_estimates
        ],
        "publishability_probability": update.publishability_probability,
        "support_expansion_probability": update.support_expansion_probability,
        "learned_covariance_scale": update.learned_covariance_scale,
        "localizer_name": str(update.localizer_name),
    }


def _deserialize_sequence_update(
    payload: Mapping[str, Any],
) -> SequenceMatchUpdateResult:
    estimate_payload = payload["estimate"]
    ambiguity_payload = payload["ambiguity_diagnostics"]
    anchor_payloads = payload.get("anchor_estimates", [])
    return SequenceMatchUpdateResult(
        estimate=SequenceMatchEstimate(
            lat_rad=float(estimate_payload["lat_rad"]),
            lon_rad=float(estimate_payload["lon_rad"]),
            height_m=float(estimate_payload["height_m"]),
            covariance_ned_m2=np.asarray(
                estimate_payload["covariance_ned_m2"],
                dtype=np.float64,
            ),
            covariance_geodetic=np.asarray(
                estimate_payload["covariance_geodetic"],
                dtype=np.float64,
            ),
            predicted_disturbance_mps2=estimate_payload.get(
                "predicted_disturbance_mps2"
            ),
            marginal_peak_probability=estimate_payload.get(
                "marginal_peak_probability"
            ),
            predicted_bathymetry_m=estimate_payload.get("predicted_bathymetry_m"),
            predicted_magnetic_total_nt=estimate_payload.get(
                "predicted_magnetic_total_nt"
            ),
        ),
        global_index=int(payload["global_index"]),
        time_s=float(payload["time_s"]),
        window_size_used=int(payload["window_size_used"]),
        delayed_by_steps=int(payload["delayed_by_steps"]),
        num_candidates=int(payload["num_candidates"]),
        posterior_entropy_nats=float(payload["posterior_entropy_nats"]),
        marginal_peak_probability=float(payload["marginal_peak_probability"]),
        predicted_disturbance_mean_mps2=float(
            payload["predicted_disturbance_mean_mps2"]
        ),
        predicted_disturbance_std_mps2=float(
            payload["predicted_disturbance_std_mps2"]
        ),
        used_gradient=bool(payload["used_gradient"]),
        used_bathymetry=bool(payload["used_bathymetry"]),
        used_magnetics=bool(payload["used_magnetics"]),
        viterbi_log_score=float(payload["viterbi_log_score"]),
        viterbi_offset_ned_m=np.asarray(
            payload["viterbi_offset_ned_m"],
            dtype=np.float64,
        ),
        posterior_mean_offset_ned_m=np.asarray(
            payload["posterior_mean_offset_ned_m"],
            dtype=np.float64,
        ),
        ambiguity_diagnostics=SequenceAmbiguityDiagnostics(
            posterior_candidate_ess=float(
                ambiguity_payload["posterior_candidate_ess"]
            ),
            posterior_candidate_ess_fraction=float(
                ambiguity_payload["posterior_candidate_ess_fraction"]
            ),
            edge_mass_fraction=float(ambiguity_payload["edge_mass_fraction"]),
            support_radius_n_m=float(ambiguity_payload["support_radius_n_m"]),
            support_radius_e_m=float(ambiguity_payload["support_radius_e_m"]),
            horizontal_covariance_eigenvalue_ratio=float(
                ambiguity_payload["horizontal_covariance_eigenvalue_ratio"]
            ),
            grid_saturated_north=bool(ambiguity_payload["grid_saturated_north"]),
            grid_saturated_east=bool(ambiguity_payload["grid_saturated_east"]),
            grid_saturated_any=bool(ambiguity_payload["grid_saturated_any"]),
            gravity_predicted_spread_mps2=float(
                ambiguity_payload["gravity_predicted_spread_mps2"]
            ),
            gravity_information_ratio=float(
                ambiguity_payload["gravity_information_ratio"]
            ),
            bathymetry_predicted_spread_m=ambiguity_payload.get(
                "bathymetry_predicted_spread_m"
            ),
            bathymetry_information_ratio=ambiguity_payload.get(
                "bathymetry_information_ratio"
            ),
            magnetic_predicted_spread_nt=ambiguity_payload.get(
                "magnetic_predicted_spread_nt"
            ),
            magnetic_information_ratio=ambiguity_payload.get(
                "magnetic_information_ratio"
            ),
            dominant_failure_mode=str(ambiguity_payload["dominant_failure_mode"]),
            grid_mode=str(ambiguity_payload["grid_mode"]),
            grid_half_span_m=np.asarray(
                ambiguity_payload["grid_half_span_m"],
                dtype=np.float64,
            ),
            grid_spacing_m=np.asarray(
                ambiguity_payload["grid_spacing_m"],
                dtype=np.float64,
            ),
        ),
        anchor_estimates=tuple(
            SequenceAnchorEstimate(
                time_s=float(anchor["time_s"]),
                lat_rad=float(anchor["lat_rad"]),
                lon_rad=float(anchor["lon_rad"]),
                height_m=float(anchor["height_m"]),
                covariance_ned_m2=np.asarray(
                    anchor["covariance_ned_m2"],
                    dtype=np.float64,
                ),
                covariance_geodetic=np.asarray(
                    anchor["covariance_geodetic"],
                    dtype=np.float64,
                ),
                posterior_mean_offset_ned_m=np.asarray(
                    anchor["posterior_mean_offset_ned_m"],
                    dtype=np.float64,
                ),
                viterbi_offset_ned_m=np.asarray(
                    anchor["viterbi_offset_ned_m"],
                    dtype=np.float64,
                ),
                marginal_peak_probability=float(anchor["marginal_peak_probability"]),
                posterior_entropy_nats=float(anchor["posterior_entropy_nats"]),
                global_index=int(anchor["global_index"]),
                delayed_by_steps=int(anchor["delayed_by_steps"]),
            )
            for anchor in anchor_payloads
        ),
        publishability_probability=payload.get("publishability_probability"),
        support_expansion_probability=payload.get("support_expansion_probability"),
        learned_covariance_scale=payload.get("learned_covariance_scale"),
        localizer_name=str(payload.get("localizer_name", "sequence")),
    )


def _event_cache_path(
    event_cache_dir: Path,
    *,
    region_name: str,
    seed: int,
) -> Path:
    return event_cache_dir / f"{_slugify(region_name)}_seed_{int(seed)}.npz"


def save_feedback_event_cache(
    cache_path: Path,
    *,
    truth: TruthTrajectory,
    ins_states: list[ErrorStateINSState],
    imu_samples: list[IMUMeasurement],
    depth_measurements_by_step: list[DepthMeasurement | None],
    velocity_measurements_by_step: list[VelocityAidMeasurement | None],
    depth_variance_m2: float,
    velocity_R: np.ndarray,
    event_rows: list[dict[str, Any]],
    metadata: Mapping[str, Any],
) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **_serialize_truth_arrays(truth),
        **_serialize_ins_state_arrays(ins_states),
        **_serialize_imu_arrays(imu_samples),
        **_serialize_depth_arrays(depth_measurements_by_step),
        **_serialize_velocity_arrays(velocity_measurements_by_step),
        "depth_variance_m2": np.asarray(float(depth_variance_m2), dtype=np.float64),
        "velocity_R": np.asarray(velocity_R, dtype=np.float64),
        "events_json": np.asarray(
            json.dumps(event_rows, separators=(",", ":")),
        ),
        "metadata_json": np.asarray(
            json.dumps(dict(metadata), separators=(",", ":")),
        ),
    }
    np.savez_compressed(cache_path, **payload)
    return cache_path


def load_feedback_event_cache(
    cache_path: Path,
) -> dict[str, Any]:
    with np.load(cache_path, allow_pickle=False) as data:
        mapping: dict[str, Any] = {key: data[key] for key in data.files}
    return {
        "truth": _deserialize_truth_arrays(mapping),
        "ins_states": _deserialize_ins_state_arrays(mapping),
        "imu_samples": _deserialize_imu_arrays(mapping),
        "depth_measurements_by_step": _deserialize_depth_arrays(mapping),
        "velocity_measurements_by_step": _deserialize_velocity_arrays(mapping),
        "depth_variance_m2": float(np.asarray(mapping["depth_variance_m2"]).item()),
        "velocity_R": np.asarray(mapping["velocity_R"], dtype=np.float64),
        "event_rows": json.loads(str(np.asarray(mapping["events_json"]).item())),
        "metadata": json.loads(str(np.asarray(mapping["metadata_json"]).item())),
    }


def _instantiate_dataclass(
    spec_cls: type[SpecT],
    mapping: Mapping[str, Any],
    *,
    ignore_unknown: bool = False,
) -> SpecT:
    if not is_dataclass(spec_cls):
        raise TypeError(f"{spec_cls!r} is not a dataclass type.")
    allowed = {field.name for field in fields(spec_cls)}
    unknown = sorted(set(mapping.keys()) - allowed)
    if unknown and not ignore_unknown:
        raise KeyError(
            f"Unsupported keys for {spec_cls.__name__}: {unknown}. "
            f"Allowed keys: {sorted(allowed)}."
        )
    return spec_cls(**{key: value for key, value in mapping.items() if key in allowed})


def _load_spec(
    spec_cls: type[SpecT],
    path: Path,
    *,
    ignore_unknown: bool = False,
) -> SpecT:
    return _instantiate_dataclass(
        spec_cls,
        load_config_mapping(path),
        ignore_unknown=ignore_unknown,
    )


def _measurements_by_step(
    truth_time_s: np.ndarray,
    measurements: list[Any],
) -> list[Any | None]:
    step_lookup = {
        int(round(float(t) * 1.0e6)): idx for idx, t in enumerate(truth_time_s.tolist())
    }
    out: list[Any | None] = [None] * int(len(truth_time_s))
    for meas in measurements:
        time_s = getattr(meas, "time_s", None)
        if time_s is None:
            continue
        key = int(round(float(time_s) * 1.0e6))
        idx = step_lookup.get(key)
        if idx is not None:
            out[int(idx)] = meas
    return out


def _horizontal_error_m(
    state: ErrorStateINSState,
    *,
    truth_lat_rad: float,
    truth_lon_rad: float,
    truth_height_m: float,
) -> float:
    offset = geodetic_offsets_to_local_ned(
        np.asarray([float(state.nominal.lat_rad)], dtype=np.float64),
        np.asarray([float(state.nominal.lon_rad)], dtype=np.float64),
        np.asarray([float(state.nominal.height_m)], dtype=np.float64),
        lat_ref_rad=float(truth_lat_rad),
        lon_ref_rad=float(truth_lon_rad),
        height_ref_m=float(truth_height_m),
    )[0]
    return float(np.linalg.norm(offset[:2]))


def _horizon_metrics(
    states: list[ErrorStateINSState],
    *,
    truth: Any,
    step_start: int,
    integrity_horizontal_alert_limit_m: float | None,
    integrity_vertical_alert_limit_m: float | None,
) -> tuple[float, bool]:
    errors_sq: list[float] = []
    hmi_safe = True
    for local_idx, state in enumerate(states):
        truth_idx = int(step_start + local_idx)
        err_h = _horizontal_error_m(
            state,
            truth_lat_rad=float(truth.lat_rad[truth_idx]),
            truth_lon_rad=float(truth.lon_rad[truth_idx]),
            truth_height_m=float(truth.height_m[truth_idx]),
        )
        errors_sq.append(err_h**2)
        snap = integrity_snapshot_from_ins(
            state,
            true_lat_rad=float(truth.lat_rad[truth_idx]),
            true_lon_rad=float(truth.lon_rad[truth_idx]),
            true_height_m=float(truth.height_m[truth_idx]),
            horizontal_alert_limit_m=integrity_horizontal_alert_limit_m,
            vertical_alert_limit_m=integrity_vertical_alert_limit_m,
            time_s=float(truth.time_s[truth_idx]),
        )
        if bool(snap.hazardously_misleading_horizontal):
            hmi_safe = False
    return float(np.sqrt(np.mean(np.asarray(errors_sq, dtype=np.float64)))), bool(hmi_safe)


def _evaluate_mode(
    *,
    runner: ScenarioSimulationRunner,
    truth: Any,
    ins_states: list[ErrorStateINSState],
    imu_samples: list[Any],
    update: Any,
    current_step: int,
    horizon_end_step: int,
    depth_measurements_by_step: list[DepthMeasurement | None],
    velocity_measurements_by_step: list[VelocityAidMeasurement | None],
    depth_variance_m2: float,
    velocity_R: np.ndarray,
    mode: str,
    measurement_geometry: str,
    covariance_inflation: float,
    transfer_rw_std_mps: float,
    nis_threshold: float | None,
    minimum_useful_improvement_m: float,
    gain_alpha: float,
) -> dict[str, Any]:
    spec = SequenceFeedbackSpec(
        enabled=True,
        mode=mode,
        measurement_geometry=measurement_geometry,
        min_window_size=1,
        min_peak_probability=0.0,
        min_horizontal_eigenvalue_ratio=1.0,
        max_horizontal_std_m=1.0e6,
        max_correction_norm_m=1.0e6,
        covariance_inflation=covariance_inflation,
        transfer_rw_std_mps=transfer_rw_std_mps,
        reset_matcher_after_apply=False,
        nis_threshold=nis_threshold,
        trust_gate_source="heuristic",
        fixed_gain_alpha_override=float(gain_alpha),
    )
    ctrl = SequenceFeedbackController(spec)
    previous_live = None if current_step <= 0 else ins_states[current_step - 1]

    if mode == "lag_replay":
        target_step = int(update.global_index)
        delayed_ins = ErrorStateINS(ins_states[target_step].copy())
        result = ctrl.evaluate(
            update,
            delayed_ins,
            current_time_s=float(truth.time_s[current_step]),
            live_ins_state=ins_states[current_step],
            previous_live_ins_state=previous_live,
        )
        if not result.applied:
            return {
                "applied": False,
                "improves_error": False,
                "hmi_safe": False,
                "useful_and_safe": False,
                "error_delta_m": 0.0,
            }
        _, replayed_states = runner._replay_ins_segment(
            truth=truth,
            start_index=target_step,
            end_index=horizon_end_step,
            initial_state=delayed_ins.state,
            imu_samples=imu_samples,
            depth_measurements_by_step=depth_measurements_by_step,
            velocity_measurements_by_step=velocity_measurements_by_step,
            depth_variance_m2=depth_variance_m2,
            velocity_R=velocity_R,
        )
        corrected_states = replayed_states[(current_step - target_step):]
    else:
        target_step = current_step
        live_ins = ErrorStateINS(ins_states[current_step].copy())
        result = ctrl.evaluate(
            update,
            live_ins,
            current_time_s=float(truth.time_s[current_step]),
            live_ins_state=ins_states[current_step],
            previous_live_ins_state=previous_live,
        )
        if not result.applied:
            return {
                "applied": False,
                "improves_error": False,
                "hmi_safe": False,
                "useful_and_safe": False,
                "error_delta_m": 0.0,
            }
        _, corrected_states = runner._replay_ins_segment(
            truth=truth,
            start_index=current_step,
            end_index=horizon_end_step,
            initial_state=live_ins.state,
            imu_samples=imu_samples,
            depth_measurements_by_step=depth_measurements_by_step,
            velocity_measurements_by_step=velocity_measurements_by_step,
            depth_variance_m2=depth_variance_m2,
            velocity_R=velocity_R,
        )

    baseline_states = [ins_states[idx].copy() for idx in range(current_step, horizon_end_step + 1)]
    baseline_rmse_m, _ = _horizon_metrics(
        baseline_states,
        truth=truth,
        step_start=current_step,
        integrity_horizontal_alert_limit_m=runner.config.integrity.horizontal_alert_limit_m,
        integrity_vertical_alert_limit_m=runner.config.integrity.vertical_alert_limit_m,
    )
    corrected_rmse_m, hmi_safe = _horizon_metrics(
        corrected_states,
        truth=truth,
        step_start=current_step,
        integrity_horizontal_alert_limit_m=runner.config.integrity.horizontal_alert_limit_m,
        integrity_vertical_alert_limit_m=runner.config.integrity.vertical_alert_limit_m,
    )
    error_delta_m = float(corrected_rmse_m - baseline_rmse_m)
    improves_error = bool(error_delta_m < 0.0)
    materially_improves_error = bool(
        error_delta_m <= -abs(float(minimum_useful_improvement_m))
    )
    return {
        "applied": True,
        "improves_error": improves_error,
        "materially_improves_error": materially_improves_error,
        "hmi_safe": bool(hmi_safe),
        "useful_and_safe": bool(materially_improves_error and hmi_safe),
        "error_delta_m": error_delta_m,
        "best_gain_alpha": float(gain_alpha),
    }


def _select_best_gain_metrics(
    *,
    runner: ScenarioSimulationRunner,
    truth: Any,
    ins_states: list[ErrorStateINSState],
    imu_samples: list[Any],
    update: Any,
    current_step: int,
    horizon_end_step: int,
    depth_measurements_by_step: list[DepthMeasurement | None],
    velocity_measurements_by_step: list[VelocityAidMeasurement | None],
    depth_variance_m2: float,
    velocity_R: np.ndarray,
    mode: str,
    measurement_geometry: str,
    covariance_inflation: float,
    transfer_rw_std_mps: float,
    nis_threshold: float | None,
    minimum_useful_improvement_m: float,
    gain_alpha_candidates: list[float],
) -> dict[str, Any]:
    best_metrics = {
        "applied": False,
        "improves_error": False,
        "materially_improves_error": False,
        "hmi_safe": True,
        "useful_and_safe": False,
        "error_delta_m": 0.0,
        "best_gain_alpha": 0.0,
    }
    best_error_delta_m = 0.0
    for gain_alpha in gain_alpha_candidates:
        alpha = float(gain_alpha)
        if not np.isfinite(alpha) or alpha <= 0.0:
            continue
        metrics = _evaluate_mode(
            runner=runner,
            truth=truth,
            ins_states=ins_states,
            imu_samples=imu_samples,
            update=update,
            current_step=current_step,
            horizon_end_step=horizon_end_step,
            depth_measurements_by_step=depth_measurements_by_step,
            velocity_measurements_by_step=velocity_measurements_by_step,
            depth_variance_m2=depth_variance_m2,
            velocity_R=velocity_R,
            mode=mode,
            measurement_geometry=measurement_geometry,
            covariance_inflation=covariance_inflation,
            transfer_rw_std_mps=transfer_rw_std_mps,
            nis_threshold=nis_threshold,
            minimum_useful_improvement_m=minimum_useful_improvement_m,
            gain_alpha=alpha,
        )
        if not bool(metrics["applied"]) or not bool(metrics["hmi_safe"]):
            continue
        if float(metrics["error_delta_m"]) < best_error_delta_m:
            best_error_delta_m = float(metrics["error_delta_m"])
            best_metrics = dict(metrics)
    return best_metrics


def _build_runner_config(
    *,
    sequence_spec: GravitySequenceMatcherSpec,
    use_bathymetry: bool,
    use_magnetics: bool,
    use_gradiometer: bool,
    use_current_correction: bool,
) -> SimulationRunnerConfig:
    cfg = SimulationRunnerConfig()
    cfg.map_match.matcher = "sequence"
    cfg.map_match.schedule.every_steps = 1
    cfg.map_match.sequence_spec = sequence_spec
    cfg.map_match.gravity_meas_std_mps2 = float(sequence_spec.gravity_meas_std_mps2)
    cfg.map_match.use_bathymetry = bool(use_bathymetry)
    cfg.map_match.use_magnetics = bool(use_magnetics)
    cfg.map_match.use_gradiometer = bool(use_gradiometer)
    cfg.map_match.bathymetry_meas_std_m = sequence_spec.bathymetry_meas_std_m
    cfg.map_match.magnetic_meas_std_nt = sequence_spec.magnetic_meas_std_nt
    cfg.map_match.magnetic_gradient_meas_std_nt_per_m = (
        sequence_spec.magnetic_gradient_meas_std_nt_per_m
    )
    cfg.velocity_aid.measurement_mode = (
        "water_relative" if use_current_correction else cfg.velocity_aid.measurement_mode
    )
    cfg.velocity_aid.use_current_correction = bool(use_current_correction)
    cfg.observability.enabled = False
    return cfg


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a trust-model corpus for hybrid sequence feedback."
    )
    parser.add_argument(
        "--demo-pack-manifests",
        nargs="+",
        required=True,
        help="Demo-pack manifests to include in the corpus.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output NPZ path for the feedback corpus.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional JSON summary path. Defaults beside --output-path.",
    )
    parser.add_argument(
        "--event-cache-dir",
        default=None,
        help=(
            "Optional directory used to persist per-region/seed replay caches so "
            "gain relabeling can be repeated without rerunning sequence matching."
        ),
    )
    parser.add_argument(
        "--reuse-event-cache",
        action="store_true",
        help=(
            "Reuse existing --event-cache-dir entries instead of rerunning the full "
            "sequence matcher for each region/seed."
        ),
    )
    parser.add_argument(
        "--imu-config",
        default=_relative_to_root(DEFAULT_IMU_CONFIG),
        help="IMU config path.",
    )
    parser.add_argument(
        "--gravimeter-config",
        default=_relative_to_root(DEFAULT_GRAVIMETER_CONFIG),
        help="Gravimeter config path.",
    )
    parser.add_argument(
        "--depth-config",
        default=_relative_to_root(DEFAULT_DEPTH_CONFIG),
        help="Depth config path.",
    )
    parser.add_argument(
        "--velocity-aid-config",
        default=_relative_to_root(DEFAULT_VELOCITY_CONFIG),
        help="Velocity-aid config path.",
    )
    parser.add_argument(
        "--gradiometer-config",
        default=_relative_to_root(DEFAULT_GRADIOMETER_CONFIG),
        help="Gravity gradiometer config path.",
    )
    parser.add_argument(
        "--bathymetry-config",
        default=_relative_to_root(DEFAULT_BATHYMETRY_CONFIG),
        help="Bathymetry-sensor config path.",
    )
    parser.add_argument(
        "--magnetometer-config",
        default=_relative_to_root(DEFAULT_MAGNETOMETER_CONFIG),
        help="Magnetometer config path.",
    )
    parser.add_argument(
        "--current-profile-config",
        default=_relative_to_root(DEFAULT_CURRENT_PROFILE_CONFIG),
        help="Current-profile sensor config path.",
    )
    parser.add_argument(
        "--dt-s",
        type=float,
        default=None,
        help="Optional truth sample interval override.",
    )
    parser.add_argument(
        "--horizon-s",
        type=float,
        default=60.0,
        help="Forward evaluation horizon in seconds.",
    )
    parser.add_argument(
        "--max-events-per-region-seed",
        type=int,
        default=64,
        help="Maximum sampled events per region and seed.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 777],
        help="Root RNG seeds used to generate run variants.",
    )
    parser.add_argument(
        "--sequence-feedback-geometry",
        choices=("directional_horizontal", "full_horizontal"),
        default="directional_horizontal",
        help="Measurement geometry used when generating trust labels.",
    )
    parser.add_argument(
        "--sequence-feedback-inflation",
        type=float,
        default=6.0,
        help="Covariance inflation used when generating trust labels.",
    )
    parser.add_argument(
        "--sequence-feedback-transfer-rw-std-mps",
        type=float,
        default=0.6,
        help="Transfer random-walk inflation for bias_transfer labels.",
    )
    parser.add_argument(
        "--sequence-feedback-nis-threshold",
        type=float,
        default=None,
        help="Optional NIS threshold used when generating trust labels.",
    )
    parser.add_argument(
        "--minimum-useful-improvement-m",
        type=float,
        default=0.0,
        help=(
            "Minimum RMSE improvement, in metres over the forward horizon, required "
            "before a correction is labeled useful."
        ),
    )
    parser.add_argument(
        "--gain-alpha-candidates",
        nargs="+",
        type=float,
        default=list(DEFAULT_GAIN_ALPHA_CANDIDATES),
        help=(
            "Candidate correction gains swept offline when labeling the best safe "
            "feedback strength for each event."
        ),
    )
    return parser


def _sample_candidate_updates(
    updates: list[SequenceMatchUpdateResult],
    *,
    max_events_per_region_seed: int,
    seed: int,
) -> list[SequenceMatchUpdateResult]:
    if len(updates) <= int(max_events_per_region_seed):
        return list(updates)
    rng = np.random.default_rng(int(seed) + 17)
    keep = np.sort(
        rng.choice(
            len(updates),
            size=int(max_events_per_region_seed),
            replace=False,
        )
    )
    return [updates[int(idx)] for idx in keep]


def _build_cached_event_rows(
    *,
    truth: TruthTrajectory,
    ins_states: list[ErrorStateINSState],
    sequence_updates: list[SequenceMatchUpdateResult],
    max_events_per_region_seed: int,
    horizon_s: float,
    seed: int,
) -> list[dict[str, Any]]:
    horizon_steps = max(
        1,
        int(
            round(
                float(horizon_s)
                / max(float(truth.time_s[1] - truth.time_s[0]), 1.0e-6)
            )
        ),
    )
    event_rows: list[dict[str, Any]] = []
    for update in _sample_candidate_updates(
        list(sequence_updates),
        max_events_per_region_seed=max_events_per_region_seed,
        seed=seed,
    ):
        target_step = int(update.global_index)
        current_step = int(update.global_index + update.delayed_by_steps)
        if target_step < 0 or current_step >= len(ins_states):
            continue
        horizon_end_step = min(len(ins_states) - 1, current_step + horizon_steps)
        if horizon_end_step <= current_step:
            continue
        feature_vector, _ = extract_sequence_feedback_features(
            update,
            live_ins_state=ins_states[current_step],
            current_time_s=float(truth.time_s[current_step]),
            previous_live_ins_state=(
                None if current_step <= 0 else ins_states[current_step - 1]
            ),
        )
        event_rows.append(
            {
                "feature_vector": np.asarray(feature_vector, dtype=np.float64).tolist(),
                "current_time_s": float(truth.time_s[current_step]),
                "update_time_s": float(update.time_s),
                "current_step": int(current_step),
                "target_step": int(target_step),
                "horizon_end_step": int(horizon_end_step),
                "update": _serialize_sequence_update(update),
            }
        )
    return event_rows


def _append_region_seed_examples(
    *,
    runner: ScenarioSimulationRunner,
    pack: Any,
    region_name: str,
    cached_run: Mapping[str, Any],
    minimum_useful_improvement_m: float,
    sequence_feedback_geometry: str,
    sequence_feedback_inflation: float,
    sequence_feedback_transfer_rw_std_mps: float,
    sequence_feedback_nis_threshold: float | None,
    gain_alpha_candidates: list[float],
    features: list[np.ndarray],
    region_names: list[str],
    current_time_s: list[float],
    update_time_s: list[float],
    current_step_index: list[int],
    target_step_index: list[int],
    lag_replay_applied: list[bool],
    lag_replay_improves_error: list[bool],
    lag_replay_hmi_safe: list[bool],
    lag_replay_useful_and_safe: list[bool],
    lag_replay_error_delta_m: list[float],
    lag_replay_best_gain_alpha: list[float],
    bias_transfer_applied: list[bool],
    bias_transfer_improves_error: list[bool],
    bias_transfer_hmi_safe: list[bool],
    bias_transfer_useful_and_safe: list[bool],
    bias_transfer_error_delta_m: list[float],
    bias_transfer_best_gain_alpha: list[float],
) -> dict[str, int]:
    truth = cached_run["truth"]
    ins_states = cached_run["ins_states"]
    imu_samples = cached_run["imu_samples"]
    depth_by_step = cached_run["depth_measurements_by_step"]
    velocity_by_step = cached_run["velocity_measurements_by_step"]
    depth_variance_m2 = float(cached_run["depth_variance_m2"])
    velocity_R = np.asarray(cached_run["velocity_R"], dtype=np.float64)
    event_rows = list(cached_run["event_rows"])

    region_event_count = 0
    region_positive_lag = 0
    region_positive_transfer = 0
    for event_row in event_rows:
        update = _deserialize_sequence_update(event_row["update"])
        current_step = int(event_row["current_step"])
        target_step = int(event_row["target_step"])
        horizon_end_step = int(event_row["horizon_end_step"])
        lag_metrics = _select_best_gain_metrics(
            runner=runner,
            truth=truth,
            ins_states=ins_states,
            imu_samples=imu_samples,
            update=update,
            current_step=current_step,
            horizon_end_step=horizon_end_step,
            depth_measurements_by_step=depth_by_step,
            velocity_measurements_by_step=velocity_by_step,
            depth_variance_m2=depth_variance_m2,
            velocity_R=velocity_R,
            mode="lag_replay",
            measurement_geometry=sequence_feedback_geometry,
            covariance_inflation=sequence_feedback_inflation,
            transfer_rw_std_mps=sequence_feedback_transfer_rw_std_mps,
            nis_threshold=sequence_feedback_nis_threshold,
            minimum_useful_improvement_m=minimum_useful_improvement_m,
            gain_alpha_candidates=gain_alpha_candidates,
        )
        transfer_metrics = _select_best_gain_metrics(
            runner=runner,
            truth=truth,
            ins_states=ins_states,
            imu_samples=imu_samples,
            update=update,
            current_step=current_step,
            horizon_end_step=horizon_end_step,
            depth_measurements_by_step=depth_by_step,
            velocity_measurements_by_step=velocity_by_step,
            depth_variance_m2=depth_variance_m2,
            velocity_R=velocity_R,
            mode="bias_transfer",
            measurement_geometry=sequence_feedback_geometry,
            covariance_inflation=sequence_feedback_inflation,
            transfer_rw_std_mps=sequence_feedback_transfer_rw_std_mps,
            nis_threshold=sequence_feedback_nis_threshold,
            minimum_useful_improvement_m=minimum_useful_improvement_m,
            gain_alpha_candidates=gain_alpha_candidates,
        )

        features.append(np.asarray(event_row["feature_vector"], dtype=np.float64))
        region_names.append(str(region_name))
        current_time_s.append(float(event_row["current_time_s"]))
        update_time_s.append(float(event_row["update_time_s"]))
        current_step_index.append(current_step)
        target_step_index.append(target_step)

        lag_replay_applied.append(bool(lag_metrics["applied"]))
        lag_replay_improves_error.append(bool(lag_metrics["improves_error"]))
        lag_replay_hmi_safe.append(bool(lag_metrics["hmi_safe"]))
        lag_replay_useful_and_safe.append(bool(lag_metrics["useful_and_safe"]))
        lag_replay_error_delta_m.append(float(lag_metrics["error_delta_m"]))
        lag_replay_best_gain_alpha.append(float(lag_metrics["best_gain_alpha"]))

        bias_transfer_applied.append(bool(transfer_metrics["applied"]))
        bias_transfer_improves_error.append(bool(transfer_metrics["improves_error"]))
        bias_transfer_hmi_safe.append(bool(transfer_metrics["hmi_safe"]))
        bias_transfer_useful_and_safe.append(bool(transfer_metrics["useful_and_safe"]))
        bias_transfer_error_delta_m.append(float(transfer_metrics["error_delta_m"]))
        bias_transfer_best_gain_alpha.append(
            float(transfer_metrics["best_gain_alpha"])
        )

        region_event_count += 1
        region_positive_lag += int(bool(lag_metrics["useful_and_safe"]))
        region_positive_transfer += int(bool(transfer_metrics["useful_and_safe"]))

    return {
        "num_events": int(region_event_count),
        "lag_replay_positive": int(region_positive_lag),
        "bias_transfer_positive": int(region_positive_transfer),
    }


def main() -> int:
    args = _build_parser().parse_args()

    output_path = _resolve_path(args.output_path)
    summary_path = (
        output_path.with_name(output_path.stem + "_summary.json")
        if args.summary_path is None
        else _resolve_path(args.summary_path)
    )
    event_cache_dir = (
        None
        if args.event_cache_dir is None
        else _resolve_path(args.event_cache_dir)
    )
    if args.reuse_event_cache and event_cache_dir is None:
        raise ValueError("--reuse-event-cache requires --event-cache-dir.")
    if event_cache_dir is not None:
        event_cache_dir.mkdir(parents=True, exist_ok=True)

    imu_spec = _load_spec(IMUSpec, _resolve_path(args.imu_config))
    gravimeter_spec = _load_spec(GravimeterSpec, _resolve_path(args.gravimeter_config))
    depth_spec = _load_spec(DepthSensorSpec, _resolve_path(args.depth_config))
    velocity_spec = _load_spec(VelocityAidSpec, _resolve_path(args.velocity_aid_config))
    gradiometer_spec = _load_spec(
        GravityGradiometerSpec,
        _resolve_path(args.gradiometer_config),
    )
    bathymetry_spec = _load_spec(
        BathymetrySensorSpec,
        _resolve_path(args.bathymetry_config),
    )
    magnetometer_spec = _load_spec(
        MagnetometerSensorSpec,
        _resolve_path(args.magnetometer_config),
    )
    current_profile_spec = _load_spec(
        CurrentProfileSensorSpec,
        _resolve_path(args.current_profile_config),
    )

    features: list[np.ndarray] = []
    region_names: list[str] = []
    current_time_s: list[float] = []
    update_time_s: list[float] = []
    current_step_index: list[int] = []
    target_step_index: list[int] = []

    lag_replay_applied: list[bool] = []
    lag_replay_improves_error: list[bool] = []
    lag_replay_hmi_safe: list[bool] = []
    lag_replay_useful_and_safe: list[bool] = []
    lag_replay_error_delta_m: list[float] = []
    lag_replay_best_gain_alpha: list[float] = []

    bias_transfer_applied: list[bool] = []
    bias_transfer_improves_error: list[bool] = []
    bias_transfer_hmi_safe: list[bool] = []
    bias_transfer_useful_and_safe: list[bool] = []
    bias_transfer_error_delta_m: list[float] = []
    bias_transfer_best_gain_alpha: list[float] = []

    build_rows: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    gain_alpha_candidates = [float(x) for x in args.gain_alpha_candidates]
    sequence_feedback_geometry = str(args.sequence_feedback_geometry)
    sequence_feedback_inflation = float(args.sequence_feedback_inflation)
    sequence_feedback_transfer_rw_std_mps = float(
        args.sequence_feedback_transfer_rw_std_mps
    )
    minimum_useful_improvement_m = float(args.minimum_useful_improvement_m)

    for manifest_text in args.demo_pack_manifests:
        pack = resolve_regional_demo_pack(_resolve_path(manifest_text))
        scenario = ScenarioSpec.from_mapping(load_config_mapping(pack.scenario_path))
        sequence_spec = _load_spec(
            GravitySequenceMatcherSpec,
            pack.sequence_profile_path,
            ignore_unknown=True,
        )
        tide_spec = (
            None
            if pack.tide_config_path is None
            else TideCorrectionSpec(**load_config_mapping(pack.tide_config_path))
        )
        use_bathymetry = pack.bathymetry_grid is not None
        use_magnetics = pack.magnetic_grid is not None
        use_current_correction = pack.current_field is not None
        cfg = _build_runner_config(
            sequence_spec=sequence_spec,
            use_bathymetry=use_bathymetry,
            use_magnetics=use_magnetics,
            use_gradiometer=True,
            use_current_correction=use_current_correction,
        )
        runner = ScenarioSimulationRunner(cfg)
        region_event_count = 0
        region_positive_lag = 0
        region_positive_transfer = 0

        for seed in [int(x) for x in args.seeds]:
            cache_path = (
                None
                if event_cache_dir is None
                else _event_cache_path(
                    event_cache_dir,
                    region_name=str(pack.manifest.region_name),
                    seed=int(seed),
                )
            )
            cache_source = "none"
            if args.reuse_event_cache:
                assert cache_path is not None
                if not cache_path.exists():
                    raise FileNotFoundError(
                        f"Missing feedback-event cache for {pack.manifest.region_name} "
                        f"seed {seed}: {cache_path}"
                    )
                cached_run = load_feedback_event_cache(cache_path)
                cache_source = "reused"
            else:
                result = runner.run_with_specs(
                    scenario_or_truth=scenario,
                    imu_spec=imu_spec,
                    gravimeter_spec=gravimeter_spec,
                    depth_spec=depth_spec,
                    velocity_aid_spec=velocity_spec,
                    gradiometer_spec=gradiometer_spec,
                    bathymetry_spec=(bathymetry_spec if use_bathymetry else None),
                    bathymetry_map=(pack.bathymetry_grid if use_bathymetry else None),
                    magnetometer_spec=(magnetometer_spec if use_magnetics else None),
                    magnetic_map=(pack.magnetic_grid if use_magnetics else None),
                    current_profile_spec=(
                        current_profile_spec if use_current_correction else None
                    ),
                    current_field=(pack.current_field if use_current_correction else None),
                    tide_correction_spec=tide_spec,
                    map_model=pack.gravity_map,
                    dt_s=args.dt_s,
                    seed=seed,
                )
                truth = result.truth
                ins_states = [state.copy() for state in result.estimators.ins_states]
                imu_samples = list(result.sensors.imu_samples)
                depth_by_step = _measurements_by_step(
                    np.asarray(truth.time_s, dtype=np.float64),
                    result.sensors.depth_samples,
                )
                velocity_by_step = _measurements_by_step(
                    np.asarray(truth.time_s, dtype=np.float64),
                    result.sensors.velocity_aid_samples,
                )
                depth_variance_m2 = float(runner.config.depth_aid.measurement_std_m**2)
                velocity_std = np.asarray(
                    runner.config.velocity_aid.measurement_std_mps,
                    dtype=np.float64,
                ).reshape(3)
                velocity_R = np.diag(velocity_std**2)
                event_rows = _build_cached_event_rows(
                    truth=truth,
                    ins_states=ins_states,
                    sequence_updates=list(result.estimators.sequence_updates),
                    max_events_per_region_seed=int(args.max_events_per_region_seed),
                    horizon_s=float(args.horizon_s),
                    seed=seed,
                )
                cached_run = {
                    "truth": truth.copy(),
                    "ins_states": [state.copy() for state in ins_states],
                    "imu_samples": list(imu_samples),
                    "depth_measurements_by_step": list(depth_by_step),
                    "velocity_measurements_by_step": list(velocity_by_step),
                    "depth_variance_m2": depth_variance_m2,
                    "velocity_R": velocity_R,
                    "event_rows": event_rows,
                    "metadata": {
                        "region_name": str(pack.manifest.region_name),
                        "seed": int(seed),
                        "scenario_path": _relative_to_root(pack.scenario_path),
                        "manifest_path": _relative_to_root(pack.manifest_path),
                        "dt_s": None if args.dt_s is None else float(args.dt_s),
                        "horizon_s": float(args.horizon_s),
                    },
                }
                if cache_path is not None:
                    save_feedback_event_cache(
                        cache_path,
                        truth=truth,
                        ins_states=ins_states,
                        imu_samples=imu_samples,
                        depth_measurements_by_step=depth_by_step,
                        velocity_measurements_by_step=velocity_by_step,
                        depth_variance_m2=depth_variance_m2,
                        velocity_R=velocity_R,
                        event_rows=event_rows,
                        metadata=cached_run["metadata"],
                    )
                    cache_source = "written"

            counts = _append_region_seed_examples(
                runner=runner,
                pack=pack,
                region_name=str(pack.manifest.region_name),
                cached_run=cached_run,
                minimum_useful_improvement_m=minimum_useful_improvement_m,
                sequence_feedback_geometry=sequence_feedback_geometry,
                sequence_feedback_inflation=sequence_feedback_inflation,
                sequence_feedback_transfer_rw_std_mps=sequence_feedback_transfer_rw_std_mps,
                sequence_feedback_nis_threshold=args.sequence_feedback_nis_threshold,
                gain_alpha_candidates=gain_alpha_candidates,
                features=features,
                region_names=region_names,
                current_time_s=current_time_s,
                update_time_s=update_time_s,
                current_step_index=current_step_index,
                target_step_index=target_step_index,
                lag_replay_applied=lag_replay_applied,
                lag_replay_improves_error=lag_replay_improves_error,
                lag_replay_hmi_safe=lag_replay_hmi_safe,
                lag_replay_useful_and_safe=lag_replay_useful_and_safe,
                lag_replay_error_delta_m=lag_replay_error_delta_m,
                lag_replay_best_gain_alpha=lag_replay_best_gain_alpha,
                bias_transfer_applied=bias_transfer_applied,
                bias_transfer_improves_error=bias_transfer_improves_error,
                bias_transfer_hmi_safe=bias_transfer_hmi_safe,
                bias_transfer_useful_and_safe=bias_transfer_useful_and_safe,
                bias_transfer_error_delta_m=bias_transfer_error_delta_m,
                bias_transfer_best_gain_alpha=bias_transfer_best_gain_alpha,
            )
            region_event_count += int(counts["num_events"])
            region_positive_lag += int(counts["lag_replay_positive"])
            region_positive_transfer += int(counts["bias_transfer_positive"])
            if cache_path is not None:
                cache_rows.append(
                    {
                        "region_name": str(pack.manifest.region_name),
                        "seed": int(seed),
                        "cache_path": _relative_to_root(cache_path),
                        "cache_source": cache_source,
                        "num_events": int(counts["num_events"]),
                    }
                )

        build_rows.append(
            {
                "region_name": str(pack.manifest.region_name),
                "scenario_path": _relative_to_root(pack.scenario_path),
                "manifest_path": _relative_to_root(pack.manifest_path),
                "num_events": int(region_event_count),
                "lag_replay_positive": int(region_positive_lag),
                "bias_transfer_positive": int(region_positive_transfer),
            }
        )

    if not features:
        raise ValueError("No sequence-feedback corpus examples were generated.")

    unique_regions = tuple(dict.fromkeys(region_names).keys())
    region_lookup = {name: idx for idx, name in enumerate(unique_regions)}
    region_index = np.asarray([region_lookup[name] for name in region_names], dtype=np.int64)

    corpus = SequenceFeedbackEventCorpus(
        feature_names=SEQUENCE_FEEDBACK_FEATURE_NAMES,
        features=np.asarray(features, dtype=np.float64),
        region_names=unique_regions,
        region_index=region_index,
        event_region_names=tuple(region_names),
        current_time_s=np.asarray(current_time_s, dtype=np.float64),
        update_time_s=np.asarray(update_time_s, dtype=np.float64),
        current_step_index=np.asarray(current_step_index, dtype=np.int64),
        target_step_index=np.asarray(target_step_index, dtype=np.int64),
        lag_replay_applied=np.asarray(lag_replay_applied, dtype=bool),
        lag_replay_improves_error=np.asarray(lag_replay_improves_error, dtype=bool),
        lag_replay_hmi_safe=np.asarray(lag_replay_hmi_safe, dtype=bool),
        lag_replay_useful_and_safe=np.asarray(lag_replay_useful_and_safe, dtype=bool),
        lag_replay_error_delta_m=np.asarray(lag_replay_error_delta_m, dtype=np.float64),
        lag_replay_best_gain_alpha=np.asarray(
            lag_replay_best_gain_alpha, dtype=np.float64
        ),
        bias_transfer_applied=np.asarray(bias_transfer_applied, dtype=bool),
        bias_transfer_improves_error=np.asarray(bias_transfer_improves_error, dtype=bool),
        bias_transfer_hmi_safe=np.asarray(bias_transfer_hmi_safe, dtype=bool),
        bias_transfer_useful_and_safe=np.asarray(bias_transfer_useful_and_safe, dtype=bool),
        bias_transfer_error_delta_m=np.asarray(bias_transfer_error_delta_m, dtype=np.float64),
        bias_transfer_best_gain_alpha=np.asarray(
            bias_transfer_best_gain_alpha, dtype=np.float64
        ),
        metadata={
            "entry_point": "scripts/build_sequence_feedback_corpus.py",
            "demo_pack_manifests": [_relative_to_root(_resolve_path(p)) for p in args.demo_pack_manifests],
            "horizon_s": float(args.horizon_s),
            "sequence_feedback_geometry": sequence_feedback_geometry,
            "sequence_feedback_inflation": sequence_feedback_inflation,
            "sequence_feedback_transfer_rw_std_mps": sequence_feedback_transfer_rw_std_mps,
            "minimum_useful_improvement_m": minimum_useful_improvement_m,
            "gain_alpha_candidates": gain_alpha_candidates,
            "dt_s": None if args.dt_s is None else float(args.dt_s),
            "seeds": [int(x) for x in args.seeds],
            "build_rows": build_rows,
            "event_cache_dir": None if event_cache_dir is None else _relative_to_root(event_cache_dir),
            "reuse_event_cache": bool(args.reuse_event_cache),
            "cache_rows": cache_rows,
        },
    )
    corpus.save_npz(output_path)

    summary = {
        "output_path": str(output_path),
        "num_examples": int(corpus.num_examples),
        "num_regions": int(len(corpus.region_names)),
        "region_example_counts": corpus.region_example_counts(),
        "minimum_useful_improvement_m": minimum_useful_improvement_m,
        "lag_replay_positive_fraction": float(np.mean(corpus.lag_replay_useful_and_safe.astype(np.float64))),
        "lag_replay_mean_best_gain_alpha": float(
            np.mean(corpus.lag_replay_best_gain_alpha.astype(np.float64))
        ),
        "bias_transfer_positive_fraction": float(np.mean(corpus.bias_transfer_useful_and_safe.astype(np.float64))),
        "bias_transfer_mean_best_gain_alpha": float(
            np.mean(corpus.bias_transfer_best_gain_alpha.astype(np.float64))
        ),
        "lag_replay_applied_fraction": float(np.mean(corpus.lag_replay_applied.astype(np.float64))),
        "bias_transfer_applied_fraction": float(np.mean(corpus.bias_transfer_applied.astype(np.float64))),
        "build_rows": build_rows,
        "event_cache_dir": None if event_cache_dir is None else str(event_cache_dir),
        "reuse_event_cache": bool(args.reuse_event_cache),
        "cache_rows": cache_rows,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote feedback corpus: {output_path}")
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
