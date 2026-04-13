#!/usr/bin/env python3
"""
Run the photonic gravimeter digital-twin calibration matrix.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, fields
import json
from math import cos, pi, sin
from pathlib import Path
import sys
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from gravnav.physics.earth import normal_gravity
from gravnav.sensors.photonic_gravimeter import (
    PhotonicGravimeterSensor,
    PhotonicGravimeterSpec,
    summarize_photonic_measurements,
)
from gravnav.utils.config import load_config_mapping

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/outputs/reports/photonic_calibration"
LAB_STATIC_CONFIG = PROJECT_ROOT / "configs/sensors/photonic_gravimeter_lab_static.json"
MARITIME_BENIGN_CONFIG = PROJECT_ROOT / "configs/sensors/photonic_gravimeter_maritime_benign.json"
MARITIME_ROUGH_CONFIG = PROJECT_ROOT / "configs/sensors/photonic_gravimeter_maritime_rough.json"


def _instantiate_dataclass(cls: type[Any], mapping: dict[str, Any]) -> Any:
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise KeyError(f"Unsupported keys for {cls.__name__}: {unknown}")
    return cls(**{k: v for k, v in mapping.items() if k in allowed})


def _load_spec(path: Path) -> PhotonicGravimeterSpec:
    return _instantiate_dataclass(PhotonicGravimeterSpec, dict(load_config_mapping(path)))


def _body_to_ned_dcm(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    cr = cos(roll_rad)
    sr = sin(roll_rad)
    cp = cos(pitch_rad)
    sp = sin(pitch_rad)
    cy = cos(yaw_rad)
    sy = sin(yaw_rad)
    return np.array(
        [
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, sr * cp, cr * cp],
        ],
        dtype=np.float64,
    )


def _static_profile(time_s: float) -> dict[str, np.ndarray | float]:
    roll_rad = np.deg2rad(0.05) * sin(2.0 * pi * 0.01 * time_s)
    pitch_rad = np.deg2rad(0.04) * sin(2.0 * pi * 0.013 * time_s + 0.4)
    yaw_rad = 0.0
    roll_dot = np.deg2rad(0.05) * 2.0 * pi * 0.01 * cos(2.0 * pi * 0.01 * time_s)
    pitch_dot = np.deg2rad(0.04) * 2.0 * pi * 0.013 * cos(2.0 * pi * 0.013 * time_s + 0.4)
    acceleration_ned_mps2 = np.array(
        [0.0, 0.0, 2.0e-4 * sin(2.0 * pi * 0.02 * time_s)],
        dtype=np.float64,
    )
    return {
        "roll_rad": roll_rad,
        "pitch_rad": pitch_rad,
        "yaw_rad": yaw_rad,
        "body_rate_b_radps": np.array([roll_dot, pitch_dot, 0.0], dtype=np.float64),
        "acceleration_ned_mps2": acceleration_ned_mps2,
    }


def _maritime_benign_profile(time_s: float) -> dict[str, np.ndarray | float]:
    roll_rad = np.deg2rad(1.1) * sin(2.0 * pi * 0.03 * time_s) + np.deg2rad(0.35) * sin(
        2.0 * pi * 0.07 * time_s + 0.2
    )
    pitch_rad = np.deg2rad(0.9) * sin(2.0 * pi * 0.04 * time_s + 0.3)
    yaw_rad = np.deg2rad(2.0) * sin(2.0 * pi * 0.005 * time_s)
    roll_dot = np.deg2rad(1.1) * 2.0 * pi * 0.03 * cos(2.0 * pi * 0.03 * time_s) + np.deg2rad(
        0.35
    ) * 2.0 * pi * 0.07 * cos(2.0 * pi * 0.07 * time_s + 0.2)
    pitch_dot = np.deg2rad(0.9) * 2.0 * pi * 0.04 * cos(2.0 * pi * 0.04 * time_s + 0.3)
    yaw_dot = np.deg2rad(2.0) * 2.0 * pi * 0.005 * cos(2.0 * pi * 0.005 * time_s)
    acceleration_ned_mps2 = np.array(
        [
            1.5e-3 * sin(2.0 * pi * 0.05 * time_s),
            1.2e-3 * sin(2.0 * pi * 0.06 * time_s + 1.0),
            3.0e-3 * sin(2.0 * pi * 0.05 * time_s + 0.5)
            + 2.0e-3 * sin(2.0 * pi * 0.11 * time_s),
        ],
        dtype=np.float64,
    )
    return {
        "roll_rad": roll_rad,
        "pitch_rad": pitch_rad,
        "yaw_rad": yaw_rad,
        "body_rate_b_radps": np.array([roll_dot, pitch_dot, yaw_dot], dtype=np.float64),
        "acceleration_ned_mps2": acceleration_ned_mps2,
    }


def _maritime_rough_profile(time_s: float) -> dict[str, np.ndarray | float]:
    roll_rad = np.deg2rad(2.2) * sin(2.0 * pi * 0.07 * time_s) + np.deg2rad(0.9) * sin(
        2.0 * pi * 0.17 * time_s + 0.2
    )
    pitch_rad = np.deg2rad(1.9) * sin(2.0 * pi * 0.09 * time_s + 0.5) + np.deg2rad(0.7) * sin(
        2.0 * pi * 0.19 * time_s
    )
    yaw_rad = np.deg2rad(3.0) * sin(2.0 * pi * 0.012 * time_s)
    roll_dot = np.deg2rad(2.2) * 2.0 * pi * 0.07 * cos(2.0 * pi * 0.07 * time_s) + np.deg2rad(
        0.9
    ) * 2.0 * pi * 0.17 * cos(2.0 * pi * 0.17 * time_s + 0.2)
    pitch_dot = np.deg2rad(1.9) * 2.0 * pi * 0.09 * cos(2.0 * pi * 0.09 * time_s + 0.5) + np.deg2rad(
        0.7
    ) * 2.0 * pi * 0.19 * cos(2.0 * pi * 0.19 * time_s)
    yaw_dot = np.deg2rad(3.0) * 2.0 * pi * 0.012 * cos(2.0 * pi * 0.012 * time_s)
    acceleration_ned_mps2 = np.array(
        [
            2.8e-3 * sin(2.0 * pi * 0.08 * time_s + 0.3),
            2.4e-3 * sin(2.0 * pi * 0.09 * time_s + 1.0),
            8.0e-3 * sin(2.0 * pi * 0.08 * time_s + 0.5)
            + 5.0e-3 * sin(2.0 * pi * 0.21 * time_s),
        ],
        dtype=np.float64,
    )
    return {
        "roll_rad": roll_rad,
        "pitch_rad": pitch_rad,
        "yaw_rad": yaw_rad,
        "body_rate_b_radps": np.array([roll_dot, pitch_dot, yaw_dot], dtype=np.float64),
        "acceleration_ned_mps2": acceleration_ned_mps2,
    }


def _run_trace(
    spec: PhotonicGravimeterSpec,
    *,
    profile_fn: Callable[[float], dict[str, np.ndarray | float]],
    duration_s: float,
    dt_s: float,
    seed: int,
) -> dict[str, Any]:
    sensor = PhotonicGravimeterSensor(spec, rng=np.random.default_rng(seed))
    gamma_mps2 = float(normal_gravity(0.0, 0.0))
    samples = []
    for idx in range(int(np.floor(duration_s / dt_s)) + 1):
        time_s = idx * dt_s
        state = profile_fn(time_s)
        roll_rad = float(state["roll_rad"])
        pitch_rad = float(state["pitch_rad"])
        yaw_rad = float(state["yaw_rad"])
        acceleration_ned_mps2 = np.asarray(state["acceleration_ned_mps2"], dtype=np.float64)
        C_n_b = _body_to_ned_dcm(roll_rad, pitch_rad, yaw_rad)
        specific_force_ned_mps2 = acceleration_ned_mps2 - np.array([0.0, 0.0, gamma_mps2], dtype=np.float64)
        specific_force_body_mps2 = C_n_b.T @ specific_force_ned_mps2
        measurement = sensor.measure_disturbance_from_specific_force_body(
            specific_force_body_mps2=specific_force_body_mps2,
            C_n_b=C_n_b,
            v_dot_ned_mps2=acceleration_ned_mps2,
            v_ned_mps=np.zeros(3, dtype=np.float64),
            lat_rad=0.0,
            height_m=0.0,
            dt_s=dt_s,
            body_angular_rate_b_radps=np.asarray(state["body_rate_b_radps"], dtype=np.float64),
            time_s=time_s,
        )
        samples.append(measurement)
    return {
        "samples": samples,
        "summary": asdict(summarize_photonic_measurements(samples)),
    }


def _without_compensation(spec: PhotonicGravimeterSpec) -> PhotonicGravimeterSpec:
    out = deepcopy(spec)
    out.vibration_compensation.enabled = False
    out.vibration_compensation.residual_correction_fraction = 0.0
    out.vibration_compensation.accelerometer_noise_density_mps2_per_sqrt_hz = 0.0
    out.vibration_compensation.accelerometer_bias_random_walk_mps2_per_sqrt_s = 0.0
    out.vibration_compensation.accelerometer_bias_mps2 = 0.0
    out.vibration_compensation.accelerometer_alignment_error_rad = 0.0
    out.vibration_compensation.accelerometer_latency_s = 0.0
    return out


def _static_acceptance(spec: PhotonicGravimeterSpec) -> dict[str, Any]:
    gamma_mps2 = float(normal_gravity(0.0, 0.0))
    sensor = PhotonicGravimeterSensor(spec, rng=np.random.default_rng(7))
    zero_meas = sensor.measure_disturbance_from_specific_force_body(
        specific_force_body_mps2=np.array([0.0, 0.0, -gamma_mps2], dtype=np.float64),
        C_n_b=np.eye(3, dtype=np.float64),
        v_dot_ned_mps2=np.zeros(3, dtype=np.float64),
        v_ned_mps=np.zeros(3, dtype=np.float64),
        lat_rad=0.0,
        height_m=0.0,
        dt_s=1.0,
        body_angular_rate_b_radps=np.zeros(3, dtype=np.float64),
        time_s=0.0,
    )
    delta_mps2 = 1.0e-5
    ratios = []
    for interrogation_time_s in (0.08, 0.16):
        scaled_spec = deepcopy(spec)
        scaled_spec.interferometer.interrogation_time_s = interrogation_time_s
        scaled_sensor = PhotonicGravimeterSensor(scaled_spec, rng=np.random.default_rng(8))
        scaled_meas = scaled_sensor.measure_disturbance_from_specific_force_body(
            specific_force_body_mps2=np.array([0.0, 0.0, -(gamma_mps2 + delta_mps2)], dtype=np.float64),
            C_n_b=np.eye(3, dtype=np.float64),
            v_dot_ned_mps2=np.zeros(3, dtype=np.float64),
            v_ned_mps=np.zeros(3, dtype=np.float64),
            lat_rad=0.0,
            height_m=0.0,
            dt_s=1.0,
            body_angular_rate_b_radps=np.zeros(3, dtype=np.float64),
            time_s=0.0,
        )
        ratios.append(float(scaled_meas.telemetry.gravity_phase_rad))

    gradient_spec = deepcopy(spec)
    gradient_spec.operating_mode = "mission"
    gradient_spec.warmup_time_s = 0.0
    gradient_spec.noise_density_mps2_per_sqrt_hz = 0.0
    gradient_spec.bias_random_walk_mps2_per_sqrt_s = 0.0
    gradient_spec.turn_on_bias_std_mps2 = 0.0
    gradient_spec.fixed_bias_mps2 = 0.0
    gradient_spec.systematics.use_normal_vertical_gradient = False
    gradient_spec.systematics.gravity_gradient_zz_per_s2 = 3.0e-6
    gradient_sensor = PhotonicGravimeterSensor(gradient_spec, rng=np.random.default_rng(9))
    gradient_meas = gradient_sensor.measure_disturbance_from_specific_force_body(
        specific_force_body_mps2=np.array([0.0, 0.0, -gamma_mps2], dtype=np.float64),
        C_n_b=np.eye(3, dtype=np.float64),
        v_dot_ned_mps2=np.zeros(3, dtype=np.float64),
        v_ned_mps=np.zeros(3, dtype=np.float64),
        lat_rad=0.0,
        height_m=0.0,
        dt_s=1.0,
        body_angular_rate_b_radps=np.zeros(3, dtype=np.float64),
        time_s=0.0,
    )
    wavefront_spec = deepcopy(spec)
    wavefront_spec.operating_mode = "mission"
    wavefront_spec.warmup_time_s = 0.0
    wavefront_spec.noise_density_mps2_per_sqrt_hz = 0.0
    wavefront_spec.bias_random_walk_mps2_per_sqrt_s = 0.0
    wavefront_spec.turn_on_bias_std_mps2 = 0.0
    wavefront_spec.fixed_bias_mps2 = 0.0
    wavefront_spec.systematics.wavefront_aberration_coeff_rad_per_m2 = 200.0
    wavefront_sensor = PhotonicGravimeterSensor(wavefront_spec, rng=np.random.default_rng(10))
    wavefront_meas = wavefront_sensor.measure_disturbance_from_specific_force_body(
        specific_force_body_mps2=np.array([0.0, 0.0, -gamma_mps2], dtype=np.float64),
        C_n_b=np.eye(3, dtype=np.float64),
        v_dot_ned_mps2=np.zeros(3, dtype=np.float64),
        v_ned_mps=np.zeros(3, dtype=np.float64),
        lat_rad=0.0,
        height_m=0.0,
        dt_s=1.0,
        body_angular_rate_b_radps=np.zeros(3, dtype=np.float64),
        time_s=0.0,
    )
    scale_ratio = ratios[1] / ratios[0] if abs(ratios[0]) > 0.0 else float("nan")
    gradient_term_m = (
        gradient_spec.atom_ensemble.launch_position_axis_m
        + gradient_spec.atom_ensemble.launch_velocity_axis_mps
        * gradient_spec.interferometer.interrogation_time_s
        - (7.0 / 12.0)
        * gamma_mps2
        * gradient_spec.interferometer.interrogation_time_s**2
    )
    expected_gradient_sign = float(
        np.sign(gradient_spec.systematics.gravity_gradient_zz_per_s2 * gradient_term_m)
    )
    checks = {
        "zero_disturbance_abs_mps2": abs(float(zero_meas.value_mps2)),
        "scale_ratio": float(scale_ratio),
        "gradient_phase_rad": float(gradient_meas.telemetry.gradient_phase_rad),
        "wavefront_phase_rad": float(wavefront_meas.telemetry.wavefront_phase_rad),
        "expected_gradient_sign": expected_gradient_sign,
    }
    acceptance = {
        "zero_disturbance_ok": checks["zero_disturbance_abs_mps2"] <= 1.0e-10,
        "scale_ratio_ok": abs(checks["scale_ratio"] - 4.0) <= 1.0e-3,
        "gradient_sign_ok": bool(
            np.sign(checks["gradient_phase_rad"]) == expected_gradient_sign
        ),
        "systematics_visible": abs(checks["wavefront_phase_rad"]) > 0.0,
    }
    return {"checks": checks, "acceptance": acceptance}


def _dynamic_acceptance(
    compensated_summary: dict[str, Any],
    uncompensated_summary: dict[str, Any],
    *,
    kind: str,
    reference_valid_fraction: float | None = None,
) -> dict[str, Any]:
    compensated_rms = float(compensated_summary["rms_vibration_residual_phase_rad"])
    uncompensated_rms = float(uncompensated_summary["rms_vibration_residual_phase_rad"])
    improvement_ratio = float(uncompensated_rms / compensated_rms) if compensated_rms > 0.0 else float("inf")
    valid_fraction = float(compensated_summary["valid_sample_fraction"])
    median_contrast = float(compensated_summary["median_fringe_contrast"])
    dominant_reason = (
        None
        if len(compensated_summary["rejection_reason_counts"]) == 0
        else max(
            sorted(compensated_summary["rejection_reason_counts"]),
            key=lambda key: compensated_summary["rejection_reason_counts"][key],
        )
    )
    if kind == "benign":
        acceptance = {
            "vibration_improvement_ok": improvement_ratio >= 10.0,
            "valid_fraction_ok": valid_fraction >= 0.85,
            "median_contrast_ok": median_contrast >= 0.45,
        }
    else:
        acceptance = {
            "vibration_improvement_ok": improvement_ratio >= 3.0,
            "valid_fraction_degraded": (
                True
                if reference_valid_fraction is None
                else valid_fraction < float(reference_valid_fraction)
            ),
            "physical_rejection_reason_ok": dominant_reason in {None, "low_contrast", "tilt_limit"},
        }
    return {
        "compensated_summary": compensated_summary,
        "uncompensated_summary": uncompensated_summary,
        "vibration_improvement_ratio": improvement_ratio,
        "dominant_rejection_reason": dominant_reason,
        "acceptance": acceptance,
    }


def _write_report(path: Path, *, calibration: dict[str, Any]) -> None:
    lines = [
        "# Photonic Digital Twin Calibration Report",
        "",
        "## Presets",
        "",
        "- `lab_static`: idealized digital twin for static and quasi-static checks",
        "- `maritime_benign`: mission-mode digital twin for moderate maritime motion",
        "- `maritime_rough`: degraded digital twin for rough maritime motion",
        "",
        "## Results",
        "",
    ]
    static = calibration["lab_static"]
    lines.extend(
        [
            "### lab_static",
            "",
            f"- zero-disturbance abs error [m/s^2]: `{static['checks']['zero_disturbance_abs_mps2']:.3e}`",
            f"- gravity phase scale ratio for doubled `T`: `{static['checks']['scale_ratio']:.6f}`",
            f"- gradient phase [rad]: `{static['checks']['gradient_phase_rad']:.3e}`",
            f"- wavefront phase [rad]: `{static['checks']['wavefront_phase_rad']:.3e}`",
            f"- acceptance: `{all(static['acceptance'].values())}`",
            "",
        ]
    )
    for key in ("maritime_benign", "maritime_rough"):
        block = calibration[key]
        summary = block["compensated_summary"]
        lines.extend(
            [
                f"### {key}",
                "",
                f"- valid fraction: `{summary['valid_sample_fraction']:.3f}`",
                f"- median fringe contrast: `{summary['median_fringe_contrast']:.3f}`",
                f"- p95 fringe contrast: `{summary['p95_fringe_contrast']:.3f}`",
                f"- RMS vibration residual phase [rad]: `{summary['rms_vibration_residual_phase_rad']:.3e}`",
                f"- RMS disturbance residual [m/s^2]: `{summary['rms_disturbance_residual_mps2']:.3e}`",
                f"- tilt exceedance fraction: `{summary['tilt_exceedance_fraction']:.3f}`",
                f"- dominant rejection reason: `{block['dominant_rejection_reason']}`",
                f"- vibration improvement ratio vs uncompensated: `{block['vibration_improvement_ratio']:.3f}`",
                f"- acceptance: `{all(block['acceptance'].values())}`",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the photonic digital-twin calibration matrix.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lab_static_spec = _load_spec(LAB_STATIC_CONFIG)
    benign_spec = _load_spec(MARITIME_BENIGN_CONFIG)
    rough_spec = _load_spec(MARITIME_ROUGH_CONFIG)

    lab_static = _static_acceptance(lab_static_spec)
    benign_compensated = _run_trace(
        benign_spec,
        profile_fn=_maritime_benign_profile,
        duration_s=600.0,
        dt_s=1.0,
        seed=11,
    )["summary"]
    benign_uncompensated = _run_trace(
        _without_compensation(benign_spec),
        profile_fn=_maritime_benign_profile,
        duration_s=600.0,
        dt_s=1.0,
        seed=11,
    )["summary"]
    rough_compensated = _run_trace(
        rough_spec,
        profile_fn=_maritime_rough_profile,
        duration_s=600.0,
        dt_s=1.0,
        seed=13,
    )["summary"]
    rough_uncompensated = _run_trace(
        _without_compensation(rough_spec),
        profile_fn=_maritime_rough_profile,
        duration_s=600.0,
        dt_s=1.0,
        seed=13,
    )["summary"]

    benign = _dynamic_acceptance(
        benign_compensated,
        benign_uncompensated,
        kind="benign",
    )
    rough = _dynamic_acceptance(
        rough_compensated,
        rough_uncompensated,
        kind="rough",
        reference_valid_fraction=float(benign_compensated["valid_sample_fraction"]),
    )
    calibration = {
        "lab_static": lab_static,
        "maritime_benign": benign,
        "maritime_rough": rough,
    }

    summary_path = output_dir / "photonic_calibration_summary.json"
    summary_path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    report_path = output_dir / "photonic_calibration_report.md"
    _write_report(report_path, calibration=calibration)

    print(f"Summary JSON: {summary_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
