#!/usr/bin/env python3
"""
Build a hybrid sequence-feedback trust corpus from real multimodal demo packs.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, TypeVar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from gravnav.datasets import resolve_regional_demo_pack
from gravnav.estimators.error_state_ins import ErrorStateINS, ErrorStateINSState
from gravnav.estimators.feedback_policy import SequenceFeedbackController, SequenceFeedbackSpec
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
from gravnav.sensors.magnetometer import MagnetometerSensorSpec
from gravnav.sensors.imu import IMUSpec
from gravnav.sensors.velocity_aid import VelocityAidMeasurement, VelocityAidSpec
from gravnav.simulation.runner import (
    GravitySequenceMatcherSpec,
    ScenarioSimulationRunner,
    SimulationRunnerConfig,
)
from gravnav.truth.scenarios import ScenarioSpec
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
DEFAULT_GAIN_ALPHA_CANDIDATES = (0.1, 0.25, 0.5, 0.75, 1.0)


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
        default=[42],
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


def main() -> int:
    args = _build_parser().parse_args()

    output_path = _resolve_path(args.output_path)
    summary_path = (
        output_path.with_name(output_path.stem + "_summary.json")
        if args.summary_path is None
        else _resolve_path(args.summary_path)
    )

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
            horizon_steps = max(
                1,
                int(
                    round(
                        float(args.horizon_s)
                        / max(float(truth.time_s[1] - truth.time_s[0]), 1.0e-6)
                    )
                ),
            )

            candidate_updates = list(result.estimators.sequence_updates)
            if len(candidate_updates) > int(args.max_events_per_region_seed):
                rng = np.random.default_rng(seed + 17)
                keep = np.sort(
                    rng.choice(
                        len(candidate_updates),
                        size=int(args.max_events_per_region_seed),
                        replace=False,
                    )
                )
                candidate_updates = [candidate_updates[int(idx)] for idx in keep]

            for update in candidate_updates:
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
                    measurement_geometry=args.sequence_feedback_geometry,
                    covariance_inflation=float(args.sequence_feedback_inflation),
                    transfer_rw_std_mps=float(
                        args.sequence_feedback_transfer_rw_std_mps
                    ),
                    nis_threshold=args.sequence_feedback_nis_threshold,
                    minimum_useful_improvement_m=float(
                        args.minimum_useful_improvement_m
                    ),
                    gain_alpha_candidates=[float(x) for x in args.gain_alpha_candidates],
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
                    measurement_geometry=args.sequence_feedback_geometry,
                    covariance_inflation=float(args.sequence_feedback_inflation),
                    transfer_rw_std_mps=float(
                        args.sequence_feedback_transfer_rw_std_mps
                    ),
                    nis_threshold=args.sequence_feedback_nis_threshold,
                    minimum_useful_improvement_m=float(
                        args.minimum_useful_improvement_m
                    ),
                    gain_alpha_candidates=[float(x) for x in args.gain_alpha_candidates],
                )

                features.append(feature_vector)
                region_names.append(str(pack.manifest.region_name))
                current_time_s.append(float(truth.time_s[current_step]))
                update_time_s.append(float(update.time_s))
                current_step_index.append(int(current_step))
                target_step_index.append(int(target_step))

                lag_replay_applied.append(bool(lag_metrics["applied"]))
                lag_replay_improves_error.append(bool(lag_metrics["improves_error"]))
                lag_replay_hmi_safe.append(bool(lag_metrics["hmi_safe"]))
                lag_replay_useful_and_safe.append(bool(lag_metrics["useful_and_safe"]))
                lag_replay_error_delta_m.append(float(lag_metrics["error_delta_m"]))
                lag_replay_best_gain_alpha.append(
                    float(lag_metrics["best_gain_alpha"])
                )

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
            "sequence_feedback_geometry": str(args.sequence_feedback_geometry),
            "sequence_feedback_inflation": float(args.sequence_feedback_inflation),
            "sequence_feedback_transfer_rw_std_mps": float(args.sequence_feedback_transfer_rw_std_mps),
            "minimum_useful_improvement_m": float(args.minimum_useful_improvement_m),
            "gain_alpha_candidates": [float(x) for x in args.gain_alpha_candidates],
            "dt_s": None if args.dt_s is None else float(args.dt_s),
            "seeds": [int(x) for x in args.seeds],
            "build_rows": build_rows,
        },
    )
    corpus.save_npz(output_path)

    summary = {
        "output_path": str(output_path),
        "num_examples": int(corpus.num_examples),
        "num_regions": int(len(corpus.region_names)),
        "region_example_counts": corpus.region_example_counts(),
        "minimum_useful_improvement_m": float(args.minimum_useful_improvement_m),
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
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote feedback corpus: {output_path}")
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
