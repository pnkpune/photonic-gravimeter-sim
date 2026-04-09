#!/usr/bin/env python3
"""
run_single_scenario.py

Minimal CLI entry point for one end-to-end gravity-aided navigation run.

This script keeps the launch path practical:
- load one scenario from a built-in name or explicit config file
- load one set of sensor specs from explicit JSON/YAML/TOML configs
- build or load a gravity map
- run the full simulator stack
- save the run archive, summary, metrics, and effective config

The default path uses JSON configs so the script works even when PyYAML is not
installed.
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

from gravnav.estimators.map_match_pf import MapMatchPFSpec
from gravnav.physics.gravity_map import (
    GaussianAnomalySource,
    GravityGridMap,
    SinusoidalAnomalySource,
    build_synthetic_gravity_map,
)
from gravnav.sensors.depth import DepthSensorSpec
from gravnav.sensors.gravimeter import GravimeterSpec
from gravnav.sensors.gravity_gradiometer import GravityGradiometerSpec
from gravnav.sensors.imu import IMUSpec
from gravnav.sensors.velocity_aid import VelocityAidSpec
from gravnav.simulation.metrics import ScenarioMetricsSummary, scenario_metrics_from_result
from gravnav.simulation.results import SimulationMetadata
from gravnav.simulation.runner import (
    DepthFusionConfig,
    IntegrityMonitorConfig,
    GravitySequenceMatcherSpec,
    MapMatchFeedbackConfig,
    PeriodicUpdateSchedule,
    ScenarioSimulationRunner,
    SimulationRunnerConfig,
    VelocityAidFusionConfig,
)
from gravnav.truth.scenarios import (
    ScenarioSpec,
    build_truth_trajectory_from_scenario,
    get_named_scenario,
)
from gravnav.utils.config import load_config_mapping

SpecT = TypeVar("SpecT")

DEFAULT_SCENARIO_CONFIG = PROJECT_ROOT / "configs/scenarios/maritime_baseline.json"
DEFAULT_IMU_CONFIG = PROJECT_ROOT / "configs/sensors/imu_nav_grade.json"
DEFAULT_GRAVIMETER_CONFIG = PROJECT_ROOT / "configs/sensors/gravimeter_proto.json"
DEFAULT_DEPTH_CONFIG = PROJECT_ROOT / "configs/sensors/depth_sensor.json"
DEFAULT_VELOCITY_AID_CONFIG = PROJECT_ROOT / "configs/sensors/velocity_aid.json"
DEFAULT_GRADIOMETER_CONFIG = PROJECT_ROOT / "configs/sensors/gravity_gradiometer_proto.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/outputs/runs/single_run"


def _relative_to_root(path: Path) -> str:
    """Pretty path relative to the project root when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _looks_like_path(text: str) -> bool:
    """Heuristic for deciding whether a CLI string should be treated as a path."""
    return any(sep in text for sep in ("/", "\\")) or Path(text).suffix != "" or text.startswith((".", "~"))


def _resolve_path(path_like: str | Path) -> Path:
    """Resolve a path relative to the project root when needed."""
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _resolve_json_config_path(path_or_name: str | Path, *, base_dir: Path) -> Path:
    """Resolve either an explicit config path or a bare JSON stem under `base_dir`."""
    text = str(path_or_name).strip()
    if _looks_like_path(text):
        return _resolve_path(text)
    return (base_dir / f"{Path(text).name}.json").resolve()


def _instantiate_dataclass(spec_cls: type[SpecT], mapping: Mapping[str, Any]) -> SpecT:
    """Instantiate a dataclass from a mapping while rejecting unknown keys."""
    if not is_dataclass(spec_cls):
        raise TypeError(f"{spec_cls!r} is not a dataclass type.")

    allowed = {field.name for field in fields(spec_cls)}
    unknown = sorted(set(mapping.keys()) - allowed)
    if unknown:
        raise KeyError(
            f"Unsupported keys for {spec_cls.__name__}: {unknown}. "
            f"Allowed keys: {sorted(allowed)}."
        )

    kwargs = {key: value for key, value in mapping.items() if key in allowed}
    return spec_cls(**kwargs)


def _load_scenario(path_or_name: str) -> tuple[ScenarioSpec, str]:
    """
    Load a scenario from:
    - a bare built-in name,
    - a bare JSON stem under `configs/scenarios`,
    - or an explicit config path.
    """
    text = str(path_or_name).strip()

    if not _looks_like_path(text):
        json_candidate = (DEFAULT_SCENARIO_CONFIG.parent / f"{text}.json").resolve()
        if json_candidate.exists():
            mapping = load_config_mapping(json_candidate)
            return ScenarioSpec.from_mapping(mapping), _relative_to_root(json_candidate)

        try:
            return get_named_scenario(text), f"built-in:{text}"
        except KeyError:
            pass

    config_path = _resolve_json_config_path(text, base_dir=DEFAULT_SCENARIO_CONFIG.parent)
    mapping = load_config_mapping(config_path)
    return ScenarioSpec.from_mapping(mapping), _relative_to_root(config_path)


def _load_sensor_spec(
    spec_cls: type[SpecT],
    path_or_name: str | Path,
    *,
    base_dir: Path,
) -> tuple[SpecT, dict[str, Any], str]:
    """Load one sensor-spec dataclass from a config file."""
    config_path = _resolve_json_config_path(path_or_name, base_dir=base_dir)
    mapping = load_config_mapping(config_path)
    return (
        _instantiate_dataclass(spec_cls, mapping),
        dict(mapping),
        _relative_to_root(config_path),
    )


def _build_synthetic_map_for_truth(
    scenario: ScenarioSpec,
    *,
    truth_lat_rad: np.ndarray,
    truth_lon_rad: np.ndarray,
    truth_height_m: np.ndarray,
    grid_size: int,
    margin_deg: float,
) -> GravityGridMap:
    """
    Build a deterministic synthetic gravity map that covers the truth route.

    The map is deliberately simple and transparent. It is meant to provide a
    runnable baseline, not a mission-realistic geophysical survey product.
    """
    if grid_size < 11:
        raise ValueError("grid_size must be at least 11.")
    if margin_deg <= 0.0:
        raise ValueError("margin_deg must be positive.")

    lat_min = float(np.min(truth_lat_rad))
    lat_max = float(np.max(truth_lat_rad))
    lon_min = float(np.min(truth_lon_rad))
    lon_max = float(np.max(truth_lon_rad))

    lat_span = max(lat_max - lat_min, np.deg2rad(0.01))
    lon_span = max(lon_max - lon_min, np.deg2rad(0.01))
    margin_rad = float(np.deg2rad(margin_deg))

    lat_axis = np.linspace(lat_min - margin_rad, lat_max + margin_rad, grid_size)
    lon_axis = np.linspace(lon_min - margin_rad, lon_max + margin_rad, grid_size)

    lat_center = 0.5 * (lat_min + lat_max)
    lon_center = 0.5 * (lon_min + lon_max)
    ref_height = float(np.mean(truth_height_m))

    gaussian_sources = (
        GaussianAnomalySource.from_mgal(
            center_lat_rad=lat_min + 0.30 * lat_span,
            center_lon_rad=lon_min + 0.25 * lon_span,
            amplitude_mgal=5.0,
            sigma_north_m=1400.0,
            sigma_east_m=1000.0,
            heading_rad=np.deg2rad(25.0),
            reference_height_m=ref_height,
        ),
        GaussianAnomalySource.from_mgal(
            center_lat_rad=lat_min + 0.70 * lat_span,
            center_lon_rad=lon_min + 0.68 * lon_span,
            amplitude_mgal=-3.5,
            sigma_north_m=1700.0,
            sigma_east_m=1300.0,
            heading_rad=np.deg2rad(-15.0),
            reference_height_m=ref_height,
        ),
        GaussianAnomalySource.from_mgal(
            center_lat_rad=lat_min + 0.55 * lat_span,
            center_lon_rad=lon_min + 0.42 * lon_span,
            amplitude_mgal=2.0,
            sigma_north_m=900.0,
            sigma_east_m=1500.0,
            heading_rad=np.deg2rad(60.0),
            reference_height_m=ref_height,
        ),
    )

    sinusoid_sources = (
        SinusoidalAnomalySource.from_mgal(
            origin_lat_rad=lat_center,
            origin_lon_rad=lon_center,
            amplitude_mgal=1.5,
            wavelength_m=8000.0,
            heading_rad=np.deg2rad(35.0),
            reference_height_m=ref_height,
        ),
    )

    return build_synthetic_gravity_map(
        lat_axis_rad=lat_axis,
        lon_axis_rad=lon_axis,
        gaussian_sources=gaussian_sources,
        sinusoid_sources=sinusoid_sources,
        reference_height_m=ref_height,
        bounds_error=False,
        fill_value_mps2=0.0,
        default_method="linear",
        name=f"{scenario.name}_synthetic_map",
        metadata={
            "scenario": scenario.name,
            "generator": "scripts/run_single_scenario.py",
            "grid_size": int(grid_size),
            "margin_deg": float(margin_deg),
        },
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write a JSON payload with stable pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=False)
        f.write("\n")
    return path


def _build_runner_config(args: argparse.Namespace) -> SimulationRunnerConfig:
    """Build the single-run orchestration config from CLI arguments."""
    velocity_aid = VelocityAidFusionConfig(
        enabled=not args.disable_velocity_aid,
        schedule=PeriodicUpdateSchedule(every_steps=args.velocity_update_every_steps),
        measurement_std_mps=args.velocity_measurement_std_mps,
        measurement_frame=args.velocity_measurement_frame,
    )
    depth_aid = DepthFusionConfig(
        enabled=not args.disable_depth_aid,
        schedule=PeriodicUpdateSchedule(every_steps=args.depth_update_every_steps),
        measurement_std_m=args.depth_measurement_std_m,
    )
    map_match = MapMatchFeedbackConfig(
        enabled=not args.disable_map_match,
        matcher=args.map_matcher,
        schedule=PeriodicUpdateSchedule(every_steps=args.map_match_every_steps),
        pf_spec=MapMatchPFSpec(
            num_particles=args.pf_particles,
            init_position_std_m=(50.0, 50.0, 5.0),
            process_position_rw_std_m_per_sqrt_s=(1.0, 1.0, 0.1),
            rejuvenation_std_m=(2.0, 2.0, 0.2),
            ins_position_prior_std_m=(50.0, 50.0, 5.0),
        ),
        sequence_spec=GravitySequenceMatcherSpec(
            window_size=args.sequence_window_size,
            grid_half_span_m=(
                args.sequence_grid_half_span_north_m,
                args.sequence_grid_half_span_east_m,
            ),
            grid_spacing_m=args.sequence_grid_spacing_m,
            transition_std_m=args.sequence_transition_std_m,
            center_prior_std_m=args.sequence_center_prior_std_m,
            gravity_meas_std_mps2=args.pf_gravity_std_mps2,
            gradient_meas_std_per_s2=(
                None
                if getattr(args, "pf_gradient_std_per_s2", None) is None
                else float(args.pf_gradient_std_per_s2)
            ),
            height_std_m=args.sequence_height_std_m,
        ),
        gravity_meas_std_mps2=args.pf_gravity_std_mps2,
        depth_meas_std_m=args.pf_depth_std_m,
        inject_position_to_ins=bool(args.enable_pf_feedback),
        feedback_covariance_inflation=args.pf_feedback_covariance_inflation,
        use_directional_feedback=bool(getattr(args, "use_directional_feedback", False)),
        use_gradiometer=bool(getattr(args, "use_gradiometer", False)),
        gradient_meas_std_per_s2=(
            None
            if getattr(args, "pf_gradient_std_per_s2", None) is None
            else float(args.pf_gradient_std_per_s2)
        ),
        use_sequence_feedback=bool(getattr(args, "use_sequence_feedback", False)),
    )
    map_match.sequence_feedback_spec.min_peak_probability = (
        float(args.sequence_feedback_min_peak_prob)
    )
    map_match.sequence_feedback_spec.mode = str(args.sequence_feedback_mode)
    map_match.sequence_feedback_spec.max_horizontal_std_m = (
        float(args.sequence_feedback_max_horizontal_std_m)
    )
    map_match.sequence_feedback_spec.max_correction_norm_m = (
        float(args.sequence_feedback_max_correction_m)
    )
    map_match.sequence_feedback_spec.covariance_inflation = (
        float(args.sequence_feedback_inflation)
    )
    map_match.sequence_feedback_spec.transfer_rw_std_mps = (
        float(args.sequence_feedback_transfer_rw_std_mps)
    )
    map_match.sequence_feedback_spec.nis_threshold = (
        None
        if args.sequence_feedback_nis_threshold is None
        else float(args.sequence_feedback_nis_threshold)
    )
    integrity = IntegrityMonitorConfig(
        enabled=not args.disable_integrity,
        horizontal_alert_limit_m=args.horizontal_alert_limit_m,
        vertical_alert_limit_m=args.vertical_alert_limit_m,
    )
    return SimulationRunnerConfig(
        velocity_aid=velocity_aid,
        depth_aid=depth_aid,
        map_match=map_match,
        integrity=integrity,
        metadata_extra={"entry_point": "scripts/run_single_scenario.py"},
    )


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run one end-to-end gravity-aided navigation scenario.",
    )
    parser.add_argument(
        "--scenario",
        default="maritime_baseline",
        help=(
            "Built-in scenario name or explicit config path. Bare names prefer "
            "JSON under configs/scenarios before falling back to built-ins."
        ),
    )
    parser.add_argument(
        "--imu-config",
        default=_relative_to_root(DEFAULT_IMU_CONFIG),
        help="IMU spec config path or bare JSON stem under configs/sensors.",
    )
    parser.add_argument(
        "--gravimeter-config",
        default=_relative_to_root(DEFAULT_GRAVIMETER_CONFIG),
        help="Gravimeter spec config path or bare JSON stem under configs/sensors.",
    )
    parser.add_argument(
        "--depth-config",
        default=_relative_to_root(DEFAULT_DEPTH_CONFIG),
        help="Depth-sensor spec config path or bare JSON stem under configs/sensors.",
    )
    parser.add_argument(
        "--velocity-aid-config",
        default=_relative_to_root(DEFAULT_VELOCITY_AID_CONFIG),
        help="Velocity-aid spec config path or bare JSON stem under configs/sensors.",
    )
    parser.add_argument(
        "--gradiometer-config",
        default=_relative_to_root(DEFAULT_GRADIOMETER_CONFIG),
        help="Gravity-gradiometer spec config path or bare JSON stem under configs/sensors.",
    )
    parser.add_argument(
        "--map-path",
        default=None,
        help="Optional NPZ gravity map path. If omitted, a synthetic map is generated.",
    )
    parser.add_argument(
        "--dt-s",
        type=float,
        default=None,
        help="Optional truth sample interval override in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Root RNG seed for the run and synthetic map generation.",
    )
    parser.add_argument(
        "--run-id",
        default="baseline",
        help="Short identifier added to output filenames.",
    )
    parser.add_argument(
        "--output-dir",
        default=_relative_to_root(DEFAULT_OUTPUT_DIR),
        help="Directory where run artifacts are written.",
    )
    parser.add_argument(
        "--map-grid-size",
        type=int,
        default=121,
        help="Synthetic gravity-map grid size per axis when --map-path is omitted.",
    )
    parser.add_argument(
        "--map-margin-deg",
        type=float,
        default=0.02,
        help="Extra geodetic margin in degrees around the truth route for synthetic maps.",
    )
    parser.add_argument(
        "--map-matcher",
        choices=("pf", "sequence"),
        default="pf",
        help="Map-matching algorithm to run.",
    )
    parser.add_argument(
        "--pf-particles",
        type=int,
        default=128,
        help="Particle count for the map-matching PF.",
    )
    parser.add_argument(
        "--sequence-window-size",
        type=int,
        default=9,
        help="Sliding-window length for the sequence matcher.",
    )
    parser.add_argument(
        "--sequence-grid-half-span-north-m",
        type=float,
        default=120.0,
        help="North half-span of the sequence-matcher candidate grid in metres.",
    )
    parser.add_argument(
        "--sequence-grid-half-span-east-m",
        type=float,
        default=120.0,
        help="East half-span of the sequence-matcher candidate grid in metres.",
    )
    parser.add_argument(
        "--sequence-grid-spacing-m",
        type=float,
        nargs=2,
        default=(20.0, 20.0),
        metavar=("DN", "DE"),
        help="North/east candidate-grid spacing for the sequence matcher in metres.",
    )
    parser.add_argument(
        "--sequence-transition-std-m",
        type=float,
        nargs=2,
        default=(20.0, 20.0),
        metavar=("TN", "TE"),
        help="North/east transition standard deviation for the sequence matcher in metres.",
    )
    parser.add_argument(
        "--sequence-center-prior-std-m",
        type=float,
        nargs=2,
        default=(80.0, 80.0),
        metavar=("PN", "PE"),
        help="North/east soft prior standard deviation about the INS center for the sequence matcher in metres.",
    )
    parser.add_argument(
        "--sequence-height-std-m",
        type=float,
        default=2.0,
        help="Vertical covariance floor used by the sequence matcher in metres.",
    )
    parser.add_argument(
        "--map-match-every-steps",
        type=int,
        default=2,
        help="Run one map-matching gravity update every N simulation steps.",
    )
    parser.add_argument(
        "--depth-update-every-steps",
        type=int,
        default=1,
        help="Fuse depth every N simulation steps.",
    )
    parser.add_argument(
        "--velocity-update-every-steps",
        type=int,
        default=1,
        help="Fuse velocity aiding every N simulation steps.",
    )
    parser.add_argument(
        "--velocity-measurement-frame",
        choices=("ned", "body"),
        default="ned",
        help="Velocity-aid measurement frame used in the fusion layer.",
    )
    parser.add_argument(
        "--velocity-measurement-std-mps",
        type=float,
        nargs=3,
        default=(0.05, 0.05, 0.05),
        metavar=("VX_STD", "VY_STD", "VZ_STD"),
        help="Velocity-aid fusion standard deviations in m/s.",
    )
    parser.add_argument(
        "--depth-measurement-std-m",
        type=float,
        default=0.5,
        help="Depth-aiding fusion standard deviation in metres.",
    )
    parser.add_argument(
        "--pf-gravity-std-mps2",
        type=float,
        default=1.0e-5,
        help="PF gravity measurement standard deviation in m/s^2.",
    )
    parser.add_argument(
        "--pf-depth-std-m",
        type=float,
        default=0.5,
        help="PF depth measurement standard deviation in metres when depth is used.",
    )
    parser.add_argument(
        "--pf-gradient-std-per-s2",
        type=float,
        default=1.0e-8,
        help="PF horizontal-gradient measurement standard deviation per axis in 1/s^2.",
    )
    parser.add_argument(
        "--pf-feedback-covariance-inflation",
        type=float,
        default=1.0,
        help="Scalar covariance inflation applied before PF position feedback into the INS.",
    )
    parser.add_argument(
        "--horizontal-alert-limit-m",
        type=float,
        default=100.0,
        help="Horizontal alert limit used by the integrity monitor.",
    )
    parser.add_argument(
        "--vertical-alert-limit-m",
        type=float,
        default=20.0,
        help="Vertical alert limit used by the integrity monitor.",
    )
    parser.add_argument(
        "--disable-map-match",
        action="store_true",
        help="Disable PF gravity map matching entirely.",
    )
    parser.add_argument(
        "--use-gradiometer",
        action="store_true",
        help="Enable horizontal gravity-gradient sampling and PF gradient likelihood.",
    )
    pf_feedback_group = parser.add_mutually_exclusive_group()
    pf_feedback_group.add_argument(
        "--enable-pf-feedback",
        action="store_true",
        help=(
            "Enable PF pseudo-position feedback into the INS. Disabled by "
            "default because the current PF feedback path is still tuning-"
            "sensitive."
        ),
    )
    pf_feedback_group.add_argument(
        "--disable-pf-feedback",
        action="store_true",
        help="Deprecated alias. PF pseudo-position feedback is disabled by default.",
    )
    pf_feedback_group.add_argument(
        "--use-directional-feedback",
        action="store_true",
        help=(
            "Enable observability-aware directional (rank-1) PF feedback. "
            "Mutually exclusive with --enable-pf-feedback (legacy full-3D path)."
        ),
    )
    parser.add_argument(
        "--use-sequence-feedback",
        action="store_true",
        help=(
            "Enable conservative delayed sequence-to-INS horizontal feedback. "
            "Only valid with --map-matcher sequence."
        ),
    )
    parser.add_argument(
        "--sequence-feedback-min-peak-prob",
        type=float,
        default=0.12,
        help="Minimum sequence posterior peak probability required for feedback.",
    )
    parser.add_argument(
        "--sequence-feedback-mode",
        choices=("lag_replay", "bias_transfer"),
        default="lag_replay",
        help=(
            "Delayed sequence feedback architecture. "
            "'lag_replay' applies the estimate at the delayed state and replays "
            "forward; 'bias_transfer' is the older current-state transfer path."
        ),
    )
    parser.add_argument(
        "--sequence-feedback-max-horizontal-std-m",
        type=float,
        default=80.0,
        help="Maximum allowed horizontal sequence standard deviation for feedback in metres.",
    )
    parser.add_argument(
        "--sequence-feedback-max-correction-m",
        type=float,
        default=150.0,
        help="Maximum allowed horizontal correction norm from sequence feedback in metres.",
    )
    parser.add_argument(
        "--sequence-feedback-inflation",
        type=float,
        default=3.0,
        help="Covariance inflation applied to sequence feedback before INS fusion.",
    )
    parser.add_argument(
        "--sequence-feedback-transfer-rw-std-mps",
        type=float,
        default=0.6,
        help="Delay-transfer random-walk inflation for sequence feedback in m/s.",
    )
    parser.add_argument(
        "--sequence-feedback-nis-threshold",
        type=float,
        default=25.0,
        help="Optional NIS gate for delayed sequence feedback.",
    )
    parser.add_argument(
        "--disable-depth-aid",
        action="store_true",
        help="Disable depth aiding.",
    )
    parser.add_argument(
        "--disable-velocity-aid",
        action="store_true",
        help="Disable external velocity aiding.",
    )
    parser.add_argument(
        "--disable-integrity",
        action="store_true",
        help="Disable integrity monitoring.",
    )
    return parser


def _metrics_summary_lines(metrics: ScenarioMetricsSummary) -> list[str]:
    """Small human-readable metrics summary for terminal output."""
    lines = []

    if metrics.ins_position_error is not None:
        lines.append(
            f"INS horizontal RMSE: {metrics.ins_position_error.horizontal_rmse_m:.3f} m"
        )
        lines.append(
            f"INS CEP95: {metrics.ins_position_error.cep95_m:.3f} m"
        )

    if metrics.gravimeter_error is not None:
        lines.append(
            f"Gravimeter RMSE: {metrics.gravimeter_error.rmse:.6e} m/s^2"
        )

    if metrics.pf_position_error is not None:
        lines.append(
            f"PF horizontal RMSE: {metrics.pf_position_error.horizontal_rmse_m:.3f} m"
        )
        lines.append(
            f"PF CEP95: {metrics.pf_position_error.cep95_m:.3f} m"
        )
    if metrics.sequence_position_error is not None:
        lines.append(
            "Sequence horizontal RMSE: "
            f"{metrics.sequence_position_error.horizontal_rmse_m:.3f} m"
        )
        lines.append(
            f"Sequence CEP95: {metrics.sequence_position_error.cep95_m:.3f} m"
        )

    if metrics.integrity is not None:
        nis_fraction = float(metrics.integrity.fraction_nis_passed)
        if np.isfinite(nis_fraction):
            lines.append(
                f"Integrity NIS pass fraction: {nis_fraction:.3f}"
            )

    return lines


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    scenario, scenario_source = _load_scenario(args.scenario)

    imu_spec, imu_cfg, imu_source = _load_sensor_spec(
        IMUSpec,
        args.imu_config,
        base_dir=DEFAULT_IMU_CONFIG.parent,
    )
    gravimeter_spec, gravimeter_cfg, gravimeter_source = _load_sensor_spec(
        GravimeterSpec,
        args.gravimeter_config,
        base_dir=DEFAULT_GRAVIMETER_CONFIG.parent,
    )
    depth_spec, depth_cfg, depth_source = _load_sensor_spec(
        DepthSensorSpec,
        args.depth_config,
        base_dir=DEFAULT_DEPTH_CONFIG.parent,
    )
    velocity_aid_spec, velocity_cfg, velocity_source = _load_sensor_spec(
        VelocityAidSpec,
        args.velocity_aid_config,
        base_dir=DEFAULT_VELOCITY_AID_CONFIG.parent,
    )
    gradiometer_spec = None
    gradiometer_cfg = None
    gradiometer_source = None
    if args.use_gradiometer:
        gradiometer_spec, gradiometer_cfg, gradiometer_source = _load_sensor_spec(
            GravityGradiometerSpec,
            args.gradiometer_config,
            base_dir=DEFAULT_GRADIOMETER_CONFIG.parent,
        )

    truth = build_truth_trajectory_from_scenario(scenario, dt_s=args.dt_s)

    map_source: str
    if args.disable_map_match:
        map_model: GravityGridMap | None = None
        map_source = "disabled"
    elif args.map_path is not None:
        map_path = _resolve_path(args.map_path)
        map_model = GravityGridMap.from_npz(map_path)
        map_source = _relative_to_root(map_path)
    else:
        map_model = _build_synthetic_map_for_truth(
            scenario,
            truth_lat_rad=np.asarray(truth.lat_rad, dtype=np.float64),
            truth_lon_rad=np.asarray(truth.lon_rad, dtype=np.float64),
            truth_height_m=np.asarray(truth.height_m, dtype=np.float64),
            grid_size=args.map_grid_size,
            margin_deg=args.map_margin_deg,
        )
        map_source = "synthetic"

    runner_config = _build_runner_config(args)

    effective_config = {
        "scenario_source": scenario_source,
        "scenario": scenario.to_mapping(),
        "sensor_configs": {
            "imu_source": imu_source,
            "gravimeter_source": gravimeter_source,
            "depth_source": depth_source,
            "velocity_aid_source": velocity_source,
            "imu": imu_cfg,
            "gravimeter": gravimeter_cfg,
            "depth": depth_cfg,
            "velocity_aid": velocity_cfg,
            "gradiometer_source": gradiometer_source,
            "gradiometer": gradiometer_cfg,
        },
        "runner": {
            "dt_s": float(scenario.default_dt_s if args.dt_s is None else args.dt_s),
            "seed": int(args.seed),
            "map_source": map_source,
            "output_dir": _relative_to_root(_resolve_path(args.output_dir)),
            "velocity_aid": {
                "enabled": not args.disable_velocity_aid,
                "every_steps": int(args.velocity_update_every_steps),
                "measurement_std_mps": list(args.velocity_measurement_std_mps),
                "measurement_frame": args.velocity_measurement_frame,
                "velocity_only_update": True,
            },
            "depth_aid": {
                "enabled": not args.disable_depth_aid,
                "every_steps": int(args.depth_update_every_steps),
                "measurement_std_m": float(args.depth_measurement_std_m),
                "height_only_update": True,
            },
            "map_match": {
                "enabled": not args.disable_map_match,
                "matcher": args.map_matcher,
                "every_steps": int(args.map_match_every_steps),
                "num_particles": int(args.pf_particles),
                "gravity_std_mps2": float(args.pf_gravity_std_mps2),
                "depth_std_m": float(args.pf_depth_std_m),
                "inject_position_to_ins": bool(args.enable_pf_feedback),
                "feedback_covariance_inflation": float(
                    args.pf_feedback_covariance_inflation
                ),
                "use_gradiometer": bool(args.use_gradiometer),
                "gradient_std_per_s2": float(args.pf_gradient_std_per_s2),
                "use_sequence_feedback": bool(args.use_sequence_feedback),
                "sequence_feedback_min_peak_prob": float(
                    args.sequence_feedback_min_peak_prob
                ),
                "sequence_feedback_max_horizontal_std_m": float(
                    args.sequence_feedback_max_horizontal_std_m
                ),
                "sequence_feedback_max_correction_m": float(
                    args.sequence_feedback_max_correction_m
                ),
                "sequence_feedback_inflation": float(
                    args.sequence_feedback_inflation
                ),
                "sequence_feedback_transfer_rw_std_mps": float(
                    args.sequence_feedback_transfer_rw_std_mps
                ),
                "sequence_feedback_nis_threshold": (
                    None
                    if args.sequence_feedback_nis_threshold is None
                    else float(args.sequence_feedback_nis_threshold)
                ),
                "sequence_window_size": int(args.sequence_window_size),
                "sequence_grid_half_span_m": [
                    float(args.sequence_grid_half_span_north_m),
                    float(args.sequence_grid_half_span_east_m),
                ],
                "sequence_grid_spacing_m": list(args.sequence_grid_spacing_m),
                "sequence_transition_std_m": list(args.sequence_transition_std_m),
                "sequence_center_prior_std_m": list(args.sequence_center_prior_std_m),
                "sequence_height_std_m": float(args.sequence_height_std_m),
            },
            "integrity": {
                "enabled": not args.disable_integrity,
                "horizontal_alert_limit_m": float(args.horizontal_alert_limit_m),
                "vertical_alert_limit_m": float(args.vertical_alert_limit_m),
            },
            "synthetic_map": None
            if map_model is None or args.map_path is not None
            else {
                "grid_size": int(args.map_grid_size),
                "margin_deg": float(args.map_margin_deg),
            },
        },
    }

    run_id = str(args.run_id).strip() or "baseline"
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{scenario.name}_{run_id}"

    metadata = SimulationMetadata(
        scenario_name=scenario.name,
        run_id=run_id,
        description="Single-scenario CLI run",
        config=effective_config,
        rng_state={"seed": int(args.seed)},
    )

    runner = ScenarioSimulationRunner(runner_config)
    result = runner.run_with_specs(
        truth,
        imu_spec=imu_spec,
        gravimeter_spec=gravimeter_spec,
        depth_spec=None if args.disable_depth_aid else depth_spec,
        velocity_aid_spec=None if args.disable_velocity_aid else velocity_aid_spec,
        gradiometer_spec=gradiometer_spec,
        map_model=map_model,
        metadata=metadata,
        seed=args.seed,
    )

    metrics = scenario_metrics_from_result(result)

    result_path = result.save_npz(output_dir / f"{stem}.npz")
    summary_path = result.save_summary_json(output_dir / f"{stem}_summary.json")
    metrics_path = _write_json(output_dir / f"{stem}_metrics.json", metrics.to_mapping())
    config_path = _write_json(output_dir / f"{stem}_config.json", effective_config)

    map_path: Path | None = None
    if map_model is not None:
        map_path = map_model.to_npz(output_dir / f"{stem}_map.npz")

    print(f"Scenario: {scenario.name}")
    print(f"Scenario source: {scenario_source}")
    print(f"Run ID: {run_id}")
    print(f"Truth samples: {len(result.truth)}")
    print(f"INS states: {len(result.estimators.ins_states)}")
    print(f"PF updates: {len(result.estimators.pf_updates)}")
    print(f"Sequence updates: {len(result.estimators.sequence_updates)}")
    print(f"Integrity snapshots: {len(result.estimators.integrity_snapshots)}")
    for line in _metrics_summary_lines(metrics):
        print(line)
    print(f"Saved run archive: {_relative_to_root(result_path)}")
    print(f"Saved summary: {_relative_to_root(summary_path)}")
    print(f"Saved metrics: {_relative_to_root(metrics_path)}")
    print(f"Saved effective config: {_relative_to_root(config_path)}")
    if map_path is not None:
        print(f"Saved map: {_relative_to_root(map_path)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
