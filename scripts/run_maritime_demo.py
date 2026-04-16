#!/usr/bin/env python3
"""
Run the Norwegian maritime photonic demo and write a compact report.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gravnav_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gravnav_xdg_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gravnav.datasets.bathymetry_loader import (
    BathymetryGrid,
    ResolvedRegionalDemoPack,
    ensure_regional_bathymetry_grid,
    resolve_regional_demo_pack,
)
from gravnav.datasets.current_loader import load_current_field_from_manifest
from gravnav.datasets.magnetic_loader import load_magnetic_grid_from_manifest
from gravnav.physics.gravity_map import GravityGridMap
from gravnav.plots.nav_plots import plot_ground_track_local_ned, plot_position_error_ned
from gravnav.sensors.bathymetry import BathymetrySensorSpec
from gravnav.sensors.current_profile import CurrentProfileSensorSpec
from gravnav.sensors.depth import DepthSensorSpec
from gravnav.sensors.gravimeter import GravimeterSpec
from gravnav.sensors.gravity_gradiometer import GravityGradiometerSpec
from gravnav.sensors.imu import IMUSpec
from gravnav.sensors.magnetometer import MagnetometerSensorSpec
from gravnav.sensors.photonic_gravimeter import PhotonicGravimeterSpec
from gravnav.sensors.photonic_gravimeter import summarize_photonic_measurements
from gravnav.sensors.velocity_aid import VelocityAidSpec
from gravnav.simulation.metrics import (
    PositionErrorMetrics,
    ScenarioMetricsSummary,
    ins_position_error_history_from_truth,
    ins_state_times,
    lag_smoothed_position_error_history_from_truth,
    sequence_position_error_history_from_truth,
    scenario_metrics_from_result,
)
from gravnav.estimators.integrity import protection_levels_from_covariance_ned
from gravnav.ml.runtime import LearnedLocalizerSpec
from gravnav.simulation.results import SimulationMetadata
from gravnav.simulation.runner import (
    DepthFusionConfig,
    GravitySequenceMatcherSpec,
    IntegrityMonitorConfig,
    MapMatchFeedbackConfig,
    PeriodicUpdateSchedule,
    ScenarioSimulationRunner,
    SimulationRunnerConfig,
    TideCorrectionSpec,
    VelocityAidFusionConfig,
)
from gravnav.truth.scenarios import ScenarioSpec
from gravnav.utils.config import load_config_mapping

DEFAULT_SCENARIO = PROJECT_ROOT / "configs/scenarios/norwegian_margin_maritime.json"
DEFAULT_SEQUENCE_PROFILE = PROJECT_ROOT / "configs/sequence_profiles/norwegian_margin_maritime_demo.json"
DEFAULT_IMU_CONFIG = PROJECT_ROOT / "configs/sensors/imu_nav_grade.json"
DEFAULT_GRAVIMETER_CONFIG = PROJECT_ROOT / "configs/sensors/gravimeter_proto.json"
DEFAULT_PHOTONIC_CONFIG = PROJECT_ROOT / "configs/sensors/photonic_gravimeter_maritime_benign.json"
DEFAULT_DEPTH_CONFIG = PROJECT_ROOT / "configs/sensors/depth_sensor.json"
DEFAULT_VELOCITY_CONFIG = PROJECT_ROOT / "configs/sensors/velocity_aid.json"
DEFAULT_GRADIOMETER_CONFIG = PROJECT_ROOT / "configs/sensors/gravity_gradiometer_proto.json"
DEFAULT_BATHY_SENSOR_CONFIG = PROJECT_ROOT / "configs/sensors/bathymetry_sensor.json"
DEFAULT_MAGNETOMETER_CONFIG = PROJECT_ROOT / "configs/sensors/magnetometer_scalar.json"
DEFAULT_CURRENT_PROFILE_CONFIG = PROJECT_ROOT / "configs/sensors/current_profile_sensor.json"
DEFAULT_TIDE_CONFIG = PROJECT_ROOT / "configs/environment/tide_correction_norway.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/outputs/reports/maritime_demo"
DEFAULT_GRAVITY_MAP = PROJECT_ROOT / "data/gravity_maps/processed/norwegian_margin_gravity_map.npz"
DEFAULT_DEMO_PACK = PROJECT_ROOT / "data/bathymetry/processed/norwegian_margin_maritime_demo_pack.json"
PREP_SCRIPT = PROJECT_ROOT / "scripts/prepare_norwegian_maritime_demo.py"


def _instantiate_dataclass(cls: type[Any], mapping: dict[str, Any]) -> Any:
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise KeyError(f"Unsupported keys for {cls.__name__}: {unknown}")
    return cls(**{k: v for k, v in mapping.items() if k in allowed})


def _load_spec(path: Path, cls: type[Any]) -> Any:
    return _instantiate_dataclass(cls, load_config_mapping(path))


def _load_scenario(path: Path) -> ScenarioSpec:
    return ScenarioSpec.from_mapping(load_config_mapping(path))


def _load_sequence_profile(path: Path) -> dict[str, Any]:
    return dict(load_config_mapping(path))


def _load_optional_spec(path: Path | None, cls: type[Any]) -> Any | None:
    if path is None:
        return None
    return _load_spec(path, cls)


def _ensure_demo_pack() -> None:
    if DEFAULT_DEMO_PACK.exists():
        return
    subprocess.run([sys.executable, str(PREP_SCRIPT)], check=True, cwd=PROJECT_ROOT)


def _load_demo_pack_assets(path: Path) -> ResolvedRegionalDemoPack:
    return resolve_regional_demo_pack(path)


def _build_runner_config(
    profile: dict[str, Any],
    *,
    matcher: str,
    learned_localizer_model_path: Path | None,
    use_bathymetry: bool,
    use_acoustic_terrain: bool,
    use_magnetics: bool,
    use_current_correction: bool,
    use_lag_smoother: bool,
    initial_position_offset_ned_m: tuple[float, float, float],
) -> SimulationRunnerConfig:
    map_match = MapMatchFeedbackConfig(
        enabled=True,
        matcher=str(matcher),
        schedule=PeriodicUpdateSchedule(
            every_steps=int(profile.get("map_match_every_steps", 2))
        ),
        sequence_spec=GravitySequenceMatcherSpec(
            window_size=int(profile["window_size"]),
            grid_half_span_m=tuple(profile["grid_half_span_m"]),
            grid_spacing_m=tuple(profile["grid_spacing_m"]),
            transition_std_m=tuple(profile["transition_std_m"]),
            center_prior_std_m=tuple(profile["center_prior_std_m"]),
            gravity_meas_std_mps2=float(profile["gravity_meas_std_mps2"]),
            gradient_meas_std_per_s2=float(profile["gradient_meas_std_per_s2"]),
            bathymetry_meas_std_m=(
                None if not use_bathymetry else float(profile["bathymetry_meas_std_m"])
            ),
            bathymetry_weight=(
                1.0 if not use_bathymetry else float(profile.get("bathymetry_weight", 1.0))
            ),
            bathymetry_gradient_meas_std_m_per_m=(
                None
                if not (use_bathymetry and use_acoustic_terrain)
                else float(profile.get("bathymetry_gradient_meas_std_m_per_m", 0.01))
            ),
            bathymetry_gradient_weight=(
                1.0
                if not (use_bathymetry and use_acoustic_terrain)
                else float(profile.get("bathymetry_gradient_weight", 1.0))
            ),
            bathymetry_rugosity_meas_std_m=(
                None
                if not (use_bathymetry and use_acoustic_terrain)
                else float(profile.get("bathymetry_rugosity_meas_std_m", 2.0))
            ),
            bathymetry_rugosity_weight=(
                1.0
                if not (use_bathymetry and use_acoustic_terrain)
                else float(profile.get("bathymetry_rugosity_weight", 1.0))
            ),
            magnetic_meas_std_nt=(
                None if not use_magnetics else float(profile.get("magnetic_meas_std_nt", 8.0))
            ),
            magnetic_weight=(
                1.0 if not use_magnetics else float(profile.get("magnetic_weight", 1.0))
            ),
            magnetic_gradient_meas_std_nt_per_m=(
                None
                if not use_magnetics
                else float(profile.get("magnetic_gradient_meas_std_nt_per_m", 0.02))
            ),
            magnetic_gradient_weight=(
                1.0
                if not use_magnetics
                else float(profile.get("magnetic_gradient_weight", 0.75))
            ),
            height_std_m=float(profile.get("height_std_m", 2.0)),
            adaptive_grid_enabled=bool(profile.get("adaptive_grid_enabled", True)),
            expanded_grid_half_span_m=profile.get(
                "expanded_grid_half_span_m",
                None,
            ),
            expanded_grid_spacing_m=profile.get(
                "expanded_grid_spacing_m",
                None,
            ),
            adaptive_expand_edge_mass_fraction=float(
                profile.get("adaptive_expand_edge_mass_fraction", 0.20)
            ),
            adaptive_expand_support_radius_fraction=float(
                profile.get("adaptive_expand_support_radius_fraction", 0.85)
            ),
            adaptive_contract_edge_mass_fraction=float(
                profile.get("adaptive_contract_edge_mass_fraction", 0.05)
            ),
            adaptive_contract_support_radius_fraction=float(
                profile.get("adaptive_contract_support_radius_fraction", 0.55)
            ),
            ambiguity_support_threshold_peak_fraction=float(
                profile.get("ambiguity_support_threshold_peak_fraction", 0.05)
            ),
            ambiguity_edge_mass_fraction=float(
                profile.get("ambiguity_edge_mass_fraction", 0.20)
            ),
            ambiguity_support_radius_fraction=float(
                profile.get("ambiguity_support_radius_fraction", 0.85)
            ),
            ambiguity_min_posterior_ess_fraction=float(
                profile.get("ambiguity_min_posterior_ess_fraction", 0.03)
            ),
            ambiguity_max_peak_probability=float(
                profile.get("ambiguity_max_peak_probability", 0.85)
            ),
            ambiguity_min_gravity_information_ratio=float(
                profile.get("ambiguity_min_gravity_information_ratio", 0.50)
            ),
            ambiguity_min_bathymetry_information_ratio=float(
                profile.get("ambiguity_min_bathymetry_information_ratio", 0.50)
            ),
            ambiguity_min_magnetic_information_ratio=float(
                profile.get("ambiguity_min_magnetic_information_ratio", 0.50)
            ),
        ),
        learned_localizer_spec=(
            None
            if learned_localizer_model_path is None
            else LearnedLocalizerSpec(
                model_export_path=str(learned_localizer_model_path),
                sequence_spec=GravitySequenceMatcherSpec(
                    window_size=int(profile["window_size"]),
                    grid_half_span_m=tuple(profile["grid_half_span_m"]),
                    grid_spacing_m=tuple(profile["grid_spacing_m"]),
                    transition_std_m=tuple(profile["transition_std_m"]),
                    center_prior_std_m=tuple(profile["center_prior_std_m"]),
                    gravity_meas_std_mps2=float(profile["gravity_meas_std_mps2"]),
                    gradient_meas_std_per_s2=float(profile["gradient_meas_std_per_s2"]),
                    bathymetry_meas_std_m=(
                        None if not use_bathymetry else float(profile["bathymetry_meas_std_m"])
                    ),
                    bathymetry_weight=(
                        1.0 if not use_bathymetry else float(profile.get("bathymetry_weight", 1.0))
                    ),
                    bathymetry_gradient_meas_std_m_per_m=(
                        None
                        if not (use_bathymetry and use_acoustic_terrain)
                        else float(profile.get("bathymetry_gradient_meas_std_m_per_m", 0.01))
                    ),
                    bathymetry_gradient_weight=(
                        1.0
                        if not (use_bathymetry and use_acoustic_terrain)
                        else float(profile.get("bathymetry_gradient_weight", 1.0))
                    ),
                    bathymetry_rugosity_meas_std_m=(
                        None
                        if not (use_bathymetry and use_acoustic_terrain)
                        else float(profile.get("bathymetry_rugosity_meas_std_m", 2.0))
                    ),
                    bathymetry_rugosity_weight=(
                        1.0
                        if not (use_bathymetry and use_acoustic_terrain)
                        else float(profile.get("bathymetry_rugosity_weight", 1.0))
                    ),
                    magnetic_meas_std_nt=(
                        None if not use_magnetics else float(profile.get("magnetic_meas_std_nt", 8.0))
                    ),
                    magnetic_weight=(
                        1.0 if not use_magnetics else float(profile.get("magnetic_weight", 1.0))
                    ),
                    magnetic_gradient_meas_std_nt_per_m=(
                        None
                        if not use_magnetics
                        else float(profile.get("magnetic_gradient_meas_std_nt_per_m", 0.02))
                    ),
                    magnetic_gradient_weight=(
                        1.0
                        if not use_magnetics
                        else float(profile.get("magnetic_gradient_weight", 0.75))
                    ),
                    height_std_m=float(profile.get("height_std_m", 2.0)),
                    adaptive_grid_enabled=bool(profile.get("adaptive_grid_enabled", True)),
                    expanded_grid_half_span_m=profile.get("expanded_grid_half_span_m", None),
                    expanded_grid_spacing_m=profile.get("expanded_grid_spacing_m", None),
                ),
                reliability_threshold=float(profile.get("learned_reliability_threshold", 0.65)),
            )
        ),
        gravity_meas_std_mps2=float(profile["gravity_meas_std_mps2"]),
        use_gradiometer=bool(profile.get("use_gradiometer", True)),
        gradient_meas_std_per_s2=float(profile["gradient_meas_std_per_s2"]),
        use_bathymetry=bool(use_bathymetry),
        use_magnetics=bool(use_magnetics),
        bathymetry_meas_std_m=(
            None if not use_bathymetry else float(profile["bathymetry_meas_std_m"])
        ),
        magnetic_meas_std_nt=(
            None if not use_magnetics else float(profile.get("magnetic_meas_std_nt", 8.0))
        ),
        magnetic_gradient_meas_std_nt_per_m=(
            None
            if not use_magnetics
            else float(profile.get("magnetic_gradient_meas_std_nt_per_m", 0.02))
        ),
        use_sequence_lag_smoother=bool(use_lag_smoother),
        use_sequence_search_centering=bool(
            profile.get("use_sequence_search_centering", False)
        ),
        sequence_search_center_gain=float(
            profile.get("sequence_search_center_gain", 0.35)
        ),
        sequence_search_center_max_norm_m=float(
            profile.get("sequence_search_center_max_norm_m", 250.0)
        ),
        sequence_search_center_max_step_m=float(
            profile.get("sequence_search_center_max_step_m", 60.0)
        ),
        sequence_search_center_min_peak_probability=float(
            profile.get("sequence_search_center_min_peak_probability", 0.15)
        ),
        sequence_search_center_max_horizontal_std_m=float(
            profile.get("sequence_search_center_max_horizontal_std_m", 120.0)
        ),
        sequence_search_center_min_ess_fraction=float(
            profile.get("sequence_search_center_min_ess_fraction", 0.02)
        ),
        sequence_search_center_max_edge_mass_fraction=float(
            profile.get("sequence_search_center_max_edge_mass_fraction", 0.12)
        ),
        sequence_search_center_max_support_radius_fraction=float(
            profile.get("sequence_search_center_max_support_radius_fraction", 0.75)
        ),
    )
    return SimulationRunnerConfig(
        initial_position_offset_ned_m=initial_position_offset_ned_m,
        velocity_aid=VelocityAidFusionConfig(
            enabled=True,
            schedule=PeriodicUpdateSchedule(every_steps=1),
            measurement_frame="ned",
            measurement_mode=(
                "water_relative" if use_current_correction else "earth_relative"
            ),
            use_current_correction=bool(use_current_correction),
            measurement_std_mps=(0.05, 0.05, 0.05),
        ),
        depth_aid=DepthFusionConfig(
            enabled=True,
            schedule=PeriodicUpdateSchedule(every_steps=1),
            measurement_std_m=0.5,
        ),
        map_match=map_match,
        integrity=IntegrityMonitorConfig(
            enabled=True,
            horizontal_alert_limit_m=100.0,
            vertical_alert_limit_m=20.0,
        ),
        metadata_extra={"entry_point": "scripts/run_maritime_demo.py"},
    )


def _run_case(
    label: str,
    *,
    scenario: ScenarioSpec,
    gravity_map: GravityGridMap,
    bathymetry_map: BathymetryGrid | None,
    magnetic_map: Any | None,
    imu_spec: IMUSpec,
    gravimeter_spec: GravimeterSpec | None,
    photonic_spec: PhotonicGravimeterSpec | None,
    depth_spec: DepthSensorSpec,
    velocity_spec: VelocityAidSpec,
    gradiometer_spec: GravityGradiometerSpec,
    bathymetry_spec: BathymetrySensorSpec | None,
    magnetometer_spec: MagnetometerSensorSpec | None,
    current_profile_spec: CurrentProfileSensorSpec | None,
    current_field: Any | None,
    tide_correction_spec: TideCorrectionSpec | None,
    profile: dict[str, Any],
    seed: int,
    output_dir: Path,
    use_bathymetry: bool,
    use_acoustic_terrain: bool,
    use_magnetics: bool,
    use_current_correction: bool,
    use_tide_correction: bool,
    matcher: str,
    learned_localizer_model_path: Path | None,
    disable_map_match: bool = False,
    use_lag_smoother: bool = False,
    dt_s: float = 2.0,
    initial_position_offset_ned_m: tuple[float, float, float] = (60.0, -30.0, 0.0),
) -> tuple[Any, ScenarioMetricsSummary]:
    cfg = _build_runner_config(
        profile,
        matcher=matcher,
        learned_localizer_model_path=learned_localizer_model_path,
        use_bathymetry=use_bathymetry,
        use_acoustic_terrain=use_acoustic_terrain,
        use_magnetics=use_magnetics,
        use_current_correction=use_current_correction,
        use_lag_smoother=use_lag_smoother,
        initial_position_offset_ned_m=initial_position_offset_ned_m,
    )
    if disable_map_match:
        cfg.map_match.enabled = False

    runner = ScenarioSimulationRunner(cfg)
    result = runner.run_with_specs(
        scenario,
        imu_spec=imu_spec,
        gravimeter_spec=gravimeter_spec,
        photonic_gravimeter_spec=photonic_spec,
        depth_spec=depth_spec,
        velocity_aid_spec=velocity_spec,
        gradiometer_spec=gradiometer_spec if not disable_map_match else None,
        bathymetry_spec=bathymetry_spec if use_bathymetry else None,
        bathymetry_map=bathymetry_map if use_bathymetry else None,
        magnetometer_spec=(
            magnetometer_spec if (use_magnetics and not disable_map_match) else None
        ),
        magnetic_map=magnetic_map if (use_magnetics and not disable_map_match) else None,
        current_profile_spec=current_profile_spec if use_current_correction else None,
        current_field=current_field if use_current_correction else None,
        tide_correction_spec=tide_correction_spec if use_tide_correction else None,
        map_model=None if disable_map_match else gravity_map,
        metadata=SimulationMetadata(
            scenario_name=scenario.name,
            run_id=label,
        ),
        dt_s=dt_s,
        seed=seed,
    )
    metrics = scenario_metrics_from_result(result)
    archive_path = output_dir / f"{scenario.name}_{label}_seed{seed}.npz"
    summary_path = output_dir / f"{scenario.name}_{label}_seed{seed}_summary.json"
    metrics_path = output_dir / f"{scenario.name}_{label}_seed{seed}_metrics.json"
    result.save_npz(archive_path)
    result.save_summary_json(summary_path)
    metrics_path.write_text(json.dumps(metrics.to_mapping(), indent=2) + "\n", encoding="utf-8")
    return result, metrics


def _sequence_ambiguity_summary_from_result(result: Any) -> dict[str, Any] | None:
    updates = list(result.estimators.sequence_updates)
    if len(updates) == 0:
        return None

    failure_counts: dict[str, int] = {}
    grid_mode_counts: dict[str, int] = {}
    edge_mass: list[float] = []
    ess_fraction: list[float] = []
    gravity_info: list[float] = []
    bathy_info: list[float] = []
    magnetic_info: list[float] = []
    support_radius_fraction: list[float] = []
    publishability: list[float] = []
    covariance_scale: list[float] = []
    localizer_names: dict[str, int] = {}

    for update in updates:
        diag = update.ambiguity_diagnostics
        failure = str(diag.dominant_failure_mode)
        failure_counts[failure] = failure_counts.get(failure, 0) + 1
        grid_mode = str(diag.grid_mode)
        grid_mode_counts[grid_mode] = grid_mode_counts.get(grid_mode, 0) + 1
        edge_mass.append(float(diag.edge_mass_fraction))
        ess_fraction.append(float(diag.posterior_candidate_ess_fraction))
        gravity_info.append(float(diag.gravity_information_ratio))
        if diag.bathymetry_information_ratio is not None:
            bathy_info.append(float(diag.bathymetry_information_ratio))
        if diag.magnetic_information_ratio is not None:
            magnetic_info.append(float(diag.magnetic_information_ratio))
        if getattr(update, "publishability_probability", None) is not None:
            publishability.append(float(update.publishability_probability))
        if getattr(update, "learned_covariance_scale", None) is not None:
            covariance_scale.append(float(update.learned_covariance_scale))
        localizer_name = str(getattr(update, "localizer_name", "sequence"))
        localizer_names[localizer_name] = localizer_names.get(localizer_name, 0) + 1
        support_radius_fraction.append(
            max(
                float(diag.support_radius_n_m) / max(float(diag.grid_half_span_m[0]), 1.0e-9),
                float(diag.support_radius_e_m) / max(float(diag.grid_half_span_m[1]), 1.0e-9),
            )
        )

    total = float(len(updates))
    return {
        "num_updates": len(updates),
        "informative_fraction": failure_counts.get("informative", 0) / total,
        "edge_clipped_fraction": failure_counts.get("edge_clipped", 0) / total,
        "prior_dominated_fraction": failure_counts.get("prior_dominated", 0) / total,
        "flat_signature_fraction": failure_counts.get("flat_signature", 0) / total,
        "bathymetry_noninformative_fraction": (
            failure_counts.get("bathymetry_noninformative", 0) / total
        ),
        "expanded_grid_fraction": grid_mode_counts.get("expanded", 0) / total,
        "median_edge_mass_fraction": float(np.median(edge_mass)),
        "p90_edge_mass_fraction": float(np.quantile(edge_mass, 0.90)),
        "median_posterior_ess_fraction": float(np.median(ess_fraction)),
        "median_gravity_information_ratio": float(np.median(gravity_info)),
        "median_bathymetry_information_ratio": (
            None if len(bathy_info) == 0 else float(np.median(bathy_info))
        ),
        "median_magnetic_information_ratio": (
            None if len(magnetic_info) == 0 else float(np.median(magnetic_info))
        ),
        "median_publishability_probability": (
            None if len(publishability) == 0 else float(np.median(publishability))
        ),
        "publishability_positive_fraction": (
            None
            if len(publishability) == 0
            else float(np.mean(np.asarray(publishability, dtype=np.float64) >= 0.65))
        ),
        "median_learned_covariance_scale": (
            None if len(covariance_scale) == 0 else float(np.median(covariance_scale))
        ),
        "median_support_radius_fraction": float(np.median(support_radius_fraction)),
        "failure_mode_counts": failure_counts,
        "grid_mode_counts": grid_mode_counts,
        "localizer_name_counts": localizer_names,
    }


def _select_reported_output(
    metrics: ScenarioMetricsSummary,
    *,
    ambiguity: dict[str, Any] | None,
) -> tuple[str, str]:
    lag = metrics.lag_smoothed_position_error
    seq = metrics.sequence_position_error
    ins = metrics.ins_position_error
    lag_integ = metrics.lag_smoothed_integrity

    if ambiguity is None or seq is None or ins is None:
        return "live_ins", "no_sequence_diagnostics"

    edge_clipped = float(ambiguity["edge_clipped_fraction"])
    median_edge_mass = float(ambiguity["median_edge_mass_fraction"])
    median_support_radius = float(ambiguity["median_support_radius_fraction"])
    median_ess = float(ambiguity["median_posterior_ess_fraction"])
    median_gravity_info = float(ambiguity["median_gravity_information_ratio"])
    median_bathy_info = ambiguity["median_bathymetry_information_ratio"]
    median_magnetic_info = ambiguity.get("median_magnetic_information_ratio")
    median_publishability = ambiguity.get("median_publishability_probability")
    if median_bathy_info is not None:
        median_bathy_info = float(median_bathy_info)
    if median_magnetic_info is not None:
        median_magnetic_info = float(median_magnetic_info)
    if median_publishability is not None:
        median_publishability = float(median_publishability)

    lag_allowed = (
        lag is not None
        and lag_integ is not None
        and float(lag_integ.fraction_hazardously_misleading_horizontal) == 0.0
        and edge_clipped <= 0.15
        and median_edge_mass <= 0.12
        and median_support_radius <= 0.75
        and median_ess >= 0.02
        and (median_publishability is None or median_publishability >= 0.65)
        and (median_bathy_info is None or median_bathy_info >= 0.10)
        and (median_magnetic_info is None or median_magnetic_info >= 0.10)
    )
    if lag_allowed:
        return "lag_smoothed", "lag_confident"

    sequence_allowed = (
        edge_clipped <= 0.25
        and median_edge_mass <= 0.20
        and median_support_radius <= 0.85
        and median_ess >= 0.02
        and (median_publishability is None or median_publishability >= 0.55)
        and (
            median_gravity_info >= 0.10
            or (median_bathy_info is not None and median_bathy_info >= 0.10)
            or (median_magnetic_info is not None and median_magnetic_info >= 0.10)
        )
    )
    if sequence_allowed:
        return "sequence", "lag_rejected_or_unavailable"

    if edge_clipped > 0.35 or median_support_radius > 0.90:
        return "live_ins", "edge_coverage_limited"
    if (
        median_gravity_info < 0.10
        and (median_bathy_info is None or median_bathy_info < 0.10)
        and (median_magnetic_info is None or median_magnetic_info < 0.10)
    ):
        return "live_ins", "flat_signature"
    if median_ess < 0.02:
        return "live_ins", "prior_dominated"
    return "live_ins", "sequence_ambiguous"


def _time_key(time_s: float) -> int:
    return int(round(float(time_s) * 1000.0))


def _support_radius_fraction_from_diag(diag: Any) -> float:
    half_span = np.asarray(
        getattr(diag, "grid_half_span_m", np.array([1.0, 1.0], dtype=np.float64)),
        dtype=np.float64,
    ).reshape(-1)
    if half_span.shape[0] < 2:
        half_span = np.array([1.0, 1.0], dtype=np.float64)
    north_frac = float(getattr(diag, "support_radius_n_m", 0.0)) / max(
        float(half_span[0]),
        1.0e-9,
    )
    east_frac = float(getattr(diag, "support_radius_e_m", 0.0)) / max(
        float(half_span[1]),
        1.0e-9,
    )
    return max(north_frac, east_frac)


def _runtime_confidence_level(
    *,
    diag: Any | None,
    alert_ok: bool,
    protection_level_m: float | None,
    horizontal_alert_limit_m: float,
) -> str:
    if diag is None or not alert_ok:
        return "none"

    local_info = max(
        float(getattr(diag, "gravity_information_ratio", 0.0)),
        float(getattr(diag, "bathymetry_information_ratio", 0.0) or 0.0),
        float(getattr(diag, "magnetic_information_ratio", 0.0) or 0.0),
    )
    informative = (
        not bool(getattr(diag, "grid_saturated_any", False))
        and str(getattr(diag, "dominant_failure_mode", "")) == "informative"
    )
    if not informative:
        return "none"

    edge_mass = float(getattr(diag, "edge_mass_fraction", 1.0))
    ess_fraction = float(getattr(diag, "posterior_candidate_ess_fraction", 0.0))
    support_radius_fraction = _support_radius_fraction_from_diag(diag)
    protection_ratio = (
        np.inf
        if protection_level_m is None
        else float(protection_level_m) / max(float(horizontal_alert_limit_m), 1.0e-9)
    )

    if (
        local_info >= 0.20
        and edge_mass <= 0.08
        and ess_fraction >= 0.05
        and support_radius_fraction <= 0.65
        and protection_ratio <= 0.60
    ):
        return "enter"
    if (
        local_info >= 0.12
        and edge_mass <= 0.12
        and ess_fraction >= 0.03
        and support_radius_fraction <= 0.80
        and protection_ratio <= 0.80
    ):
        return "stay"
    return "none"


def _hybrid_lag_ins_from_runtime_signals(
    *,
    ins_times_s: np.ndarray,
    ins_error_ned_m: np.ndarray,
    ins_hmi_horizontal: np.ndarray,
    lag_times_s: np.ndarray,
    lag_error_ned_m: np.ndarray,
    lag_hmi_horizontal: np.ndarray,
    lag_alert_ok: np.ndarray,
    sequence_times_s: np.ndarray,
    sequence_diagnostics: list[Any],
) -> dict[str, Any] | None:
    if (
        ins_times_s.size == 0
        or ins_error_ned_m.shape[0] == 0
        or lag_times_s.size == 0
        or lag_error_ned_m.shape[0] == 0
        or len(sequence_diagnostics) == 0
    ):
        return None

    lag_by_time = {
        _time_key(t): (
            np.asarray(lag_error_ned_m[idx], dtype=np.float64),
            bool(lag_hmi_horizontal[idx]),
            bool(lag_alert_ok[idx]),
        )
        for idx, t in enumerate(np.asarray(lag_times_s, dtype=np.float64))
    }
    diag_by_time = {
        _time_key(t): diag
        for t, diag in zip(np.asarray(sequence_times_s, dtype=np.float64), sequence_diagnostics)
    }

    chosen_error: list[np.ndarray] = []
    chosen_hmi: list[bool] = []
    lag_selected = 0
    for idx, t in enumerate(np.asarray(ins_times_s, dtype=np.float64)):
        lag_item = lag_by_time.get(_time_key(t))
        diag = diag_by_time.get(_time_key(t))
        use_lag = False
        if lag_item is not None and diag is not None:
            lag_err, lag_hmi, lag_ok = lag_item
            local_info = max(
                float(getattr(diag, "gravity_information_ratio", 0.0)),
                float(getattr(diag, "bathymetry_information_ratio", 0.0) or 0.0),
                float(getattr(diag, "magnetic_information_ratio", 0.0) or 0.0),
            )
            use_lag = (
                lag_ok
                and not bool(getattr(diag, "grid_saturated_any", False))
                and str(getattr(diag, "dominant_failure_mode", "")) == "informative"
                and local_info >= 0.10
            )
        if use_lag:
            chosen_error.append(np.asarray(lag_err, dtype=np.float64))
            chosen_hmi.append(bool(lag_hmi))
            lag_selected += 1
        else:
            chosen_error.append(np.asarray(ins_error_ned_m[idx], dtype=np.float64))
            chosen_hmi.append(bool(ins_hmi_horizontal[idx]))

    if len(chosen_error) == 0:
        return None

    err = np.asarray(chosen_error, dtype=np.float64)
    return {
        "position_error": PositionErrorMetrics.from_error_series(err),
        "hmi_horizontal": float(np.mean(np.asarray(chosen_hmi, dtype=bool))),
        "lag_selected_fraction": float(lag_selected / max(len(chosen_hmi), 1)),
        "lag_selected_count": int(lag_selected),
    }


def _hybrid_earth_signature_from_runtime_signals(
    *,
    ins_times_s: np.ndarray,
    ins_error_ned_m: np.ndarray,
    ins_hmi_horizontal: np.ndarray,
    lag_times_s: np.ndarray,
    lag_error_ned_m: np.ndarray,
    lag_hmi_horizontal: np.ndarray,
    lag_alert_ok: np.ndarray,
    sequence_times_s: np.ndarray,
    sequence_error_ned_m: np.ndarray,
    sequence_hmi_horizontal: np.ndarray,
    sequence_alert_ok: np.ndarray,
    lag_protection_level_m: np.ndarray | None = None,
    sequence_protection_level_m: np.ndarray | None = None,
    sequence_diagnostics: list[Any],
    horizontal_alert_limit_m: float = 100.0,
    enter_consecutive_steps: int = 2,
) -> dict[str, Any] | None:
    if ins_times_s.size == 0 or ins_error_ned_m.shape[0] == 0:
        return None

    if lag_protection_level_m is None:
        lag_protection_level_m = np.full(lag_times_s.shape, np.nan, dtype=np.float64)
    if sequence_protection_level_m is None:
        sequence_protection_level_m = np.full(
            sequence_times_s.shape,
            np.nan,
            dtype=np.float64,
        )

    lag_by_time = {
        _time_key(t): (
            np.asarray(lag_error_ned_m[idx], dtype=np.float64),
            bool(lag_hmi_horizontal[idx]),
            bool(lag_alert_ok[idx]),
            float(lag_protection_level_m[idx]),
        )
        for idx, t in enumerate(np.asarray(lag_times_s, dtype=np.float64))
    }
    sequence_by_time = {
        _time_key(t): (
            np.asarray(sequence_error_ned_m[idx], dtype=np.float64),
            bool(sequence_hmi_horizontal[idx]),
            bool(sequence_alert_ok[idx]),
            float(sequence_protection_level_m[idx]),
            diag,
        )
        for idx, (t, diag) in enumerate(
            zip(np.asarray(sequence_times_s, dtype=np.float64), sequence_diagnostics)
        )
    }

    chosen_error: list[np.ndarray] = []
    chosen_hmi: list[bool] = []
    lag_selected = 0
    sequence_selected = 0
    ins_selected = 0
    lag_enter_streak = 0
    sequence_enter_streak = 0
    active_mode = "ins"

    for idx, t in enumerate(np.asarray(ins_times_s, dtype=np.float64)):
        time_key = _time_key(t)
        lag_item = lag_by_time.get(time_key)
        sequence_item = sequence_by_time.get(time_key)

        diag = None if sequence_item is None else sequence_item[4]
        sequence_level = (
            "none"
            if sequence_item is None
            else _runtime_confidence_level(
                diag=diag,
                alert_ok=bool(sequence_item[2]),
                protection_level_m=float(sequence_item[3]),
                horizontal_alert_limit_m=horizontal_alert_limit_m,
            )
        )
        lag_level = (
            "none"
            if lag_item is None
            else _runtime_confidence_level(
                diag=diag,
                alert_ok=bool(lag_item[2]),
                protection_level_m=float(lag_item[3]),
                horizontal_alert_limit_m=horizontal_alert_limit_m,
            )
        )

        lag_enter_streak = lag_enter_streak + 1 if lag_level == "enter" else 0
        sequence_enter_streak = (
            sequence_enter_streak + 1 if sequence_level == "enter" else 0
        )

        if active_mode == "lag":
            if lag_level in {"enter", "stay"}:
                pass
            elif sequence_enter_streak >= enter_consecutive_steps:
                active_mode = "sequence"
            else:
                active_mode = "ins"
        elif active_mode == "sequence":
            if lag_enter_streak >= enter_consecutive_steps:
                active_mode = "lag"
            elif sequence_level in {"enter", "stay"}:
                pass
            else:
                active_mode = "ins"
        else:
            if lag_enter_streak >= enter_consecutive_steps:
                active_mode = "lag"
            elif sequence_enter_streak >= enter_consecutive_steps:
                active_mode = "sequence"

        if active_mode == "lag" and lag_item is not None:
            lag_err, lag_hmi, _, _ = lag_item
            chosen_error.append(np.asarray(lag_err, dtype=np.float64))
            chosen_hmi.append(bool(lag_hmi))
            lag_selected += 1
        elif active_mode == "sequence" and sequence_item is not None:
            seq_err, seq_hmi, _, _, _ = sequence_item
            chosen_error.append(np.asarray(seq_err, dtype=np.float64))
            chosen_hmi.append(bool(seq_hmi))
            sequence_selected += 1
        else:
            active_mode = "ins"
            chosen_error.append(np.asarray(ins_error_ned_m[idx], dtype=np.float64))
            chosen_hmi.append(bool(ins_hmi_horizontal[idx]))
            ins_selected += 1

    if len(chosen_error) == 0:
        return None

    count = max(len(chosen_hmi), 1)
    err = np.asarray(chosen_error, dtype=np.float64)
    return {
        "position_error": PositionErrorMetrics.from_error_series(err),
        "hmi_horizontal": float(np.mean(np.asarray(chosen_hmi, dtype=bool))),
        "lag_selected_fraction": float(lag_selected / count),
        "lag_selected_count": int(lag_selected),
        "sequence_selected_fraction": float(sequence_selected / count),
        "sequence_selected_count": int(sequence_selected),
        "ins_selected_fraction": float(ins_selected / count),
        "ins_selected_count": int(ins_selected),
    }


def _runtime_hybrid_lag_ins_summary(result: Any) -> dict[str, Any] | None:
    if (
        len(result.estimators.ins_states) == 0
        or len(result.estimators.sequence_updates) == 0
        or len(result.estimators.integrity_snapshots) == 0
    ):
        return None

    ins_times = np.asarray(ins_state_times(result.estimators.ins_states), dtype=np.float64)
    ins_err = np.asarray(
        ins_position_error_history_from_truth(result.truth, result.estimators.ins_states),
        dtype=np.float64,
    )
    lag_times = np.asarray(
        ins_state_times(result.estimators.lag_smoothed_states),
        dtype=np.float64,
    )
    lag_err = np.asarray(
        lag_smoothed_position_error_history_from_truth(
            result.truth,
            result.estimators.lag_smoothed_states,
        ),
        dtype=np.float64,
    )

    ins_hmi = np.asarray(
        [
            bool(getattr(snap, "hazardously_misleading_horizontal", False))
            for snap in result.estimators.integrity_snapshots[: ins_times.size]
        ],
        dtype=bool,
    )
    if len(result.estimators.lag_smoothed_states) > 0:
        lag_times = np.asarray(
            ins_state_times(result.estimators.lag_smoothed_states),
            dtype=np.float64,
        )
        lag_err = np.asarray(
            lag_smoothed_position_error_history_from_truth(
                result.truth,
                result.estimators.lag_smoothed_states,
            ),
            dtype=np.float64,
        )
        lag_snaps = result.estimators.lag_smoothed_integrity_snapshots[: lag_times.size]
        lag_hmi = np.asarray(
            [
                bool(getattr(snap, "hazardously_misleading_horizontal", False))
                for snap in lag_snaps
            ],
            dtype=bool,
        )
        lag_alert_ok = np.asarray(
            [
                bool(
                    getattr(
                        getattr(snap, "protection_levels", None),
                        "horizontal_within_alert_limit",
                        False,
                    )
                )
                for snap in lag_snaps
            ],
            dtype=bool,
        )
        lag_hpl_m = np.asarray(
            [
                float(getattr(getattr(snap, "protection_levels", None), "horizontal_m", np.nan))
                for snap in lag_snaps
            ],
            dtype=np.float64,
        )
    else:
        lag_times = np.empty(0, dtype=np.float64)
        lag_err = np.empty((0, 3), dtype=np.float64)
        lag_hmi = np.empty(0, dtype=bool)
        lag_alert_ok = np.empty(0, dtype=bool)
        lag_hpl_m = np.empty(0, dtype=np.float64)

    seq_times = np.asarray(
        [float(getattr(upd, "time_s")) for upd in result.estimators.sequence_updates],
        dtype=np.float64,
    )
    seq_err = np.asarray(
        sequence_position_error_history_from_truth(
            result.truth,
            result.estimators.sequence_updates,
        ),
        dtype=np.float64,
    )
    seq_diags = [upd.ambiguity_diagnostics for upd in result.estimators.sequence_updates]
    seq_hmi = np.empty(seq_times.size, dtype=bool)
    seq_alert_ok = np.empty(seq_times.size, dtype=bool)
    seq_hpl_m = np.empty(seq_times.size, dtype=np.float64)
    for idx, (upd, err) in enumerate(zip(result.estimators.sequence_updates, seq_err)):
        pls = protection_levels_from_covariance_ned(
            upd.estimate.covariance_ned_m2,
            horizontal_alert_limit_m=100.0,
        )
        horizontal_err = float(np.linalg.norm(np.asarray(err, dtype=np.float64)[:2]))
        seq_hmi[idx] = bool(
            pls.horizontal_m <= 100.0 and horizontal_err > 100.0
        )
        seq_alert_ok[idx] = bool(pls.horizontal_within_alert_limit)
        seq_hpl_m[idx] = float(pls.horizontal_m)

    return _hybrid_earth_signature_from_runtime_signals(
        ins_times_s=ins_times,
        ins_error_ned_m=ins_err,
        ins_hmi_horizontal=ins_hmi,
        lag_times_s=lag_times,
        lag_error_ned_m=lag_err,
        lag_hmi_horizontal=lag_hmi,
        lag_alert_ok=lag_alert_ok,
        lag_protection_level_m=lag_hpl_m,
        sequence_times_s=seq_times,
        sequence_error_ned_m=seq_err,
        sequence_hmi_horizontal=seq_hmi,
        sequence_alert_ok=seq_alert_ok,
        sequence_protection_level_m=seq_hpl_m,
        sequence_diagnostics=seq_diags,
        horizontal_alert_limit_m=100.0,
        enter_consecutive_steps=2,
    )


def _metrics_row(
    label: str,
    metrics: ScenarioMetricsSummary,
    *,
    result: Any | None = None,
    ambiguity: dict[str, Any] | None,
    search_center: dict[str, Any] | None = None,
) -> dict[str, float | str | None]:
    ins = metrics.ins_position_error
    seq = metrics.sequence_position_error
    lag = metrics.lag_smoothed_position_error
    integ = metrics.integrity
    lag_integ = metrics.lag_smoothed_integrity
    reported_mode, reported_reason = _select_reported_output(
        metrics,
        ambiguity=ambiguity,
    )
    hybrid = None if result is None else _runtime_hybrid_lag_ins_summary(result)
    if hybrid is not None and (
        float(hybrid["lag_selected_fraction"]) > 0.0
        or float(hybrid["sequence_selected_fraction"]) > 0.0
    ):
        hybrid_metrics = hybrid["position_error"]
        earth_rmse = float(hybrid_metrics.horizontal_rmse_m)
        earth_cep95 = float(hybrid_metrics.cep95_m)
        earth_hmi = float(hybrid["hmi_horizontal"])
        reported_mode = "hybrid_runtime_selector"
        reported_reason = "per_step_lag_then_sequence_when_confident"
    elif reported_mode == "lag_smoothed" and lag is not None:
        earth_rmse = float(lag.horizontal_rmse_m)
        earth_cep95 = float(lag.cep95_m)
        earth_hmi = (
            None
            if lag_integ is None
            else float(lag_integ.fraction_hazardously_misleading_horizontal)
        )
    elif reported_mode == "sequence" and seq is not None:
        earth_rmse = float(seq.horizontal_rmse_m)
        earth_cep95 = float(seq.cep95_m)
        earth_hmi = (
            None
            if integ is None
            else float(integ.fraction_hazardously_misleading_horizontal)
        )
    elif ins is not None:
        earth_rmse = float(ins.horizontal_rmse_m)
        earth_cep95 = float(ins.cep95_m)
        earth_hmi = (
            None
            if integ is None
            else float(integ.fraction_hazardously_misleading_horizontal)
        )
    else:
        earth_rmse = None
        earth_cep95 = None
        earth_hmi = None
    return {
        "label": label,
        "ins_horizontal_rmse_m": None if ins is None else float(ins.horizontal_rmse_m),
        "ins_cep95_m": None if ins is None else float(ins.cep95_m),
        "sequence_horizontal_rmse_m": None if seq is None else float(seq.horizontal_rmse_m),
        "sequence_cep95_m": None if seq is None else float(seq.cep95_m),
        "lag_horizontal_rmse_m": None if lag is None else float(lag.horizontal_rmse_m),
        "lag_cep95_m": None if lag is None else float(lag.cep95_m),
        "hmi_horizontal": None if integ is None else float(integ.fraction_hazardously_misleading_horizontal),
        "lag_hmi_horizontal": None if lag_integ is None else float(lag_integ.fraction_hazardously_misleading_horizontal),
        "earth_signature_horizontal_rmse_m": earth_rmse,
        "earth_signature_cep95_m": earth_cep95,
        "earth_signature_hmi_horizontal": earth_hmi,
        "earth_signature_mode": reported_mode,
        "reported_output_mode": reported_mode,
        "reported_output_reason": reported_reason,
        "ambiguity_informative_fraction": None
        if ambiguity is None
        else float(ambiguity["informative_fraction"]),
        "ambiguity_edge_clipped_fraction": None
        if ambiguity is None
        else float(ambiguity["edge_clipped_fraction"]),
        "ambiguity_prior_dominated_fraction": None
        if ambiguity is None
        else float(ambiguity["prior_dominated_fraction"]),
        "ambiguity_flat_signature_fraction": None
        if ambiguity is None
        else float(ambiguity["flat_signature_fraction"]),
        "ambiguity_expanded_grid_fraction": None
        if ambiguity is None
        else float(ambiguity["expanded_grid_fraction"]),
        "ambiguity_median_edge_mass_fraction": None
        if ambiguity is None
        else float(ambiguity["median_edge_mass_fraction"]),
        "ambiguity_median_support_radius_fraction": None
        if ambiguity is None
        else float(ambiguity["median_support_radius_fraction"]),
        "ambiguity_median_posterior_ess_fraction": None
        if ambiguity is None
        else float(ambiguity["median_posterior_ess_fraction"]),
        "ambiguity_median_gravity_information_ratio": None
        if ambiguity is None
        else float(ambiguity["median_gravity_information_ratio"]),
        "ambiguity_median_bathymetry_information_ratio": None
        if ambiguity is None or ambiguity["median_bathymetry_information_ratio"] is None
        else float(ambiguity["median_bathymetry_information_ratio"]),
        "ambiguity_median_magnetic_information_ratio": None
        if ambiguity is None or ambiguity["median_magnetic_information_ratio"] is None
        else float(ambiguity["median_magnetic_information_ratio"]),
        "ambiguity_median_publishability_probability": None
        if ambiguity is None or ambiguity["median_publishability_probability"] is None
        else float(ambiguity["median_publishability_probability"]),
        "ambiguity_publishability_positive_fraction": None
        if ambiguity is None or ambiguity["publishability_positive_fraction"] is None
        else float(ambiguity["publishability_positive_fraction"]),
        "ambiguity_failure_mode_counts": None
        if ambiguity is None
        else dict(ambiguity["failure_mode_counts"]),
        "ambiguity_grid_mode_counts": None
        if ambiguity is None
        else dict(ambiguity["grid_mode_counts"]),
        "search_center_accepted_count": None
        if search_center is None
        else float(search_center["accepted_count"]),
        "search_center_accepted_fraction": None
        if search_center is None
        else float(search_center["accepted_fraction"]),
        "search_center_final_offset_norm_m": None
        if search_center is None
        else float(search_center["final_offset_horizontal_norm_m"]),
        "search_center_reason_counts": None
        if search_center is None
        else dict(search_center["reason_counts"]),
        "search_center_proposal_mode_counts": None
        if search_center is None
        else dict(search_center["proposal_mode_counts"]),
        "hybrid_lag_selected_fraction": None
        if hybrid is None
        else float(hybrid["lag_selected_fraction"]),
        "hybrid_lag_selected_count": None
        if hybrid is None
        else float(hybrid["lag_selected_count"]),
        "hybrid_sequence_selected_fraction": None
        if hybrid is None
        else float(hybrid["sequence_selected_fraction"]),
        "hybrid_sequence_selected_count": None
        if hybrid is None
        else float(hybrid["sequence_selected_count"]),
        "hybrid_ins_selected_fraction": None
        if hybrid is None
        else float(hybrid["ins_selected_fraction"]),
        "hybrid_ins_selected_count": None
        if hybrid is None
        else float(hybrid["ins_selected_count"]),
    }


def _photonic_summary_from_result(result: Any) -> dict[str, Any] | None:
    samples = result.sensors.custom_streams.get("photonic_gravimeter", [])
    if len(samples) == 0:
        return None
    return asdict(summarize_photonic_measurements(samples))


def _search_center_summary_from_result(result: Any) -> dict[str, Any] | None:
    rows = result.estimators.custom_streams.get("sequence_search_center_offset", [])
    if len(rows) == 0:
        return None
    accepted = [row for row in rows if bool(row.get("accepted", False))]
    reason_counts: dict[str, int] = {}
    proposal_mode_counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        proposal_mode = str(row.get("proposal_mode", "unknown"))
        proposal_mode_counts[proposal_mode] = (
            proposal_mode_counts.get(proposal_mode, 0) + 1
        )
    final_offset = (
        np.zeros(3, dtype=np.float64)
        if len(accepted) == 0
        else np.asarray(accepted[-1]["updated_offset_ned_m"], dtype=np.float64)
    )
    final_offset[2] = 0.0
    return {
        "accepted_count": len(accepted),
        "accepted_fraction": len(accepted) / float(len(rows)),
        "final_offset_horizontal_norm_m": float(np.linalg.norm(final_offset[:2])),
        "reason_counts": reason_counts,
        "proposal_mode_counts": proposal_mode_counts,
    }


def _make_summary_plot(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
) -> None:
    labels = [row["label"] for row in rows]
    earth_rmse = [row["earth_signature_horizontal_rmse_m"] for row in rows]
    ins_rmse = [row["ins_horizontal_rmse_m"] for row in rows]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.18, ins_rmse, width=0.36, label="live INS")
    ax.bar(x + 0.18, earth_rmse, width=0.36, label="reported Earth-signature output")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("horizontal RMSE [m]")
    ax.set_title("Norwegian maritime demo comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_report(
    path: Path,
    *,
    scenario: ScenarioSpec,
    demo_pack_path: Path | None,
    demo_pack_region: str | None,
    gravity_map_path: Path,
    gravity_manifest_path: Path | None,
    bathymetry_grid_path: Path | None,
    bathymetry_manifest_path: Path | None,
    magnetic_grid_path: Path | None,
    magnetic_manifest_path: Path | None,
    current_field_grid_path: Path | None,
    current_manifest_path: Path | None,
    tide_config_path: Path | None,
    profile_path: Path,
    profile: dict[str, Any],
    rows: list[dict[str, Any]],
    photonic_rows: list[dict[str, Any]] | None = None,
    seeds: list[int],
    lag_validated: bool,
    dt_s: float,
    initial_position_offset_ned_m: tuple[float, float, float],
) -> None:
    if photonic_rows is None:
        photonic_rows = []
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row["label"]), []).append(row)
    photonic_by_label: dict[str, list[dict[str, Any]]] = {}
    for row in photonic_rows:
        photonic_by_label.setdefault(str(row["label"]), []).append(row)

    def median(label: str, key: str) -> float:
        if label not in by_label:
            return float("nan")
        vals = [
            float(r[key])
            for r in by_label[label]
            if key in r and r[key] is not None
        ]
        if len(vals) == 0:
            return float("nan")
        return float(np.median(vals))

    def photonic_median(label: str, key: str) -> float | None:
        entries = photonic_by_label.get(label, [])
        vals = [float(r[key]) for r in entries if r.get(key) is not None]
        if len(vals) == 0:
            return None
        return float(np.median(vals))

    def photonic_top_rejection_reason(label: str) -> str | None:
        counts: dict[str, int] = {}
        for row in photonic_by_label.get(label, []):
            for reason, count in row.get("rejection_reason_counts", {}).items():
                counts[str(reason)] = counts.get(str(reason), 0) + int(count)
        if not counts:
            return None
        return max(sorted(counts), key=lambda key: counts[key])

    def search_center_median(label: str, key: str) -> float | None:
        vals = [
            float(r[key])
            for r in by_label.get(label, [])
            if r.get(key) is not None
        ]
        if len(vals) == 0:
            return None
        return float(np.median(vals))

    ordered_labels = [
        "live_ins",
        "photonic_gravity_baseline",
        "photonic_gravity_tide",
        "photonic_gravity_tide_acoustic",
        "photonic_gravity_tide_acoustic_magnetic",
        "photonic_gravity_tide_acoustic_magnetic_current",
    ]
    lines = [
        "# Norway-First Multi-Modal Earth-Signature Demo Report",
        "",
        "## Scope",
        "",
        f"- scenario: `{scenario.name}`",
        f"- demo pack manifest: `{demo_pack_path}`" if demo_pack_path is not None else "- demo pack manifest: `None (direct asset selection)`",
        f"- demo pack region: `{demo_pack_region}`" if demo_pack_region is not None else "- demo pack region: `n/a`",
        f"- gravity map runtime source: `{gravity_map_path}`",
        f"- gravity manifest: `{gravity_manifest_path}`" if gravity_manifest_path is not None else "- gravity manifest: `n/a`",
        f"- bathymetry grid source: `{bathymetry_grid_path}`" if bathymetry_grid_path is not None else "- bathymetry grid source: `n/a`",
        f"- bathymetry manifest: `{bathymetry_manifest_path}`" if bathymetry_manifest_path is not None else "- bathymetry manifest: `n/a`",
        f"- magnetic grid source: `{magnetic_grid_path}`" if magnetic_grid_path is not None else "- magnetic grid source: `n/a`",
        f"- magnetic manifest: `{magnetic_manifest_path}`" if magnetic_manifest_path is not None else "- magnetic manifest: `n/a`",
        f"- current-field grid source: `{current_field_grid_path}`" if current_field_grid_path is not None else "- current-field grid source: `n/a`",
        f"- current-field manifest: `{current_manifest_path}`" if current_manifest_path is not None else "- current-field manifest: `n/a`",
        f"- tide config: `{tide_config_path}`" if tide_config_path is not None else "- tide config: `n/a`",
        "- estimator: observe-only sequence matcher with gravity mandatory and other channels additive",
        "- live INS path unchanged",
        f"- sample period: `{dt_s:.1f} s`",
        f"- initial position offset NED [m]: `{list(initial_position_offset_ned_m)}`",
        f"- seeds: {', '.join(str(s) for s in seeds)}",
        "",
        "## Frozen Sequence Profile",
        "",
        f"- profile: `{profile_path}`",
        f"- window_size: `{profile['window_size']}`",
        f"- grid_half_span_m: `{profile['grid_half_span_m']}`",
        f"- grid_spacing_m: `{profile['grid_spacing_m']}`",
        f"- transition_std_m: `{profile['transition_std_m']}`",
        f"- center_prior_std_m: `{profile['center_prior_std_m']}`",
        f"- bathymetry_meas_std_m: `{profile['bathymetry_meas_std_m']}`",
        f"- bathymetry_weight: `{profile['bathymetry_weight']}`",
        f"- bathymetry_gradient_meas_std_m_per_m: `{profile.get('bathymetry_gradient_meas_std_m_per_m', 'default')}`",
        f"- bathymetry_rugosity_meas_std_m: `{profile.get('bathymetry_rugosity_meas_std_m', 'default')}`",
        f"- magnetic_meas_std_nt: `{profile.get('magnetic_meas_std_nt', 'default')}`",
        f"- magnetic_gradient_meas_std_nt_per_m: `{profile.get('magnetic_gradient_meas_std_nt_per_m', 'default')}`",
        f"- map_match_every_steps: `{profile['map_match_every_steps']}`",
        "",
        "## Median Results Across Seeds",
        "",
        "| Mode | Reported output | Reason | Live INS RMSE [m] | Reported RMSE [m] | Live INS CEP95 [m] | Reported CEP95 [m] | HMI horiz |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in ordered_labels:
        if label not in by_label:
            continue
        lines.append(
            f"| `{label}` | "
            f"`{by_label[label][0].get('reported_output_mode', by_label[label][0].get('earth_signature_mode', 'live_ins'))}` | "
            f"`{by_label[label][0].get('reported_output_reason', 'n/a')}` | "
            f"{median(label, 'ins_horizontal_rmse_m'):.3f} | "
            f"{median(label, 'earth_signature_horizontal_rmse_m'):.3f} | "
            f"{median(label, 'ins_cep95_m'):.3f} | "
            f"{median(label, 'earth_signature_cep95_m'):.3f} | "
            f"{median(label, 'earth_signature_hmi_horizontal'):.3f} |"
        )
    if any(
        any(row.get("ambiguity_informative_fraction") is not None for row in by_label[label])
        for label in by_label
    ):
        lines.extend(
            [
                "",
                "## Sequence Ambiguity Summary",
                "",
                "| Mode | Informative frac | Edge-clipped frac | Prior-dominated frac | Flat-signature frac | Expanded-grid frac | Median edge mass | Median support radius frac | Median gravity info ratio | Median bathy info ratio | Median magnetic info ratio |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label in ordered_labels:
            if label not in by_label:
                continue
            lines.append(
                f"| `{label}` | "
                f"{median(label, 'ambiguity_informative_fraction'):.3f} | "
                f"{median(label, 'ambiguity_edge_clipped_fraction'):.3f} | "
                f"{median(label, 'ambiguity_prior_dominated_fraction'):.3f} | "
                f"{median(label, 'ambiguity_flat_signature_fraction'):.3f} | "
                f"{median(label, 'ambiguity_expanded_grid_fraction'):.3f} | "
                f"{median(label, 'ambiguity_median_edge_mass_fraction'):.3f} | "
                f"{median(label, 'ambiguity_median_support_radius_fraction'):.3f} | "
                f"{median(label, 'ambiguity_median_gravity_information_ratio'):.3f} | "
                f"{median(label, 'ambiguity_median_bathymetry_information_ratio'):.3f} | "
                f"{median(label, 'ambiguity_median_magnetic_information_ratio'):.3f} |"
            )
    if "photonic_gravity_tide_acoustic_magnetic_current" in by_label:
        lines.extend(
            [
                "",
                "## Product Output Result",
                "",
                f"- final multi-modal reported RMSE: `{median('photonic_gravity_tide_acoustic_magnetic_current', 'earth_signature_horizontal_rmse_m'):.3f} m`",
                f"- final multi-modal reported CEP95: `{median('photonic_gravity_tide_acoustic_magnetic_current', 'earth_signature_cep95_m'):.3f} m`",
                f"- lag-smoothed output validated: `{lag_validated}`",
            ]
        )
    if any(row.get("search_center_accepted_count") is not None for row in rows):
        lines.extend(
            [
                "",
                "## Search-Center Summary",
                "",
                "| Mode | Accepted count | Accepted fraction | Final offset norm [m] |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for label in ordered_labels:
            if label not in by_label:
                continue
            accepted_count = search_center_median(label, "search_center_accepted_count")
            accepted_fraction = search_center_median(
                label,
                "search_center_accepted_fraction",
            )
            final_norm = search_center_median(
                label,
                "search_center_final_offset_norm_m",
            )
            lines.append(
                f"| `{label}` | "
                f"{'n/a' if accepted_count is None else f'{accepted_count:.1f}'} | "
                f"{'n/a' if accepted_fraction is None else f'{accepted_fraction:.3f}'} | "
                f"{'n/a' if final_norm is None else f'{final_norm:.3f}'} |"
            )

    photonic_labels = [label for label in ordered_labels if label in photonic_by_label]
    if photonic_labels:
        lines.extend(
            [
                "",
                "## Photonic Sensor Diagnostics",
                "",
                "| Mode | Valid fraction | Median contrast | P95 contrast | RMS vibration residual phase [rad] | RMS disturbance residual [m/s^2] | Tilt exceedance fraction | Dominant rejection |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for label in photonic_labels:
            valid_fraction = photonic_median(label, "valid_sample_fraction")
            median_contrast = photonic_median(label, "median_fringe_contrast")
            p95_contrast = photonic_median(label, "p95_fringe_contrast")
            rms_phase = photonic_median(label, "rms_vibration_residual_phase_rad")
            rms_residual = photonic_median(label, "rms_disturbance_residual_mps2")
            tilt_fraction = photonic_median(label, "tilt_exceedance_fraction")
            dominant_reason = photonic_top_rejection_reason(label)
            lines.append(
                f"| `{label}` | "
                f"{'n/a' if valid_fraction is None else f'{valid_fraction:.3f}'} | "
                f"{'n/a' if median_contrast is None else f'{median_contrast:.3f}'} | "
                f"{'n/a' if p95_contrast is None else f'{p95_contrast:.3f}'} | "
                f"{'n/a' if rms_phase is None else f'{rms_phase:.3e}'} | "
                f"{'n/a' if rms_residual is None else f'{rms_residual:.3e}'} | "
                f"{'n/a' if tilt_fraction is None else f'{tilt_fraction:.3f}'} | "
                f"`{dominant_reason or 'none'}` |"
            )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- photonic gravity baseline beats live INS on median RMSE: `{median('photonic_gravity_baseline', 'earth_signature_horizontal_rmse_m') < median('live_ins', 'ins_horizontal_rmse_m')}`",
            f"- tide/datum correction improves over the gravity baseline on median RMSE: `{median('photonic_gravity_tide', 'earth_signature_horizontal_rmse_m') < median('photonic_gravity_baseline', 'earth_signature_horizontal_rmse_m')}`",
            f"- acoustic terrain improves over tide-only on median RMSE: `{median('photonic_gravity_tide_acoustic', 'earth_signature_horizontal_rmse_m') < median('photonic_gravity_tide', 'earth_signature_horizontal_rmse_m')}`",
            f"- scalar magnetic improves over tide+acoustic on median RMSE: `{median('photonic_gravity_tide_acoustic_magnetic', 'earth_signature_horizontal_rmse_m') < median('photonic_gravity_tide_acoustic', 'earth_signature_horizontal_rmse_m')}`",
            f"- current-aware prior improves over tide+acoustic+magnetic on median RMSE: `{median('photonic_gravity_tide_acoustic_magnetic_current', 'earth_signature_horizontal_rmse_m') < median('photonic_gravity_tide_acoustic_magnetic', 'earth_signature_horizontal_rmse_m')}`",
            f"- final reported output beats live INS on median RMSE: `{median('photonic_gravity_tide_acoustic_magnetic_current', 'earth_signature_horizontal_rmse_m') < median('live_ins', 'ins_horizontal_rmse_m')}`",
            f"- final reported output beats live INS on median CEP95: `{median('photonic_gravity_tide_acoustic_magnetic_current', 'earth_signature_cep95_m') < median('live_ins', 'ins_cep95_m')}`",
            "- gravity remains mandatory and the additional channels are additive disambiguators or process corrections.",
            "- tide is treated as correction hygiene, not as a standalone localization signature.",
            "- the recommended demo output is selected adaptively: lag when confidence is high, sequence when lag is too ambiguous, otherwise live INS.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Norway-first multi-modal Earth-signature demo."
    )
    parser.add_argument(
        "--demo-pack-manifest",
        default=str(DEFAULT_DEMO_PACK),
        help=(
            "Optional demo-pack manifest. When it exists, it overrides the scenario, "
            "sequence profile, gravity-map, and bathymetry asset selection."
        ),
    )
    parser.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    parser.add_argument("--sequence-profile", default=str(DEFAULT_SEQUENCE_PROFILE))
    parser.add_argument("--imu-config", default=str(DEFAULT_IMU_CONFIG))
    parser.add_argument("--gravimeter-config", default=str(DEFAULT_GRAVIMETER_CONFIG))
    parser.add_argument("--photonic-config", default=str(DEFAULT_PHOTONIC_CONFIG))
    parser.add_argument("--depth-config", default=str(DEFAULT_DEPTH_CONFIG))
    parser.add_argument("--velocity-config", default=str(DEFAULT_VELOCITY_CONFIG))
    parser.add_argument("--gradiometer-config", default=str(DEFAULT_GRADIOMETER_CONFIG))
    parser.add_argument("--bathymetry-config", default=str(DEFAULT_BATHY_SENSOR_CONFIG))
    parser.add_argument("--magnetometer-config", default=str(DEFAULT_MAGNETOMETER_CONFIG))
    parser.add_argument(
        "--current-profile-config",
        default=str(DEFAULT_CURRENT_PROFILE_CONFIG),
    )
    parser.add_argument(
        "--tide-config",
        default=None,
        help=(
            "Optional tide/datum correction config. Defaults to the demo-pack path "
            "when present, else the tracked Norway tide config."
        ),
    )
    parser.add_argument("--gravity-map", default=str(DEFAULT_GRAVITY_MAP))
    parser.add_argument(
        "--magnetic-manifest",
        default=None,
        help="Optional magnetic-manifest override when the demo pack does not supply one.",
    )
    parser.add_argument(
        "--current-manifest",
        default=None,
        help="Optional current-field manifest override when the demo pack does not supply one.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    parser.add_argument("--dt-s", type=float, default=2.0)
    parser.add_argument(
        "--map-matcher",
        choices=("sequence", "learned_sequence"),
        default="sequence",
    )
    parser.add_argument(
        "--learned-localizer-model",
        default=None,
        help="Optional exported learned-localizer model bundle (.npz).",
    )
    parser.add_argument(
        "--initial-position-offset-ned-m",
        type=float,
        nargs=3,
        default=[60.0, -30.0, 0.0],
        metavar=("NORTH_M", "EAST_M", "DOWN_M"),
        help="Deterministic initial INS position offset in local NED metres.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    demo_pack_path = Path(args.demo_pack_manifest).expanduser().resolve()
    if demo_pack_path == DEFAULT_DEMO_PACK:
        _ensure_demo_pack()

    demo_pack_assets: ResolvedRegionalDemoPack | None = None
    gravity_manifest_path: Path | None = None
    bathymetry_grid_path: Path | None = None
    bathymetry_manifest_path: Path | None = None
    magnetic_map = None
    magnetic_grid_path: Path | None = None
    magnetic_manifest_path: Path | None = None
    current_field = None
    current_field_grid_path: Path | None = None
    current_manifest_path: Path | None = None
    tide_config_path: Path | None = None
    if demo_pack_path.exists():
        demo_pack_assets = _load_demo_pack_assets(demo_pack_path)
        scenario_path = demo_pack_assets.scenario_path
        profile_path = demo_pack_assets.sequence_profile_path
        gravity_map_path = demo_pack_assets.gravity_map_path
        gravity_map = demo_pack_assets.gravity_map
        bathymetry_map = demo_pack_assets.bathymetry_grid
        gravity_manifest_path = demo_pack_assets.gravity_manifest_path
        bathymetry_grid_path = demo_pack_assets.bathymetry_grid_path
        bathymetry_manifest_path = demo_pack_assets.bathymetry_manifest_path
        magnetic_map = demo_pack_assets.magnetic_grid
        magnetic_grid_path = demo_pack_assets.magnetic_grid_path
        magnetic_manifest_path = demo_pack_assets.magnetic_manifest_path
        current_field = demo_pack_assets.current_field
        current_field_grid_path = demo_pack_assets.current_field_grid_path
        current_manifest_path = demo_pack_assets.current_manifest_path
        tide_config_path = demo_pack_assets.tide_config_path
    else:
        scenario_path = Path(args.scenario).expanduser().resolve()
        profile_path = Path(args.sequence_profile).expanduser().resolve()
        gravity_map_path = Path(args.gravity_map).expanduser().resolve()
        gravity_map = GravityGridMap.from_npz(gravity_map_path)
        bathymetry_map, _, bathymetry_grid_path, bathymetry_manifest_path = (
            ensure_regional_bathymetry_grid(
                "norwegian_margin_public",
                project_root=PROJECT_ROOT,
            )
        )
    if args.magnetic_manifest:
        (
            magnetic_map,
            _,
            magnetic_grid_path,
            magnetic_manifest_path,
        ) = load_magnetic_grid_from_manifest(
            Path(args.magnetic_manifest).expanduser().resolve()
        )
    if args.current_manifest:
        (
            current_field,
            _,
            current_field_grid_path,
            current_manifest_path,
        ) = load_current_field_from_manifest(
            Path(args.current_manifest).expanduser().resolve()
        )
    if args.tide_config:
        tide_config_path = Path(args.tide_config).expanduser().resolve()
    elif tide_config_path is None:
        tide_config_path = DEFAULT_TIDE_CONFIG.resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    learned_localizer_model_path = (
        None
        if args.learned_localizer_model is None
        else Path(args.learned_localizer_model).expanduser().resolve()
    )
    if args.map_matcher == "learned_sequence" and learned_localizer_model_path is None:
        parser.error(
            "--learned-localizer-model is required with --map-matcher learned_sequence."
        )
    initial_position_offset_ned_m = tuple(float(x) for x in args.initial_position_offset_ned_m)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = _load_scenario(scenario_path)
    profile = _load_sequence_profile(profile_path)

    imu_spec = _load_spec(Path(args.imu_config).expanduser().resolve(), IMUSpec)
    gravimeter_spec = _load_spec(Path(args.gravimeter_config).expanduser().resolve(), GravimeterSpec)
    photonic_spec = _load_spec(Path(args.photonic_config).expanduser().resolve(), PhotonicGravimeterSpec)
    depth_spec = _load_spec(Path(args.depth_config).expanduser().resolve(), DepthSensorSpec)
    velocity_spec = _load_spec(Path(args.velocity_config).expanduser().resolve(), VelocityAidSpec)
    gradiometer_spec = _load_spec(Path(args.gradiometer_config).expanduser().resolve(), GravityGradiometerSpec)
    bathymetry_spec = _load_spec(Path(args.bathymetry_config).expanduser().resolve(), BathymetrySensorSpec)
    magnetometer_spec = _load_spec(
        Path(args.magnetometer_config).expanduser().resolve(),
        MagnetometerSensorSpec,
    )
    current_profile_spec = _load_spec(
        Path(args.current_profile_config).expanduser().resolve(),
        CurrentProfileSensorSpec,
    )
    tide_correction_spec = _load_optional_spec(tide_config_path, TideCorrectionSpec)

    rows: list[dict[str, Any]] = []
    photonic_rows: list[dict[str, Any]] = []
    representative_result = None
    representative_label = None
    lag_validated = True

    for seed in [int(s) for s in args.seeds]:
        cases = [
            (
                "live_ins",
                dict(
                    disable_map_match=True,
                    gravimeter_spec=None,
                    photonic_spec=None,
                    use_tide_correction=False,
                    use_bathymetry=False,
                    use_acoustic_terrain=False,
                    use_magnetics=False,
                    use_current_correction=False,
                    use_lag_smoother=False,
                    matcher=args.map_matcher,
                    learned_localizer_model_path=learned_localizer_model_path,
                ),
            ),
            (
                "photonic_gravity_baseline",
                dict(
                    disable_map_match=False,
                    gravimeter_spec=None,
                    photonic_spec=photonic_spec,
                    use_tide_correction=False,
                    use_bathymetry=False,
                    use_acoustic_terrain=False,
                    use_magnetics=False,
                    use_current_correction=False,
                    use_lag_smoother=bool(profile.get("use_lag_smoother", True)),
                    matcher=args.map_matcher,
                    learned_localizer_model_path=learned_localizer_model_path,
                ),
            ),
            (
                "photonic_gravity_tide",
                dict(
                    disable_map_match=False,
                    gravimeter_spec=None,
                    photonic_spec=photonic_spec,
                    use_tide_correction=True,
                    use_bathymetry=False,
                    use_acoustic_terrain=False,
                    use_magnetics=False,
                    use_current_correction=False,
                    use_lag_smoother=bool(profile.get("use_lag_smoother", True)),
                    matcher=args.map_matcher,
                    learned_localizer_model_path=learned_localizer_model_path,
                ),
            ),
            (
                "photonic_gravity_tide_acoustic",
                dict(
                    disable_map_match=False,
                    gravimeter_spec=None,
                    photonic_spec=photonic_spec,
                    use_tide_correction=True,
                    use_bathymetry=True,
                    use_acoustic_terrain=True,
                    use_magnetics=False,
                    use_current_correction=False,
                    use_lag_smoother=bool(profile.get("use_lag_smoother", True)),
                    matcher=args.map_matcher,
                    learned_localizer_model_path=learned_localizer_model_path,
                ),
            ),
            (
                "photonic_gravity_tide_acoustic_magnetic",
                dict(
                    disable_map_match=False,
                    gravimeter_spec=None,
                    photonic_spec=photonic_spec,
                    use_tide_correction=True,
                    use_bathymetry=True,
                    use_acoustic_terrain=True,
                    use_magnetics=True,
                    use_current_correction=False,
                    use_lag_smoother=bool(profile.get("use_lag_smoother", True)),
                    matcher=args.map_matcher,
                    learned_localizer_model_path=learned_localizer_model_path,
                ),
            ),
            (
                "photonic_gravity_tide_acoustic_magnetic_current",
                dict(
                    disable_map_match=False,
                    gravimeter_spec=None,
                    photonic_spec=photonic_spec,
                    use_tide_correction=True,
                    use_bathymetry=True,
                    use_acoustic_terrain=True,
                    use_magnetics=True,
                    use_current_correction=True,
                    use_lag_smoother=bool(profile.get("use_lag_smoother", True)),
                    matcher=args.map_matcher,
                    learned_localizer_model_path=learned_localizer_model_path,
                ),
            ),
        ]
        for label, cfg in cases:
            result, metrics = _run_case(
                label,
                scenario=scenario,
                gravity_map=gravity_map,
                bathymetry_map=bathymetry_map,
                magnetic_map=magnetic_map,
                imu_spec=imu_spec,
                gravimeter_spec=cfg["gravimeter_spec"],
                photonic_spec=cfg["photonic_spec"],
                depth_spec=depth_spec,
                velocity_spec=velocity_spec,
                gradiometer_spec=gradiometer_spec,
                bathymetry_spec=bathymetry_spec,
                magnetometer_spec=magnetometer_spec,
                current_profile_spec=current_profile_spec,
                current_field=current_field,
                tide_correction_spec=tide_correction_spec,
                profile=profile,
                seed=seed,
                output_dir=output_dir,
                use_bathymetry=cfg["use_bathymetry"],
                use_acoustic_terrain=cfg["use_acoustic_terrain"],
                use_magnetics=cfg["use_magnetics"],
                use_current_correction=cfg["use_current_correction"],
                use_tide_correction=cfg["use_tide_correction"],
                matcher=cfg["matcher"],
                learned_localizer_model_path=cfg["learned_localizer_model_path"],
                disable_map_match=cfg["disable_map_match"],
                use_lag_smoother=cfg["use_lag_smoother"],
                dt_s=float(args.dt_s),
                initial_position_offset_ned_m=initial_position_offset_ned_m,
            )
            ambiguity_summary = _sequence_ambiguity_summary_from_result(result)
            search_center_summary = _search_center_summary_from_result(result)
            row = _metrics_row(
                label,
                metrics,
                result=result,
                ambiguity=ambiguity_summary,
                search_center=search_center_summary,
            )
            row["seed"] = seed
            row["map_matcher"] = str(args.map_matcher)
            row["learned_localizer_model"] = (
                None
                if learned_localizer_model_path is None
                else str(learned_localizer_model_path)
            )
            photonic_summary = _photonic_summary_from_result(result)
            if photonic_summary is not None:
                row.update(
                    {
                        "photonic_valid_sample_fraction": float(photonic_summary["valid_sample_fraction"]),
                        "photonic_median_fringe_contrast": float(photonic_summary["median_fringe_contrast"]),
                        "photonic_p95_fringe_contrast": float(photonic_summary["p95_fringe_contrast"]),
                        "photonic_rms_vibration_residual_phase_rad": float(
                            photonic_summary["rms_vibration_residual_phase_rad"]
                        ),
                        "photonic_rms_disturbance_residual_mps2": float(
                            photonic_summary["rms_disturbance_residual_mps2"]
                        ),
                        "photonic_tilt_exceedance_fraction": float(
                            photonic_summary["tilt_exceedance_fraction"]
                        ),
                        "photonic_median_estimated_measurement_variance_mps4": float(
                            photonic_summary["median_estimated_measurement_variance_mps4"]
                        ),
                    }
                )
                photonic_summary["label"] = label
                photonic_summary["seed"] = seed
                photonic_rows.append(photonic_summary)
            row["has_tide_correction"] = bool(cfg["use_tide_correction"] and tide_correction_spec is not None)
            row["has_bathymetry_map"] = bool(cfg["use_bathymetry"] and bathymetry_map is not None)
            row["has_magnetic_map"] = bool(cfg["use_magnetics"] and magnetic_map is not None)
            row["has_current_field"] = bool(cfg["use_current_correction"] and current_field is not None)
            rows.append(row)
            if label == "photonic_gravity_tide_acoustic" and representative_result is None:
                representative_result = result
                representative_label = label
            if label == "photonic_gravity_tide_acoustic_magnetic_current":
                lag_hmi = row["lag_hmi_horizontal"]
                lag_validated = lag_validated and (lag_hmi is not None) and (float(lag_hmi) == 0.0)

    summary_json_path = output_dir / "hardware_tied_maritime_demo_summary.json"
    summary_json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    photonic_summary_json_path = output_dir / "hardware_tied_maritime_demo_photonic_summary.json"
    photonic_summary_json_path.write_text(
        json.dumps(photonic_rows, indent=2) + "\n",
        encoding="utf-8",
    )

    _make_summary_plot(rows, out_path=output_dir / "hardware_tied_maritime_demo_summary.png")

    if representative_result is not None and representative_label is not None:
        fig1, ax1 = plot_ground_track_local_ned(representative_result, show_pf=False)
        fig1.savefig(output_dir / f"{representative_label}_ground_track.png", dpi=180)
        plt.close(fig1)
        fig2, _ = plot_position_error_ned(representative_result)
        fig2.savefig(output_dir / f"{representative_label}_position_error.png", dpi=180)
        plt.close(fig2)

    report_path = output_dir / "hardware_tied_maritime_demo_report.md"
    _write_report(
        report_path,
        scenario=scenario,
        demo_pack_path=(demo_pack_assets.manifest_path if demo_pack_assets is not None else None),
        demo_pack_region=(demo_pack_assets.manifest.region_name if demo_pack_assets is not None else None),
        gravity_map_path=gravity_map_path,
        gravity_manifest_path=gravity_manifest_path,
        bathymetry_grid_path=bathymetry_grid_path,
        bathymetry_manifest_path=bathymetry_manifest_path,
        magnetic_grid_path=magnetic_grid_path,
        magnetic_manifest_path=magnetic_manifest_path,
        current_field_grid_path=current_field_grid_path,
        current_manifest_path=current_manifest_path,
        tide_config_path=tide_config_path,
        profile_path=profile_path,
        profile=profile,
        rows=rows,
        photonic_rows=photonic_rows,
        seeds=[int(s) for s in args.seeds],
        lag_validated=lag_validated,
        dt_s=float(args.dt_s),
        initial_position_offset_ned_m=initial_position_offset_ned_m,
    )

    print(f"Summary JSON: {summary_json_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
