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
from gravnav.physics.gravity_map import GravityGridMap
from gravnav.plots.nav_plots import plot_ground_track_local_ned, plot_position_error_ned
from gravnav.sensors.bathymetry import BathymetrySensorSpec
from gravnav.sensors.depth import DepthSensorSpec
from gravnav.sensors.gravimeter import GravimeterSpec
from gravnav.sensors.gravity_gradiometer import GravityGradiometerSpec
from gravnav.sensors.imu import IMUSpec
from gravnav.sensors.photonic_gravimeter import PhotonicGravimeterSpec
from gravnav.sensors.photonic_gravimeter import summarize_photonic_measurements
from gravnav.sensors.velocity_aid import VelocityAidSpec
from gravnav.simulation.metrics import (
    ScenarioMetricsSummary,
    scenario_metrics_from_result,
)
from gravnav.simulation.results import SimulationMetadata
from gravnav.simulation.runner import (
    DepthFusionConfig,
    GravitySequenceMatcherSpec,
    IntegrityMonitorConfig,
    MapMatchFeedbackConfig,
    PeriodicUpdateSchedule,
    ScenarioSimulationRunner,
    SimulationRunnerConfig,
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


def _ensure_demo_pack() -> None:
    if DEFAULT_DEMO_PACK.exists():
        return
    subprocess.run([sys.executable, str(PREP_SCRIPT)], check=True, cwd=PROJECT_ROOT)


def _load_demo_pack_assets(path: Path) -> ResolvedRegionalDemoPack:
    return resolve_regional_demo_pack(path)


def _build_runner_config(
    profile: dict[str, Any],
    *,
    use_bathymetry: bool,
    use_lag_smoother: bool,
    initial_position_offset_ned_m: tuple[float, float, float],
) -> SimulationRunnerConfig:
    map_match = MapMatchFeedbackConfig(
        enabled=True,
        matcher="sequence",
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
            height_std_m=float(profile.get("height_std_m", 2.0)),
            adaptive_grid_enabled=bool(profile.get("adaptive_grid_enabled", True)),
            expanded_grid_half_span_m=profile.get(
                "expanded_grid_half_span_m",
                [2.0 * float(v) for v in profile["grid_half_span_m"]],
            ),
            expanded_grid_spacing_m=profile.get(
                "expanded_grid_spacing_m",
                list(profile["grid_spacing_m"]),
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
        ),
        gravity_meas_std_mps2=float(profile["gravity_meas_std_mps2"]),
        use_gradiometer=bool(profile.get("use_gradiometer", True)),
        gradient_meas_std_per_s2=float(profile["gradient_meas_std_per_s2"]),
        use_bathymetry=bool(use_bathymetry),
        bathymetry_meas_std_m=(
            None if not use_bathymetry else float(profile["bathymetry_meas_std_m"])
        ),
        use_sequence_lag_smoother=bool(use_lag_smoother),
    )
    return SimulationRunnerConfig(
        initial_position_offset_ned_m=initial_position_offset_ned_m,
        velocity_aid=VelocityAidFusionConfig(
            enabled=True,
            schedule=PeriodicUpdateSchedule(every_steps=1),
            measurement_frame="ned",
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
    imu_spec: IMUSpec,
    gravimeter_spec: GravimeterSpec | None,
    photonic_spec: PhotonicGravimeterSpec | None,
    depth_spec: DepthSensorSpec,
    velocity_spec: VelocityAidSpec,
    gradiometer_spec: GravityGradiometerSpec,
    bathymetry_spec: BathymetrySensorSpec | None,
    profile: dict[str, Any],
    seed: int,
    output_dir: Path,
    use_bathymetry: bool,
    disable_map_match: bool = False,
    use_lag_smoother: bool = False,
    dt_s: float = 2.0,
    initial_position_offset_ned_m: tuple[float, float, float] = (60.0, -30.0, 0.0),
) -> tuple[Any, ScenarioMetricsSummary]:
    cfg = _build_runner_config(
        profile,
        use_bathymetry=use_bathymetry,
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
    support_radius_fraction: list[float] = []

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
        "median_support_radius_fraction": float(np.median(support_radius_fraction)),
        "failure_mode_counts": failure_counts,
        "grid_mode_counts": grid_mode_counts,
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
    if median_bathy_info is not None:
        median_bathy_info = float(median_bathy_info)

    lag_allowed = (
        lag is not None
        and lag_integ is not None
        and float(lag_integ.fraction_hazardously_misleading_horizontal) == 0.0
        and edge_clipped <= 0.15
        and median_edge_mass <= 0.12
        and median_support_radius <= 0.75
        and median_ess >= 0.02
        and (median_bathy_info is None or median_bathy_info >= 0.10)
    )
    if lag_allowed:
        return "lag_smoothed", "lag_confident"

    sequence_allowed = (
        edge_clipped <= 0.25
        and median_edge_mass <= 0.20
        and median_support_radius <= 0.85
        and median_ess >= 0.02
        and (median_gravity_info >= 0.10 or (median_bathy_info is not None and median_bathy_info >= 0.10))
    )
    if sequence_allowed:
        return "sequence", "lag_rejected_or_unavailable"

    if edge_clipped > 0.35 or median_support_radius > 0.90:
        return "live_ins", "edge_coverage_limited"
    if median_gravity_info < 0.10 and (median_bathy_info is None or median_bathy_info < 0.10):
        return "live_ins", "flat_signature"
    if median_ess < 0.02:
        return "live_ins", "prior_dominated"
    return "live_ins", "sequence_ambiguous"


def _metrics_row(
    label: str,
    metrics: ScenarioMetricsSummary,
    *,
    ambiguity: dict[str, Any] | None,
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
    if reported_mode == "lag_smoothed" and lag is not None:
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
        "ambiguity_failure_mode_counts": None
        if ambiguity is None
        else dict(ambiguity["failure_mode_counts"]),
        "ambiguity_grid_mode_counts": None
        if ambiguity is None
        else dict(ambiguity["grid_mode_counts"]),
    }


def _photonic_summary_from_result(result: Any) -> dict[str, Any] | None:
    samples = result.sensors.custom_streams.get("photonic_gravimeter", [])
    if len(samples) == 0:
        return None
    return asdict(summarize_photonic_measurements(samples))


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

    ordered_labels = [
        "live_ins",
        "surrogate_gravity",
        "photonic_gravity",
        "photonic_gravity_bathymetry",
        "photonic_gravity_bathymetry_lag",
    ]
    lines = [
        "# Hardware-Tied Maritime Demo Report",
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
        "- estimator: observe-only sequence matcher with gravity always enabled",
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
                "| Mode | Informative frac | Edge-clipped frac | Prior-dominated frac | Flat-signature frac | Expanded-grid frac | Median edge mass | Median support radius frac | Median gravity info ratio |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
                f"{median(label, 'ambiguity_median_gravity_information_ratio'):.3f} |"
            )
    if "photonic_gravity_bathymetry_lag" in by_label:
        lines.extend(
            [
                "",
                "## Lag-Smoother Result",
                "",
                f"- median lag-smoothed RMSE: `{median('photonic_gravity_bathymetry_lag', 'lag_horizontal_rmse_m'):.3f} m`",
                f"- median lag-smoothed CEP95: `{median('photonic_gravity_bathymetry_lag', 'lag_cep95_m'):.3f} m`",
                f"- lag-smoothed validated: `{lag_validated}`",
            ]
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
            f"- photonic gravity beats live INS on median RMSE: `{median('photonic_gravity', 'sequence_horizontal_rmse_m') < median('live_ins', 'ins_horizontal_rmse_m')}`",
            f"- photonic gravity plus bathymetry beats photonic gravity on median RMSE: `{median('photonic_gravity_bathymetry', 'sequence_horizontal_rmse_m') < median('photonic_gravity', 'sequence_horizontal_rmse_m')}`",
            f"- reported output beats live INS on median RMSE: `{median('photonic_gravity_bathymetry_lag', 'earth_signature_horizontal_rmse_m') < median('live_ins', 'ins_horizontal_rmse_m')}`",
            f"- reported output beats live INS on median CEP95: `{median('photonic_gravity_bathymetry_lag', 'earth_signature_cep95_m') < median('live_ins', 'ins_cep95_m')}`",
            "- gravity remains the primary discriminator because the photonic-gravity path is compared directly against live INS before bathymetry is added.",
            "- bathymetry is treated as supporting passive context, not as a replacement for the gravity signature.",
            "- the recommended demo output is selected adaptively: lag when confidence is high, sequence when lag is too ambiguous, otherwise live INS.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Norwegian maritime photonic demo.")
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
    parser.add_argument("--gravity-map", default=str(DEFAULT_GRAVITY_MAP))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    parser.add_argument("--dt-s", type=float, default=2.0)
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
    output_dir = Path(args.output_dir).expanduser().resolve()
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

    rows: list[dict[str, Any]] = []
    photonic_rows: list[dict[str, Any]] = []
    representative_result = None
    representative_label = None
    lag_validated = True

    for seed in [int(s) for s in args.seeds]:
        cases = [
            ("live_ins", dict(disable_map_match=True, gravimeter_spec=None, photonic_spec=None, use_bathymetry=False, use_lag_smoother=False)),
            ("surrogate_gravity", dict(disable_map_match=False, gravimeter_spec=gravimeter_spec, photonic_spec=None, use_bathymetry=False, use_lag_smoother=False)),
            ("photonic_gravity", dict(disable_map_match=False, gravimeter_spec=None, photonic_spec=photonic_spec, use_bathymetry=False, use_lag_smoother=False)),
            ("photonic_gravity_bathymetry", dict(disable_map_match=False, gravimeter_spec=None, photonic_spec=photonic_spec, use_bathymetry=True, use_lag_smoother=False)),
            ("photonic_gravity_bathymetry_lag", dict(disable_map_match=False, gravimeter_spec=None, photonic_spec=photonic_spec, use_bathymetry=True, use_lag_smoother=True)),
        ]
        for label, cfg in cases:
            result, metrics = _run_case(
                label,
                scenario=scenario,
                gravity_map=gravity_map,
                bathymetry_map=bathymetry_map,
                imu_spec=imu_spec,
                gravimeter_spec=cfg["gravimeter_spec"],
                photonic_spec=cfg["photonic_spec"],
                depth_spec=depth_spec,
                velocity_spec=velocity_spec,
                gradiometer_spec=gradiometer_spec,
                bathymetry_spec=bathymetry_spec,
                profile=profile,
                seed=seed,
                output_dir=output_dir,
                use_bathymetry=cfg["use_bathymetry"],
                disable_map_match=cfg["disable_map_match"],
                use_lag_smoother=cfg["use_lag_smoother"],
                dt_s=float(args.dt_s),
                initial_position_offset_ned_m=initial_position_offset_ned_m,
            )
            ambiguity_summary = _sequence_ambiguity_summary_from_result(result)
            row = _metrics_row(
                label,
                metrics,
                ambiguity=ambiguity_summary,
            )
            row["seed"] = seed
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
            rows.append(row)
            if label == "photonic_gravity_bathymetry" and representative_result is None:
                representative_result = result
                representative_label = label
            if label == "photonic_gravity_bathymetry_lag":
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
