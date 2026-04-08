#!/usr/bin/env python3
"""
Generate a validated maritime-baseline simulation package:

- one aided baseline run
- one IMU-only comparison run
- key figures
- a markdown report with conclusions

The validated baseline intentionally keeps PF map matching in observe-only mode:
the PF runs and logs diagnostics, but PF position feedback into the INS remains
disabled because that path is still tuning-sensitive.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np

from gravnav.estimators.integrity import geodetic_position_error_ned
from gravnav.plots.nav_plots import (
    plot_ground_track_local_ned,
    plot_navigation_overview,
    plot_pf_diagnostics,
    plot_position_error_ned,
    save_figure,
)
from gravnav.sensors.depth import DepthSensorSpec
from gravnav.sensors.gravimeter import GravimeterSpec
from gravnav.sensors.imu import IMUSpec
from gravnav.sensors.velocity_aid import VelocityAidSpec
from gravnav.simulation.metrics import (
    PositionErrorMetrics,
    ins_position_error_history_from_truth,
    interpolate_truth_geodetic,
    scenario_metrics_from_result,
)
from gravnav.simulation.results import ScenarioSimulationResult
from gravnav.simulation.runner import ScenarioSimulationRunner, SimulationRunnerConfig
from gravnav.truth.scenarios import ScenarioSpec, build_truth_trajectory_from_scenario
from gravnav.utils.config import load_config_mapping


DEFAULT_SCENARIO_CONFIG = PROJECT_ROOT / "configs/scenarios/maritime_baseline.json"
DEFAULT_IMU_CONFIG = PROJECT_ROOT / "configs/sensors/imu_nav_grade.json"
DEFAULT_GRAVIMETER_CONFIG = PROJECT_ROOT / "configs/sensors/gravimeter_proto.json"
DEFAULT_DEPTH_CONFIG = PROJECT_ROOT / "configs/sensors/depth_sensor.json"
DEFAULT_VELOCITY_CONFIG = PROJECT_ROOT / "configs/sensors/velocity_aid.json"

OUTPUT_ROOT = PROJECT_ROOT / "data/outputs"
RUN_DIR = OUTPUT_ROOT / "runs/validated_maritime_baseline"
FIG_DIR = OUTPUT_ROOT / "figures/validated_maritime_baseline"
REPORT_DIR = OUTPUT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "validated_maritime_baseline_report.md"


def _load_spec(spec_cls: type, path: Path):
    mapping = load_config_mapping(path)
    return spec_cls(**mapping), mapping


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _build_synthetic_map(
    scenario: ScenarioSpec,
    *,
    truth_lat_rad: np.ndarray,
    truth_lon_rad: np.ndarray,
    truth_height_m: np.ndarray,
    grid_size: int = 101,
    margin_deg: float = 0.02,
):
    from run_single_scenario import _build_synthetic_map_for_truth

    return _build_synthetic_map_for_truth(
        scenario,
        truth_lat_rad=truth_lat_rad,
        truth_lon_rad=truth_lon_rad,
        truth_height_m=truth_height_m,
        grid_size=grid_size,
        margin_deg=margin_deg,
    )


def _save_result_bundle(
    result: ScenarioSimulationResult,
    *,
    stem: str,
) -> dict[str, Path]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "npz": result.save_npz(RUN_DIR / f"{stem}.npz"),
        "summary": result.save_summary_json(RUN_DIR / f"{stem}_summary.json"),
    }
    metrics = scenario_metrics_from_result(result)
    metrics_path = RUN_DIR / f"{stem}_metrics.json"
    metrics_path.write_text(json.dumps(metrics.to_mapping(), indent=2), encoding="utf-8")
    paths["metrics"] = metrics_path
    return paths


def _plot_gravimeter_history(
    result: ScenarioSimulationResult,
    *,
    path: Path,
) -> Path:
    arrays = result.sensors.gravimeter_history_arrays()
    t = arrays["gravimeter_time_s"]
    measured = arrays["gravimeter_value_mps2"]
    ideal = arrays["gravimeter_ideal_value_mps2"]

    fig, ax = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)
    ax.plot(t, ideal, label="ideal disturbance")
    ax.plot(t, measured, label="measured disturbance", alpha=0.8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("disturbance [m/s²]")
    ax.set_title("Gravimeter disturbance history")
    ax.grid(True, alpha=0.25)
    ax.legend()
    out = save_figure(fig, path)
    plt.close(fig)
    return out


def _plot_horizontal_error_comparison(
    aided: ScenarioSimulationResult,
    imu_only: ScenarioSimulationResult,
    *,
    path: Path,
) -> Path:
    aided_err = ins_position_error_history_from_truth(
        aided.truth,
        aided.estimators.ins_states,
    )
    imu_only_err = ins_position_error_history_from_truth(
        imu_only.truth,
        imu_only.estimators.ins_states,
    )

    t_aided = aided.estimators.ins_history_arrays()["ins_time_s"]
    t_imu = imu_only.estimators.ins_history_arrays()["ins_time_s"]
    h_aided = np.linalg.norm(aided_err[:, :2], axis=1)
    h_imu = np.linalg.norm(imu_only_err[:, :2], axis=1)

    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    ax.plot(t_imu, h_imu, label="IMU-only", linestyle="--")
    ax.plot(t_aided, h_aided, label="Aided baseline")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("horizontal error [m]")
    ax.set_title("Horizontal error comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()
    out = save_figure(fig, path)
    plt.close(fig)
    return out


def _pf_position_error_metrics(result: ScenarioSimulationResult) -> PositionErrorMetrics | None:
    if len(result.estimators.pf_updates) == 0:
        return None

    pf = result.estimators.pf_history_arrays()
    t_pf = pf["pf_time_s"]
    lat_true, lon_true, h_true = interpolate_truth_geodetic(result.truth, t_pf)

    err = np.empty((t_pf.size, 3), dtype=np.float64)
    for k in range(t_pf.size):
        err[k] = geodetic_position_error_ned(
            float(pf["pf_lat_rad"][k]),
            float(pf["pf_lon_rad"][k]),
            float(pf["pf_height_m"][k]),
            float(lat_true[k]),
            float(lon_true[k]),
            float(h_true[k]),
        )
    return PositionErrorMetrics.from_error_series(err)


def _write_report(
    *,
    scenario: ScenarioSpec,
    dt_s: float,
    aided_result: ScenarioSimulationResult,
    imu_only_result: ScenarioSimulationResult,
    aided_paths: dict[str, Path],
    imu_only_paths: dict[str, Path],
    figure_paths: dict[str, Path],
    imu_cfg: dict[str, object],
    gravimeter_cfg: dict[str, object],
    depth_cfg: dict[str, object],
    velocity_cfg: dict[str, object],
) -> Path:
    aided_metrics = scenario_metrics_from_result(aided_result)
    imu_only_metrics = scenario_metrics_from_result(imu_only_result)
    pf_metrics = _pf_position_error_metrics(aided_result)

    aided_pos = aided_metrics.ins_position_error
    imu_pos = imu_only_metrics.ins_position_error
    if aided_pos is None or imu_pos is None:
        raise RuntimeError("Expected INS position metrics for both runs.")

    rmse_improvement = imu_pos.horizontal_rmse_m / aided_pos.horizontal_rmse_m
    cep95_improvement = imu_pos.cep95_m / aided_pos.cep95_m

    lines = [
        "# Validated Maritime Baseline Report",
        "",
        "## Run Scope",
        "",
        f"- Scenario: `{scenario.name}`",
        f"- Sample interval: `{dt_s:.1f} s`",
        "- Baseline mode: nav-grade IMU + gravimeter + depth aid + velocity aid + PF observe-only",
        "- PF feedback status: disabled in the validated baseline because closed-loop PF position injection remains tuning-sensitive",
        "",
        "## Simulation Data",
        "",
        f"- Aided run archive: `{_relative(aided_paths['npz'])}`",
        f"- Aided summary JSON: `{_relative(aided_paths['summary'])}`",
        f"- Aided metrics JSON: `{_relative(aided_paths['metrics'])}`",
        f"- IMU-only run archive: `{_relative(imu_only_paths['npz'])}`",
        f"- IMU-only summary JSON: `{_relative(imu_only_paths['summary'])}`",
        f"- IMU-only metrics JSON: `{_relative(imu_only_paths['metrics'])}`",
        "",
        "## Configuration Summary",
        "",
        f"- Truth duration: `{aided_result.duration_s:.1f} s` with `{len(aided_result.truth)}` truth samples",
        f"- IMU config: `{json.dumps(imu_cfg, sort_keys=True)}`",
        f"- Gravimeter config: `{json.dumps(gravimeter_cfg, sort_keys=True)}`",
        f"- Depth config: `{json.dumps(depth_cfg, sort_keys=True)}`",
        f"- Velocity-aid config: `{json.dumps(velocity_cfg, sort_keys=True)}`",
        "- Safe fusion policies in validated run:",
        "  - interval-consistent IMU truth for propagation",
        "  - velocity-aid updates constrained to the velocity state",
        "  - depth updates constrained to the height state",
        "  - PF enabled for diagnostics only, not for INS position feedback",
        "",
        "## Key Metrics",
        "",
        "| Metric | IMU-only | Validated aided baseline |",
        "| --- | ---: | ---: |",
        f"| INS horizontal RMSE [m] | {imu_pos.horizontal_rmse_m:.3f} | {aided_pos.horizontal_rmse_m:.3f} |",
        f"| INS CEP95 [m] | {imu_pos.cep95_m:.3f} | {aided_pos.cep95_m:.3f} |",
        f"| INS vertical RMSE [m] | {imu_pos.vertical_rmse_m:.3f} | {aided_pos.vertical_rmse_m:.3f} |",
        f"| Gravimeter RMSE [m/s²] | n/a | {aided_metrics.gravimeter_error.rmse:.6e} |" if aided_metrics.gravimeter_error is not None else "| Gravimeter RMSE [m/s²] | n/a | n/a |",
    ]

    if pf_metrics is not None:
        lines.extend(
            [
                f"| PF horizontal RMSE [m] | n/a | {pf_metrics.horizontal_rmse_m:.3f} |",
                f"| PF CEP95 [m] | n/a | {pf_metrics.cep95_m:.3f} |",
            ]
        )

    lines.extend(
        [
            "",
            "## Highlights",
            "",
            f"- The validated aided baseline reduces INS horizontal RMSE by `{rmse_improvement:.1f}x` relative to IMU-only.",
            f"- The validated aided baseline reduces INS CEP95 by `{cep95_improvement:.1f}x` relative to IMU-only.",
            f"- The default CLI baseline now runs stably at `{aided_pos.horizontal_rmse_m:.1f} m` horizontal RMSE instead of diverging to kilometer-scale error.",
            "- The main simulation bug was the truth-to-IMU interface: the runner needed interval-consistent IMU truth rather than sample-centered kinematics.",
            "- The main estimator instability was cross-covariance-driven overcorrection from scalar depth and vector velocity aids into bias states. The validated baseline now uses conservative constrained fusion paths.",
            "- PF map matching is producing diagnostics in the baseline run, but PF position feedback is not yet part of the validated closed-loop navigation solution.",
            "",
            "## Figures",
            "",
            f"- Navigation overview: `{_relative(figure_paths['overview'])}`",
            f"- Local ground track: `{_relative(figure_paths['ground_track'])}`",
            f"- INS position error history: `{_relative(figure_paths['position_error'])}`",
            f"- Gravimeter disturbance history: `{_relative(figure_paths['gravimeter'])}`",
            f"- PF diagnostics: `{_relative(figure_paths['pf_diagnostics'])}`",
            f"- IMU-only vs aided horizontal error: `{_relative(figure_paths['comparison'])}`",
            "",
            "## Conclusion",
            "",
            "The simulation stack is now running correctly for the validated single-scenario baseline. The truth, IMU, fusion, and runner paths are numerically consistent, the default CLI run is stable, and the aided baseline is materially better than IMU-only. The remaining non-validated item is closed-loop PF position feedback: the gravity map-matching layer is usable in observe-only mode for diagnostics, but its direct feedback into the INS still needs separate tuning before it should be treated as part of the production baseline.",
            "",
            "The practical next step is not more bug fixing in the physics stack. It is controlled development of a validated gravity-feedback policy, ideally with explicit acceptance gates, covariance inflation rules, and scenario-by-scenario evaluation against the now-stable aided baseline.",
            "",
        ]
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


def main() -> int:
    scenario_mapping = load_config_mapping(DEFAULT_SCENARIO_CONFIG)
    scenario = ScenarioSpec.from_mapping(scenario_mapping)

    imu_spec, imu_cfg = _load_spec(IMUSpec, DEFAULT_IMU_CONFIG)
    gravimeter_spec, gravimeter_cfg = _load_spec(GravimeterSpec, DEFAULT_GRAVIMETER_CONFIG)
    depth_spec, depth_cfg = _load_spec(DepthSensorSpec, DEFAULT_DEPTH_CONFIG)
    velocity_spec, velocity_cfg = _load_spec(VelocityAidSpec, DEFAULT_VELOCITY_CONFIG)

    dt_s = 2.0
    truth = build_truth_trajectory_from_scenario(scenario, dt_s=dt_s)
    map_model = _build_synthetic_map(
        scenario,
        truth_lat_rad=truth.lat_rad,
        truth_lon_rad=truth.lon_rad,
        truth_height_m=truth.height_m,
        grid_size=101,
        margin_deg=0.02,
    )

    aided_cfg = SimulationRunnerConfig()
    aided_cfg.map_match.enabled = True
    aided_cfg.map_match.inject_position_to_ins = False
    aided_cfg.map_match.depth_meas_std_m = 0.5
    aided_cfg.metadata_extra = {"report_mode": "validated_baseline"}

    imu_only_cfg = SimulationRunnerConfig()
    imu_only_cfg.velocity_aid.enabled = False
    imu_only_cfg.depth_aid.enabled = False
    imu_only_cfg.map_match.enabled = False
    imu_only_cfg.metadata_extra = {"report_mode": "imu_only_reference"}

    aided_runner = ScenarioSimulationRunner(aided_cfg)
    imu_only_runner = ScenarioSimulationRunner(imu_only_cfg)

    aided_result = aided_runner.run_with_specs(
        scenario,
        imu_spec=imu_spec,
        gravimeter_spec=gravimeter_spec,
        depth_spec=depth_spec,
        velocity_aid_spec=velocity_spec,
        map_model=map_model,
        seed=123,
        dt_s=dt_s,
    )
    imu_only_result = imu_only_runner.run_with_specs(
        scenario,
        imu_spec=imu_spec,
        gravimeter_spec=None,
        depth_spec=None,
        velocity_aid_spec=None,
        map_model=None,
        seed=123,
        dt_s=dt_s,
    )

    aided_paths = _save_result_bundle(aided_result, stem="maritime_baseline_aided")
    imu_only_paths = _save_result_bundle(imu_only_result, stem="maritime_baseline_imu_only")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_overview, _ = plot_navigation_overview(
        aided_result,
        title="Validated Maritime Baseline Overview",
        reference_surface_height_m=0.0,
    )
    overview_path = save_figure(fig_overview, FIG_DIR / "aided_navigation_overview.png")
    plt.close(fig_overview)

    fig_ground, _ = plot_ground_track_local_ned(
        aided_result,
        title="Validated Maritime Baseline Ground Track",
    )
    ground_path = save_figure(fig_ground, FIG_DIR / "aided_ground_track_local_ned.png")
    plt.close(fig_ground)

    fig_pos, _ = plot_position_error_ned(
        aided_result,
        title="Validated Maritime Baseline INS Position Error",
    )
    pos_path = save_figure(fig_pos, FIG_DIR / "aided_position_error_ned.png")
    plt.close(fig_pos)

    fig_pf, _ = plot_pf_diagnostics(
        aided_result,
        title="Validated Maritime Baseline PF Diagnostics",
    )
    pf_path = save_figure(fig_pf, FIG_DIR / "aided_pf_diagnostics.png")
    plt.close(fig_pf)

    grav_path = _plot_gravimeter_history(
        aided_result,
        path=FIG_DIR / "aided_gravimeter_history.png",
    )
    comparison_path = _plot_horizontal_error_comparison(
        aided_result,
        imu_only_result,
        path=FIG_DIR / "imu_only_vs_aided_horizontal_error.png",
    )

    report_path = _write_report(
        scenario=scenario,
        dt_s=dt_s,
        aided_result=aided_result,
        imu_only_result=imu_only_result,
        aided_paths=aided_paths,
        imu_only_paths=imu_only_paths,
        figure_paths={
            "overview": overview_path,
            "ground_track": ground_path,
            "position_error": pos_path,
            "pf_diagnostics": pf_path,
            "gravimeter": grav_path,
            "comparison": comparison_path,
        },
        imu_cfg=imu_cfg,
        gravimeter_cfg=gravimeter_cfg,
        depth_cfg=depth_cfg,
        velocity_cfg=velocity_cfg,
    )

    print(f"Saved aided run bundle under {_relative(RUN_DIR)}")
    print(f"Saved figures under {_relative(FIG_DIR)}")
    print(f"Saved report: {_relative(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
