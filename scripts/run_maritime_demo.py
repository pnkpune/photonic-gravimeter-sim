#!/usr/bin/env python3
"""
Run the Norwegian maritime photonic demo and write a compact report.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
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
    ensure_regional_bathymetry_grid,
    load_demo_pack_manifest,
)
from gravnav.physics.gravity_map import GravityGridMap
from gravnav.plots.nav_plots import plot_ground_track_local_ned, plot_position_error_ned
from gravnav.sensors.bathymetry import BathymetrySensorSpec
from gravnav.sensors.depth import DepthSensorSpec
from gravnav.sensors.gravimeter import GravimeterSpec
from gravnav.sensors.gravity_gradiometer import GravityGradiometerSpec
from gravnav.sensors.imu import IMUSpec
from gravnav.sensors.photonic_gravimeter import PhotonicGravimeterSpec
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
DEFAULT_PHOTONIC_CONFIG = PROJECT_ROOT / "configs/sensors/photonic_gravimeter_proto.json"
DEFAULT_DEPTH_CONFIG = PROJECT_ROOT / "configs/sensors/depth_sensor.json"
DEFAULT_VELOCITY_CONFIG = PROJECT_ROOT / "configs/sensors/velocity_aid.json"
DEFAULT_GRADIOMETER_CONFIG = PROJECT_ROOT / "configs/sensors/gravity_gradiometer_proto.json"
DEFAULT_BATHY_SENSOR_CONFIG = PROJECT_ROOT / "configs/sensors/bathymetry_sensor.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/outputs/reports/maritime_demo"
DEFAULT_GRAVITY_MAP = PROJECT_ROOT / "data/gravity_maps/processed/norwegian_margin_gravity_map.npz"
DEFAULT_DEMO_PACK = PROJECT_ROOT / "data/bathymetry/processed/norwegian_margin_public_demo_pack.json"
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


def _metrics_row(label: str, metrics: ScenarioMetricsSummary) -> dict[str, float | str | None]:
    ins = metrics.ins_position_error
    seq = metrics.sequence_position_error
    lag = metrics.lag_smoothed_position_error
    integ = metrics.integrity
    lag_integ = metrics.lag_smoothed_integrity
    earth_rmse = (
        None
        if lag is None and seq is None and ins is None
        else float(lag.horizontal_rmse_m if lag is not None else seq.horizontal_rmse_m if seq is not None else ins.horizontal_rmse_m)
    )
    earth_cep95 = (
        None
        if lag is None and seq is None and ins is None
        else float(lag.cep95_m if lag is not None else seq.cep95_m if seq is not None else ins.cep95_m)
    )
    earth_hmi = (
        None
        if lag_integ is None and integ is None
        else float(
            lag_integ.fraction_hazardously_misleading_horizontal
            if lag_integ is not None
            else integ.fraction_hazardously_misleading_horizontal
        )
    )
    earth_mode = (
        "lag_smoothed"
        if lag is not None
        else "sequence"
        if seq is not None
        else "live_ins"
    )
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
        "earth_signature_mode": earth_mode,
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
    profile_path: Path,
    profile: dict[str, Any],
    rows: list[dict[str, Any]],
    seeds: list[int],
    lag_validated: bool,
    dt_s: float,
    initial_position_offset_ned_m: tuple[float, float, float],
) -> None:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row["label"]), []).append(row)

    def median(label: str, key: str) -> float:
        vals = [float(r[key]) for r in by_label[label] if r[key] is not None]
        return float(np.median(vals))

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
        "- theater: Norwegian-margin fixture gravity map + GEBCO bathymetry fixture",
        f"- gravity map runtime source: `{DEFAULT_GRAVITY_MAP}`",
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
        "| Mode | Reported Earth-signature output | Live INS RMSE [m] | Earth-signature RMSE [m] | Live INS CEP95 [m] | Earth-signature CEP95 [m] | HMI horiz |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in ordered_labels:
        if label not in by_label:
            continue
        lines.append(
            f"| `{label}` | `{by_label[label][0]['earth_signature_mode']}` | "
            f"{median(label, 'ins_horizontal_rmse_m'):.3f} | "
            f"{median(label, 'earth_signature_horizontal_rmse_m'):.3f} | "
            f"{median(label, 'ins_cep95_m'):.3f} | "
            f"{median(label, 'earth_signature_cep95_m'):.3f} | "
            f"{median(label, 'earth_signature_hmi_horizontal'):.3f} |"
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

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- photonic gravity beats live INS on median RMSE: `{median('photonic_gravity', 'sequence_horizontal_rmse_m') < median('live_ins', 'ins_horizontal_rmse_m')}`",
            f"- photonic gravity plus bathymetry beats photonic gravity on median RMSE: `{median('photonic_gravity_bathymetry', 'sequence_horizontal_rmse_m') < median('photonic_gravity', 'sequence_horizontal_rmse_m')}`",
            f"- lag-smoothed photonic gravity plus bathymetry beats live INS on median RMSE: `{median('photonic_gravity_bathymetry_lag', 'lag_horizontal_rmse_m') < median('live_ins', 'ins_horizontal_rmse_m')}`",
            f"- lag-smoothed photonic gravity plus bathymetry beats live INS on median CEP95: `{median('photonic_gravity_bathymetry_lag', 'lag_cep95_m') < median('live_ins', 'ins_cep95_m')}`",
            "- gravity remains the primary discriminator because the photonic-gravity path is compared directly against live INS before bathymetry is added.",
            "- bathymetry is treated as supporting passive context, not as a replacement for the gravity signature.",
            "- the recommended demo output is the bounded-lag Earth-signature track when it retains zero horizontal HMI.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Norwegian maritime photonic demo.")
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

    _ensure_demo_pack()
    load_demo_pack_manifest(DEFAULT_DEMO_PACK)

    scenario_path = Path(args.scenario).expanduser().resolve()
    profile_path = Path(args.sequence_profile).expanduser().resolve()
    gravity_map_path = Path(args.gravity_map).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    initial_position_offset_ned_m = tuple(float(x) for x in args.initial_position_offset_ned_m)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = _load_scenario(scenario_path)
    profile = _load_sequence_profile(profile_path)
    gravity_map = GravityGridMap.from_npz(gravity_map_path)
    bathymetry_map, _, _, _ = ensure_regional_bathymetry_grid(
        "norwegian_margin_public",
        project_root=PROJECT_ROOT,
    )

    imu_spec = _load_spec(Path(args.imu_config).expanduser().resolve(), IMUSpec)
    gravimeter_spec = _load_spec(Path(args.gravimeter_config).expanduser().resolve(), GravimeterSpec)
    photonic_spec = _load_spec(Path(args.photonic_config).expanduser().resolve(), PhotonicGravimeterSpec)
    depth_spec = _load_spec(Path(args.depth_config).expanduser().resolve(), DepthSensorSpec)
    velocity_spec = _load_spec(Path(args.velocity_config).expanduser().resolve(), VelocityAidSpec)
    gradiometer_spec = _load_spec(Path(args.gradiometer_config).expanduser().resolve(), GravityGradiometerSpec)
    bathymetry_spec = _load_spec(Path(args.bathymetry_config).expanduser().resolve(), BathymetrySensorSpec)

    rows: list[dict[str, Any]] = []
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
            row = _metrics_row(label, metrics)
            row["seed"] = seed
            rows.append(row)
            if label == "photonic_gravity_bathymetry" and representative_result is None:
                representative_result = result
                representative_label = label
            if label == "photonic_gravity_bathymetry_lag":
                lag_hmi = row["lag_hmi_horizontal"]
                lag_validated = lag_validated and (lag_hmi is not None) and (float(lag_hmi) == 0.0)

    summary_json_path = output_dir / "hardware_tied_maritime_demo_summary.json"
    summary_json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

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
        profile_path=profile_path,
        profile=profile,
        rows=rows,
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
