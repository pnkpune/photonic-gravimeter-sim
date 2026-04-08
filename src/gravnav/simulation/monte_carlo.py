"""
monte_carlo.py

Monte Carlo orchestration and aggregate analysis for the gravity-aided
navigation simulator.

Why this file exists
--------------------
The repository already has:
- truth trajectories and named scenarios
- sensor models
- INS / fusion / PF / integrity logic
- one-run simulation result containers
- one-run scenario metric summaries

What is still missing is the study-level layer that can:
1) run many independent scenario realizations,
2) keep reproducible per-run RNG provenance,
3) aggregate scalar performance metrics across runs,
4) optionally parallelize CPU-bound studies,
5) optionally save individual run archives and a study summary.

Design philosophy
-----------------
This file intentionally builds on the already-implemented one-run APIs instead of
creating a parallel framework-specific abstraction. The core idea is:

    Monte Carlo study
        = repeated calls to ScenarioSimulationRunner
        + one ScenarioMetricsSummary per run
        + aggregate statistics over the metric leaves

Conventions
-----------
- One Monte Carlo "run" means one full scenario simulation with independent
  stochastic sensor realizations.
- Aggregation is performed over numeric leaves extracted from each
  `ScenarioMetricsSummary`.
- Missing values remain missing: each aggregate tracks how many finite samples it
  actually used.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, is_dataclass
import json
from pathlib import Path
import traceback
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..estimators.error_state_ins import ErrorStateINSProcessNoise
from ..estimators.map_match_pf import MapMatchPFSpec
from ..sensors.depth import DepthSensorSpec
from ..sensors.gravimeter import GravimeterSpec
from ..sensors.imu import IMUSpec
from ..sensors.velocity_aid import VelocityAidSpec
from ..truth.scenarios import ScenarioSpec
from ..truth.trajectory import TruthTrajectory
from ..utils.config import (
    apply_overrides,
    dump_config_mapping,
    load_monte_carlo_config,
    load_scenario_mapping,
    load_scenario_spec,
    load_sensor_config,
)
from ..utils.rng import monte_carlo_generators, random_uint64, seed_record_from_seed
from .metrics import ScenarioMetricsSummary, scenario_metrics_from_result
from .results import ScenarioSimulationResult, SimulationMetadata
from .runner import (
    DepthFusionConfig,
    IntegrityMonitorConfig,
    MapMatchFeedbackConfig,
    PeriodicUpdateSchedule,
    ScenarioSimulationRunner,
    SimulationRunnerConfig,
    VelocityAidFusionConfig,
)

FloatArray = NDArray[np.float64]


# -----------------------------------------------------------------------------
# Low-level helpers
# -----------------------------------------------------------------------------


def _as_float_array(x: ArrayLike) -> FloatArray:
    """Convert input to a NumPy float64 array."""
    return np.asarray(x, dtype=np.float64)


def _positive_int(x: int, *, name: str) -> int:
    """Validate and return a positive integer."""
    value = int(x)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _nonnegative_int(x: int, *, name: str) -> int:
    """Validate and return a nonnegative integer."""
    value = int(x)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}.")
    return value


def _jsonable(obj: Any) -> Any:
    """
    Convert nested dataclass / NumPy-heavy structures into JSON-friendly values.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if is_dataclass(obj):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _object_to_mapping(obj: Any) -> Any:
    """
    Convert a supported object into a mapping/list/scalar structure for flattening.

    Priority:
    - None / scalar passthrough
    - `.to_mapping()`
    - dataclass via `asdict`
    - generic mapping
    """
    if obj is None or isinstance(obj, (str, int, float, bool, np.generic)):
        return _jsonable(obj)

    to_mapping = getattr(obj, "to_mapping", None)
    if callable(to_mapping):
        return _jsonable(to_mapping())

    if is_dataclass(obj):
        return _jsonable(asdict(obj))

    if isinstance(obj, Mapping):
        return _jsonable(obj)

    if isinstance(obj, np.ndarray):
        return _jsonable(obj)

    return _jsonable(obj)


def _flatten_numeric_leaves(
    obj: Any,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """
    Flatten nested mappings/dataclasses into a mapping of numeric scalar leaves.

    Non-numeric leaves are ignored.

    Examples
    --------
    A nested object like:

        {"ins_position_error": {"horizontal_rmse_m": 12.3}}

    becomes:

        {"ins_position_error.horizontal_rmse_m": 12.3}
    """
    root = _object_to_mapping(obj)
    out: dict[str, float] = {}

    def walk(node: Any, path: str) -> None:
        if node is None:
            return

        if isinstance(node, np.generic):
            node = node.item()

        if isinstance(node, bool):
            out[path] = float(node)
            return

        if isinstance(node, (int, float)):
            out[path] = float(node)
            return

        if isinstance(node, Mapping):
            for key, value in node.items():
                key_str = str(key)
                child_path = key_str if path == "" else f"{path}.{key_str}"
                walk(value, child_path)
            return

        if isinstance(node, (list, tuple)):
            # Metric summaries in this repository are mapping/dataclass based.
            # Lists are not expected here, so skip them rather than inventing
            # brittle positional names.
            return

    walk(root, prefix)
    return out


def _archive_filename_from_template(
    template: str,
    *,
    run_index: int,
    run_id: Optional[str],
) -> str:
    """
    Format an archive filename template.

    Supported fields
    ----------------
    - `{index}`
    - `{index04d}`
    - `{run_id}`
    """
    text = str(template)
    safe_run_id = "" if run_id is None else str(run_id)
    return (
        text.replace("{index04d}", f"{int(run_index):04d}")
        .replace("{index}", str(int(run_index)))
        .replace("{run_id}", safe_run_id)
    )


# -----------------------------------------------------------------------------
# Aggregate metric containers
# -----------------------------------------------------------------------------


@dataclass
class MonteCarloScalarAggregate:
    """
    Aggregate summary for one scalar metric across Monte Carlo runs.

    Attributes
    ----------
    count : int
        Number of finite values used.
    mean : float
        Arithmetic mean.
    std : float
        Population standard deviation.
    median : float
        Median.
    p05 : float
        5th percentile.
    p95 : float
        95th percentile.
    min : float
        Minimum finite value.
    max : float
        Maximum finite value.
    """

    count: int
    mean: float
    std: float
    median: float
    p05: float
    p95: float
    min: float
    max: float

    @classmethod
    def from_values(cls, values: ArrayLike) -> "MonteCarloScalarAggregate":
        """
        Build an aggregate summary from a 1D array-like input.
        """
        arr = _as_float_array(values).reshape(-1)
        vals = arr[np.isfinite(arr)]

        if vals.size == 0:
            nan = float(np.nan)
            return cls(
                count=0,
                mean=nan,
                std=nan,
                median=nan,
                p05=nan,
                p95=nan,
                min=nan,
                max=nan,
            )

        return cls(
            count=int(vals.size),
            mean=float(np.mean(vals)),
            std=float(np.std(vals)),
            median=float(np.median(vals)),
            p05=float(np.percentile(vals, 5.0)),
            p95=float(np.percentile(vals, 95.0)),
            min=float(np.min(vals)),
            max=float(np.max(vals)),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return _jsonable(asdict(self))


@dataclass
class MonteCarloStudySummary:
    """
    Aggregate summary of one Monte Carlo study.

    Attributes
    ----------
    n_runs_requested : int
        Number of runs requested.
    n_runs_completed : int
        Number of successful runs.
    n_runs_failed : int
        Number of failed runs.
    failed_run_indices : tuple[int, ...]
        Indices of failed runs.
    metric_aggregates : dict[str, MonteCarloScalarAggregate]
        Aggregate statistics keyed by flattened metric name, for example:
        `ins_position_error.horizontal_rmse_m`.
    """

    n_runs_requested: int
    n_runs_completed: int
    n_runs_failed: int
    failed_run_indices: tuple[int, ...] = ()
    metric_aggregates: dict[str, MonteCarloScalarAggregate] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return {
            "n_runs_requested": int(self.n_runs_requested),
            "n_runs_completed": int(self.n_runs_completed),
            "n_runs_failed": int(self.n_runs_failed),
            "failed_run_indices": [int(v) for v in self.failed_run_indices],
            "metric_aggregates": {
                key: value.to_mapping() for key, value in self.metric_aggregates.items()
            },
        }

    def get(self, metric_name: str) -> Optional[MonteCarloScalarAggregate]:
        """
        Return one aggregate by metric name, or None when absent.
        """
        return self.metric_aggregates.get(str(metric_name))


# -----------------------------------------------------------------------------
# Run record and top-level study result
# -----------------------------------------------------------------------------


@dataclass
class MonteCarloRunRecord:
    """
    Result record for one Monte Carlo run.

    Attributes
    ----------
    run_index : int
        Zero-based run index.
    run_seed : int
        Integer seed used for this run's top-level sensor spawning.
    success : bool
        True when the run completed successfully.
    result : ScenarioSimulationResult or None
        Full run result when retained.
    metrics : ScenarioMetricsSummary or None
        One-run metric summary when available.
    error_message : str or None
        Short failure message.
    traceback_text : str or None
        Full traceback for failed runs.
    archive_path : str or None
        Optional path to a saved NPZ archive.
    """

    run_index: int
    run_seed: int
    success: bool
    result: Optional[ScenarioSimulationResult] = None
    metrics: Optional[ScenarioMetricsSummary] = None
    error_message: Optional[str] = None
    traceback_text: Optional[str] = None
    archive_path: Optional[str] = None

    def to_mapping(self) -> dict[str, Any]:
        """
        Return a JSON-friendly mapping.

        The full `result` object is intentionally not serialized here. Use the
        saved archive path or `result.summary_mapping()` when you need a compact
        representation.
        """
        summary = None
        metadata = None
        if self.result is not None:
            summary = self.result.summary_mapping()
            metadata = self.result.metadata.to_mapping()

        return {
            "run_index": int(self.run_index),
            "run_seed": int(self.run_seed),
            "success": bool(self.success),
            "summary": summary,
            "metadata": metadata,
            "metrics": None if self.metrics is None else self.metrics.to_mapping(),
            "error_message": self.error_message,
            "traceback_text": self.traceback_text,
            "archive_path": self.archive_path,
        }


@dataclass
class MonteCarloStudyResult:
    """
    Complete result of one Monte Carlo study.

    Attributes
    ----------
    config : MonteCarloStudyConfig
        Study-level execution configuration.
    runner_config : SimulationRunnerConfig
        Per-run simulation runner configuration.
    metadata : SimulationMetadata
        Base metadata used for the study.
    runs : list[MonteCarloRunRecord]
        Per-run results in run-index order.
    """

    config: "MonteCarloStudyConfig"
    runner_config: SimulationRunnerConfig
    metadata: SimulationMetadata = field(default_factory=SimulationMetadata)
    runs: list[MonteCarloRunRecord] = field(default_factory=list)

    def successful_runs(self) -> list[MonteCarloRunRecord]:
        """Return successful run records."""
        return [r for r in self.runs if r.success]

    def failed_runs(self) -> list[MonteCarloRunRecord]:
        """Return failed run records."""
        return [r for r in self.runs if not r.success]

    def metric_series(self) -> dict[str, FloatArray]:
        """
        Collect flattened scalar metric series across successful runs.

        Returns
        -------
        dict[str, np.ndarray]
            Metric-name -> values across successful runs.
        """
        series: dict[str, list[float]] = {}

        for record in self.successful_runs():
            if record.metrics is None:
                continue
            flat = _flatten_numeric_leaves(record.metrics)
            for key, value in flat.items():
                series.setdefault(key, []).append(float(value))

        return {
            key: np.asarray(values, dtype=np.float64)
            for key, values in series.items()
        }

    def summary(self) -> MonteCarloStudySummary:
        """
        Build a study-level aggregate summary.
        """
        successful = self.successful_runs()
        failed = self.failed_runs()

        aggregates = {
            key: MonteCarloScalarAggregate.from_values(values)
            for key, values in self.metric_series().items()
        }

        return MonteCarloStudySummary(
            n_runs_requested=len(self.runs),
            n_runs_completed=len(successful),
            n_runs_failed=len(failed),
            failed_run_indices=tuple(int(r.run_index) for r in failed),
            metric_aggregates=aggregates,
        )

    def to_mapping(self) -> dict[str, Any]:
        """
        Return a JSON-friendly mapping of the study result.
        """
        return {
            "config": self.config.to_mapping(),
            "runner_config": _jsonable(_object_to_mapping(self.runner_config)),
            "metadata": self.metadata.to_mapping(),
            "summary": self.summary().to_mapping(),
            "runs": [r.to_mapping() for r in self.runs],
        }

    def save_summary_json(
        self,
        path: str | Path,
        *,
        sort_keys: bool = False,
        indent: int = 2,
    ) -> Path:
        """
        Save the study mapping to JSON.
        """
        return dump_config_mapping(
            self.to_mapping(),
            path,
            sort_keys=sort_keys,
            indent=indent,
        )

    def save_individual_run_archives(
        self,
        directory: str | Path,
        *,
        filename_template: str = "run_{index04d}.npz",
    ) -> tuple[Path, ...]:
        """
        Save retained per-run results as `.npz` archives.

        Only successful runs with a retained `result` object are written.
        """
        out_dir = Path(directory).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        for record in self.successful_runs():
            if record.result is None:
                continue

            run_id = record.result.metadata.run_id
            filename = _archive_filename_from_template(
                filename_template,
                run_index=record.run_index,
                run_id=run_id,
            )
            if not filename.endswith(".npz"):
                filename = f"{filename}.npz"

            path = out_dir / filename
            written_path = record.result.save_npz(path)
            record.archive_path = str(written_path)
            written.append(written_path)

        return tuple(written)


# -----------------------------------------------------------------------------
# Study configuration
# -----------------------------------------------------------------------------


@dataclass
class MonteCarloStudyConfig:
    """
    Study-level configuration for repeated simulation runs.

    Parameters
    ----------
    n_runs : int, default=100
        Number of independent Monte Carlo runs.
    root_seed : Any, default=12345
        Root seed source. The file passes this through the repository RNG helpers.
    parallel : bool, default=False
        Whether to use `ProcessPoolExecutor`.
    max_workers : int, optional
        Worker count for parallel mode.
    keep_full_results : bool, default=True
        Whether to retain the full `ScenarioSimulationResult` for each successful
        run in memory.
    save_run_archives : bool, default=False
        Whether to automatically save successful retained run results as `.npz`
        archives after the study finishes.
    output_directory : str or Path, optional
        Directory used when `save_run_archives=True`.
    archive_filename_template : str, default="run_{index04d}.npz"
        Filename template for per-run archives.
    run_id_prefix : str, default="mc"
        Prefix used when generating run IDs.
    stop_on_failure : bool, default=False
        If True, sequential mode re-raises immediately on the first failure.
        Parallel mode still records failures per run and completes submitted work.
    """

    n_runs: int = 100
    root_seed: Any = 12345
    parallel: bool = False
    max_workers: Optional[int] = None
    keep_full_results: bool = True
    save_run_archives: bool = False
    output_directory: Optional[str | Path] = None
    archive_filename_template: str = "run_{index04d}.npz"
    run_id_prefix: str = "mc"
    stop_on_failure: bool = False

    def __post_init__(self) -> None:
        self.n_runs = _positive_int(self.n_runs, name="n_runs")
        if self.max_workers is not None:
            self.max_workers = _positive_int(self.max_workers, name="max_workers")
        self.parallel = bool(self.parallel)
        self.keep_full_results = bool(self.keep_full_results)
        self.save_run_archives = bool(self.save_run_archives)
        self.run_id_prefix = str(self.run_id_prefix)
        self.stop_on_failure = bool(self.stop_on_failure)

        if self.save_run_archives and self.output_directory is None:
            raise ValueError(
                "output_directory must be provided when save_run_archives=True."
            )

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return _jsonable(asdict(self))


# -----------------------------------------------------------------------------
# Config / spec resolution helpers
# -----------------------------------------------------------------------------


def runner_config_from_mapping(mapping: Mapping[str, Any]) -> SimulationRunnerConfig:
    """
    Build `SimulationRunnerConfig` from a raw nested mapping.

    This helper handles nested dataclass blocks so the Monte Carlo config loader
    can accept natural YAML/JSON structures.
    """
    data = dict(mapping)

    if "process_noise" in data and isinstance(data["process_noise"], Mapping):
        data["process_noise"] = ErrorStateINSProcessNoise(**dict(data["process_noise"]))

    if "velocity_aid" in data and isinstance(data["velocity_aid"], Mapping):
        va = dict(data["velocity_aid"])
        if "schedule" in va and isinstance(va["schedule"], Mapping):
            va["schedule"] = PeriodicUpdateSchedule(**dict(va["schedule"]))
        data["velocity_aid"] = VelocityAidFusionConfig(**va)

    if "depth_aid" in data and isinstance(data["depth_aid"], Mapping):
        da = dict(data["depth_aid"])
        if "schedule" in da and isinstance(da["schedule"], Mapping):
            da["schedule"] = PeriodicUpdateSchedule(**dict(da["schedule"]))
        data["depth_aid"] = DepthFusionConfig(**da)

    if "map_match" in data and isinstance(data["map_match"], Mapping):
        mm = dict(data["map_match"])
        if "schedule" in mm and isinstance(mm["schedule"], Mapping):
            mm["schedule"] = PeriodicUpdateSchedule(**dict(mm["schedule"]))
        if "pf_spec" in mm and isinstance(mm["pf_spec"], Mapping):
            mm["pf_spec"] = MapMatchPFSpec(**dict(mm["pf_spec"]))
        data["map_match"] = MapMatchFeedbackConfig(**mm)

    if "integrity" in data and isinstance(data["integrity"], Mapping):
        data["integrity"] = IntegrityMonitorConfig(**dict(data["integrity"]))

    return SimulationRunnerConfig(**data)


def _resolve_mapping_or_config_reference(
    value: Mapping[str, Any] | str | Path,
    *,
    loader,
) -> dict[str, Any]:
    """
    Resolve either a raw mapping or a config reference with optional overrides.

    Accepted forms
    --------------
    1) raw mapping:
         {"gyro_noise_density_radps_per_sqrt_hz": ...}

    2) config reference with optional overrides:
         {"config": "imu_nav_grade", "overrides": {...}}
         {"path": "configs/sensors/imu_nav_grade.yaml", "overrides": {...}}
    """
    if isinstance(value, (str, Path)):
        return dict(loader(value))

    if not isinstance(value, Mapping):
        raise TypeError(
            f"Expected mapping/str/Path config input, got {type(value).__name__}."
        )

    if "config" in value or "path" in value:
        ref = value.get("config", value.get("path"))
        base = loader(ref)
        overrides = value.get("overrides")
        return apply_overrides(base, overrides)

    return dict(value)


def resolve_scenario_input(
    scenario_input: TruthTrajectory | ScenarioSpec | Mapping[str, Any] | str | Path,
) -> TruthTrajectory | ScenarioSpec | str | Path:
    """
    Resolve a scenario input into a form accepted by `ScenarioSimulationRunner`.

    Accepted inputs
    ---------------
    - TruthTrajectory
    - ScenarioSpec
    - config path / built-in scenario name
    - raw scenario mapping
    - {"config": "...", "overrides": {...}} mapping
    """
    if isinstance(scenario_input, (TruthTrajectory, ScenarioSpec, str, Path)):
        return scenario_input

    if not isinstance(scenario_input, Mapping):
        raise TypeError(
            "scenario_input must be TruthTrajectory, ScenarioSpec, mapping, str, or Path."
        )

    if "config" in scenario_input or "path" in scenario_input:
        merged = _resolve_mapping_or_config_reference(
            scenario_input,
            loader=load_scenario_mapping,
        )
        return ScenarioSpec.from_mapping(merged)

    keys = set(str(k) for k in scenario_input.keys())
    if keys <= {"name", "overrides"} and "name" in keys:
        base = load_scenario_mapping(str(scenario_input["name"]))
        merged = apply_overrides(base, scenario_input.get("overrides"))
        return ScenarioSpec.from_mapping(merged)

    return ScenarioSpec.from_mapping(dict(scenario_input))


def resolve_sensor_spec(
    spec_input: Any,
    *,
    spec_cls,
) -> Any:
    """
    Resolve a sensor spec from:
    - an already-instantiated spec object
    - a raw mapping
    - a sensor config path/name
    - a config-ref mapping with optional overrides
    """
    if spec_input is None:
        return None

    if isinstance(spec_input, spec_cls):
        return spec_input

    mapping = _resolve_mapping_or_config_reference(
        spec_input,
        loader=load_sensor_config,
    )
    return spec_cls(**mapping)


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------


def _execute_one_monte_carlo_run(
    *,
    run_index: int,
    run_seed: int,
    scenario_input: TruthTrajectory | ScenarioSpec | str | Path,
    runner_config: SimulationRunnerConfig,
    imu_spec: IMUSpec,
    gravimeter_spec: Optional[GravimeterSpec],
    depth_spec: Optional[DepthSensorSpec],
    velocity_aid_spec: Optional[VelocityAidSpec],
    map_model: Any | str | Path | None,
    base_metadata: Optional[SimulationMetadata],
    keep_full_result: bool,
    dt_s: Optional[float],
    run_id_prefix: str,
) -> MonteCarloRunRecord:
    """
    Execute one Monte Carlo run and return a `MonteCarloRunRecord`.

    This function is module-top-level so it remains picklable for
    `ProcessPoolExecutor`.
    """
    try:
        runner = ScenarioSimulationRunner(config=runner_config)

        metadata = SimulationMetadata() if base_metadata is None else base_metadata.copy()
        if metadata.run_id is None:
            metadata.run_id = f"{run_id_prefix}_{int(run_index):04d}"
        else:
            metadata.run_id = f"{metadata.run_id}_{int(run_index):04d}"

        metadata.extra.update(
            {
                "run_index": int(run_index),
                "run_seed": int(run_seed),
            }
        )
        metadata.rng_state = {"seed": int(run_seed)}

        result = runner.run_with_specs(
            scenario_input,
            imu_spec=imu_spec,
            gravimeter_spec=gravimeter_spec,
            depth_spec=depth_spec,
            velocity_aid_spec=velocity_aid_spec,
            map_model=map_model,
            metadata=metadata,
            dt_s=dt_s,
            seed=int(run_seed),
        )
        metrics = scenario_metrics_from_result(result)

        return MonteCarloRunRecord(
            run_index=int(run_index),
            run_seed=int(run_seed),
            success=True,
            result=result if keep_full_result else None,
            metrics=metrics,
        )

    except Exception as exc:
        return MonteCarloRunRecord(
            run_index=int(run_index),
            run_seed=int(run_seed),
            success=False,
            result=None,
            metrics=None,
            error_message=f"{type(exc).__name__}: {exc}",
            traceback_text=traceback.format_exc(),
        )


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------


class MonteCarloStudyRunner:
    """
    Orchestrate repeated scenario simulations and aggregate their metrics.

    Parameters
    ----------
    study_config : MonteCarloStudyConfig
        Study-level execution policy.
    runner_config : SimulationRunnerConfig, optional
        Per-run simulation runner configuration.
    """

    def __init__(
        self,
        study_config: Optional[MonteCarloStudyConfig] = None,
        *,
        runner_config: Optional[SimulationRunnerConfig] = None,
    ) -> None:
        self.study_config = (
            MonteCarloStudyConfig() if study_config is None else study_config
        )
        self.runner_config = (
            SimulationRunnerConfig() if runner_config is None else runner_config
        )

    def _run_seeds(self) -> list[int]:
        """
        Create one integer seed per Monte Carlo run from the configured root seed.
        """
        generators = monte_carlo_generators(
            self.study_config.root_seed,
            self.study_config.n_runs,
        )
        return [int(random_uint64(g)) for g in generators]

    def run(
        self,
        scenario_input: TruthTrajectory | ScenarioSpec | Mapping[str, Any] | str | Path,
        *,
        imu_spec: IMUSpec,
        gravimeter_spec: Optional[GravimeterSpec] = None,
        depth_spec: Optional[DepthSensorSpec] = None,
        velocity_aid_spec: Optional[VelocityAidSpec] = None,
        map_model: Any | str | Path | None = None,
        metadata: Optional[SimulationMetadata] = None,
        dt_s: Optional[float] = None,
    ) -> MonteCarloStudyResult:
        """
        Run a full Monte Carlo study.

        Parameters
        ----------
        scenario_input : TruthTrajectory, ScenarioSpec, mapping, str, or Path
            Scenario source accepted by `resolve_scenario_input(...)`.
        imu_spec, gravimeter_spec, depth_spec, velocity_aid_spec
            Sensor specifications for each run.
        map_model : object, path, or None, optional
            Gravity-map backend or path to a saved map.
        metadata : SimulationMetadata, optional
            Base metadata copied into each run.
        dt_s : float, optional
            Optional scenario resampling interval when the input is a scenario spec.

        Returns
        -------
        MonteCarloStudyResult
            Full study result with per-run records.
        """
        scenario_resolved = resolve_scenario_input(scenario_input)
        seeds = self._run_seeds()

        base_metadata = SimulationMetadata() if metadata is None else metadata.copy()
        base_metadata.extra.update(
            {
                "monte_carlo_root_seed": _jsonable(seed_record_from_seed(self.study_config.root_seed)),
                "monte_carlo_n_runs": int(self.study_config.n_runs),
            }
        )

        worker_kwargs = [
            dict(
                run_index=k,
                run_seed=seeds[k],
                scenario_input=scenario_resolved,
                runner_config=self.runner_config,
                imu_spec=imu_spec,
                gravimeter_spec=gravimeter_spec,
                depth_spec=depth_spec,
                velocity_aid_spec=velocity_aid_spec,
                map_model=map_model,
                base_metadata=base_metadata,
                keep_full_result=self.study_config.keep_full_results,
                dt_s=dt_s,
                run_id_prefix=self.study_config.run_id_prefix,
            )
            for k in range(self.study_config.n_runs)
        ]

        records_by_index: dict[int, MonteCarloRunRecord] = {}

        if self.study_config.parallel and self.study_config.n_runs > 1:
            with ProcessPoolExecutor(
                max_workers=self.study_config.max_workers
            ) as executor:
                futures = {
                    executor.submit(_execute_one_monte_carlo_run, **kwargs): kwargs["run_index"]
                    for kwargs in worker_kwargs
                }
                for future in as_completed(futures):
                    record = future.result()
                    records_by_index[int(record.run_index)] = record
        else:
            for kwargs in worker_kwargs:
                record = _execute_one_monte_carlo_run(**kwargs)
                records_by_index[int(record.run_index)] = record

                if self.study_config.stop_on_failure and not record.success:
                    raise RuntimeError(
                        f"Monte Carlo run {record.run_index} failed: {record.error_message}"
                    )

        ordered_records = [
            records_by_index[k] for k in sorted(records_by_index.keys())
        ]

        study_result = MonteCarloStudyResult(
            config=self.study_config,
            runner_config=self.runner_config,
            metadata=base_metadata,
            runs=ordered_records,
        )

        if self.study_config.save_run_archives:
            study_result.save_individual_run_archives(
                self.study_config.output_directory,  # type: ignore[arg-type]
                filename_template=self.study_config.archive_filename_template,
            )

        return study_result


# -----------------------------------------------------------------------------
# Config-driven convenience functions
# -----------------------------------------------------------------------------


def run_monte_carlo_study(
    scenario_input: TruthTrajectory | ScenarioSpec | Mapping[str, Any] | str | Path,
    *,
    imu_spec: IMUSpec,
    gravimeter_spec: Optional[GravimeterSpec] = None,
    depth_spec: Optional[DepthSensorSpec] = None,
    velocity_aid_spec: Optional[VelocityAidSpec] = None,
    map_model: Any | str | Path | None = None,
    study_config: Optional[MonteCarloStudyConfig] = None,
    runner_config: Optional[SimulationRunnerConfig] = None,
    metadata: Optional[SimulationMetadata] = None,
    dt_s: Optional[float] = None,
) -> MonteCarloStudyResult:
    """
    Functional convenience wrapper around `MonteCarloStudyRunner`.
    """
    runner = MonteCarloStudyRunner(
        study_config=study_config,
        runner_config=runner_config,
    )
    return runner.run(
        scenario_input,
        imu_spec=imu_spec,
        gravimeter_spec=gravimeter_spec,
        depth_spec=depth_spec,
        velocity_aid_spec=velocity_aid_spec,
        map_model=map_model,
        metadata=metadata,
        dt_s=dt_s,
    )


def run_monte_carlo_from_mapping(
    mapping: Mapping[str, Any],
    *,
    map_model: Any | str | Path | None = None,
) -> MonteCarloStudyResult:
    """
    Run a Monte Carlo study from a raw configuration mapping.

    Expected structure
    ------------------
    A practical configuration looks like:

        {
          "monte_carlo": {...study-level settings...},
          "runner": {...SimulationRunnerConfig fields...},
          "scenario": "maritime_baseline" | {...scenario mapping...},
          "sensors": {
              "imu": "imu_nav_grade" | {...},
              "gravimeter": "gravimeter_proto" | {...} | None,
              "depth": "depth_sensor" | {...} | None,
              "velocity_aid": "velocity_aid" | {...} | None,
          },
          "metadata": {...SimulationMetadata fields...},
          "dt_s": 1.0,
        }

    Notes
    -----
    - The `"monte_carlo"` block is optional; if omitted, the root mapping itself
      is interpreted as the study-config block where compatible.
    - Sensor entries may be:
      - already-instantiated spec objects,
      - raw mappings,
      - strings/paths resolved via `configs/sensors`,
      - or `{config: ..., overrides: ...}` mappings.
    """
    root = dict(mapping)

    study_block = root.get("monte_carlo", {})
    if not isinstance(study_block, Mapping):
        raise TypeError("`monte_carlo` block must be a mapping when provided.")

    runner_block = root.get("runner", {})
    if not isinstance(runner_block, Mapping):
        raise TypeError("`runner` block must be a mapping when provided.")

    sensors_block = root.get("sensors", {})
    if not isinstance(sensors_block, Mapping):
        raise TypeError("`sensors` block must be a mapping when provided.")

    if "scenario" not in root:
        raise KeyError("Monte Carlo config mapping must contain a `scenario` entry.")

    scenario_input = resolve_scenario_input(root["scenario"])

    study_config = MonteCarloStudyConfig(**dict(study_block))
    runner_config = runner_config_from_mapping(runner_block)

    imu_spec = resolve_sensor_spec(sensors_block.get("imu"), spec_cls=IMUSpec)
    if imu_spec is None:
        raise KeyError("`sensors.imu` must be provided in a Monte Carlo config.")

    gravimeter_spec = resolve_sensor_spec(
        sensors_block.get("gravimeter"),
        spec_cls=GravimeterSpec,
    )
    depth_spec = resolve_sensor_spec(
        sensors_block.get("depth"),
        spec_cls=DepthSensorSpec,
    )
    velocity_aid_spec = resolve_sensor_spec(
        sensors_block.get("velocity_aid"),
        spec_cls=VelocityAidSpec,
    )

    metadata_block = root.get("metadata", {})
    if metadata_block is None:
        metadata = None
    elif isinstance(metadata_block, SimulationMetadata):
        metadata = metadata_block
    elif isinstance(metadata_block, Mapping):
        metadata = SimulationMetadata(**dict(metadata_block))
    else:
        raise TypeError("`metadata` must be a mapping, SimulationMetadata, or None.")

    dt_s = root.get("dt_s", None)
    if dt_s is not None:
        dt_s = float(dt_s)

    return run_monte_carlo_study(
        scenario_input,
        imu_spec=imu_spec,
        gravimeter_spec=gravimeter_spec,
        depth_spec=depth_spec,
        velocity_aid_spec=velocity_aid_spec,
        map_model=map_model,
        study_config=study_config,
        runner_config=runner_config,
        metadata=metadata,
        dt_s=dt_s,
    )


def run_monte_carlo_from_config(
    path_or_name: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    map_model: Any | str | Path | None = None,
) -> MonteCarloStudyResult:
    """
    Load a Monte Carlo config from `configs/monte_carlo` (or an explicit path)
    and run the study.

    Parameters
    ----------
    path_or_name : str or Path
        Config file path or bare config stem.
    overrides : mapping, optional
        Recursive overrides applied on top of the loaded config.
    map_model : object, path, or None, optional
        Optional explicit gravity-map override.

    Returns
    -------
    MonteCarloStudyResult
        Completed study result.
    """
    base = load_monte_carlo_config(path_or_name)
    merged = apply_overrides(base, overrides)
    return run_monte_carlo_from_mapping(
        merged,
        map_model=map_model,
    )


__all__ = [
    "FloatArray",
    "MonteCarloRunRecord",
    "MonteCarloScalarAggregate",
    "MonteCarloStudyConfig",
    "MonteCarloStudyResult",
    "MonteCarloStudyRunner",
    "MonteCarloStudySummary",
    "resolve_scenario_input",
    "resolve_sensor_spec",
    "run_monte_carlo_from_config",
    "run_monte_carlo_from_mapping",
    "run_monte_carlo_study",
    "runner_config_from_mapping",
]