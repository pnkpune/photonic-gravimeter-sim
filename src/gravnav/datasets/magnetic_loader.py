"""
magnetic_loader.py

Regional scalar magnetic-field and anomaly grid ingestion helpers.

The runtime stack uses this map only as an optional supporting passive aid.
The default public-facing use is scalar total-field or total-field-anomaly
matching over ocean regions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..physics.gravity_map import bilinear_interpolate_rectilinear
from ..utils.config import find_project_root

FloatArray = NDArray[np.float64]


def _project_root(project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).expanduser().resolve()
    return find_project_root()


def _as_float_array(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def _vec1_strictly_increasing(x: ArrayLike, *, name: str) -> FloatArray:
    arr = _as_float_array(x).reshape(-1)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be one-dimensional with at least 2 values.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.all(np.diff(arr) > 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return arr


def _grid2(x: ArrayLike, *, shape: tuple[int, int], name: str) -> FloatArray:
    arr = _as_float_array(x)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _relative_to_root(path: Path, *, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path.resolve())


def _resolve_manifest_ref(path_ref: str | Path, *, manifest_path: Path) -> Path:
    ref = Path(path_ref).expanduser()
    if ref.is_absolute():
        return ref.resolve()
    candidate = (manifest_path.parent / ref).resolve()
    if candidate.exists():
        return candidate
    project_root = _project_root()
    root_candidate = (project_root / ref).resolve()
    if root_candidate.exists():
        return root_candidate
    return candidate


@dataclass(frozen=True)
class MagneticGrid:
    """
    Regular lat/lon scalar magnetic map.
    """

    lat_axis_deg: FloatArray
    lon_axis_deg: FloatArray
    total_field_grid_nt: FloatArray
    anomaly_grid_nt: FloatArray
    reference_height_m: float = 0.0
    default_method: str = "linear"
    bounds_error: bool = False
    fill_value_nt: float = np.nan
    name: str = "magnetic_grid"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lat_axis_deg = _vec1_strictly_increasing(self.lat_axis_deg, name="lat_axis_deg")
        lon_axis_deg = _vec1_strictly_increasing(self.lon_axis_deg, name="lon_axis_deg")
        shape = (lat_axis_deg.size, lon_axis_deg.size)
        total_field_grid_nt = _grid2(
            self.total_field_grid_nt,
            shape=shape,
            name="total_field_grid_nt",
        )
        anomaly_grid_nt = _grid2(
            self.anomaly_grid_nt,
            shape=shape,
            name="anomaly_grid_nt",
        )
        object.__setattr__(self, "lat_axis_deg", lat_axis_deg)
        object.__setattr__(self, "lon_axis_deg", lon_axis_deg)
        object.__setattr__(self, "total_field_grid_nt", total_field_grid_nt)
        object.__setattr__(self, "anomaly_grid_nt", anomaly_grid_nt)
        object.__setattr__(self, "reference_height_m", float(self.reference_height_m))
        object.__setattr__(self, "default_method", str(self.default_method))
        object.__setattr__(self, "bounds_error", bool(self.bounds_error))
        object.__setattr__(self, "fill_value_nt", float(self.fill_value_nt))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.lat_axis_deg.size), int(self.lon_axis_deg.size))

    def contains(self, lat_deg: ArrayLike, lon_deg: ArrayLike) -> NDArray[np.bool_]:
        lat = _as_float_array(lat_deg)
        lon = _as_float_array(lon_deg)
        lat_b, lon_b = np.broadcast_arrays(lat, lon)
        return (
            (lat_b >= self.lat_axis_deg[0])
            & (lat_b <= self.lat_axis_deg[-1])
            & (lon_b >= self.lon_axis_deg[0])
            & (lon_b <= self.lon_axis_deg[-1])
        )

    def _evaluate_grid(self, grid: FloatArray, lat_deg: ArrayLike, lon_deg: ArrayLike):
        return bilinear_interpolate_rectilinear(
            self.lat_axis_deg,
            self.lon_axis_deg,
            grid,
            lat_deg,
            lon_deg,
            bounds_error=self.bounds_error,
            fill_value=self.fill_value_nt,
            method=self.default_method,
        )

    def evaluate_total_field_nt(self, lat_deg: ArrayLike, lon_deg: ArrayLike):
        return self._evaluate_grid(self.total_field_grid_nt, lat_deg, lon_deg)

    def evaluate_anomaly_nt(self, lat_deg: ArrayLike, lon_deg: ArrayLike):
        return self._evaluate_grid(self.anomaly_grid_nt, lat_deg, lon_deg)

    def evaluate_horizontal_gradient_nt_per_m(
        self,
        lat_deg: ArrayLike,
        lon_deg: ArrayLike,
        *,
        delta_lat_deg: Optional[float] = None,
        delta_lon_deg: Optional[float] = None,
    ) -> FloatArray:
        lat = np.asarray(lat_deg, dtype=np.float64)
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat_b, lon_b = np.broadcast_arrays(lat, lon)
        lat_step = (
            float(delta_lat_deg)
            if delta_lat_deg is not None
            else max(1.0e-4, 0.5 * float(np.mean(np.diff(self.lat_axis_deg))))
        )
        lon_step = (
            float(delta_lon_deg)
            if delta_lon_deg is not None
            else max(1.0e-4, 0.5 * float(np.mean(np.diff(self.lon_axis_deg))))
        )
        north_m_per_deg = 111_320.0
        east_m_per_deg = north_m_per_deg * np.cos(np.deg2rad(lat_b))
        north_m_per_deg = np.maximum(north_m_per_deg, 1.0)
        east_m_per_deg = np.maximum(east_m_per_deg, 1.0)
        plus_n = np.asarray(self.evaluate_total_field_nt(lat_b + lat_step, lon_b), dtype=np.float64)
        minus_n = np.asarray(self.evaluate_total_field_nt(lat_b - lat_step, lon_b), dtype=np.float64)
        plus_e = np.asarray(self.evaluate_total_field_nt(lat_b, lon_b + lon_step), dtype=np.float64)
        minus_e = np.asarray(self.evaluate_total_field_nt(lat_b, lon_b - lon_step), dtype=np.float64)
        d_dn = (plus_n - minus_n) / (2.0 * lat_step * north_m_per_deg)
        d_de = (plus_e - minus_e) / (2.0 * lon_step * east_m_per_deg)
        out = np.stack([d_dn, d_de], axis=-1)
        return np.asarray(out, dtype=np.float64)

    def to_npz(self, path: str | Path) -> Path:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            lat_axis_deg=np.asarray(self.lat_axis_deg, dtype=np.float64),
            lon_axis_deg=np.asarray(self.lon_axis_deg, dtype=np.float64),
            total_field_grid_nt=np.asarray(self.total_field_grid_nt, dtype=np.float64),
            anomaly_grid_nt=np.asarray(self.anomaly_grid_nt, dtype=np.float64),
            reference_height_m=np.array(float(self.reference_height_m), dtype=np.float64),
            default_method=np.array(str(self.default_method)),
            bounds_error=np.array(bool(self.bounds_error)),
            fill_value_nt=np.array(float(self.fill_value_nt), dtype=np.float64),
            name=np.array(str(self.name)),
            metadata_json=np.array(json.dumps(_jsonable(self.metadata))),
        )
        return p

    @classmethod
    def from_npz(cls, path: str | Path) -> "MagneticGrid":
        p = Path(path).expanduser().resolve()
        with np.load(p, allow_pickle=False) as data:
            return cls(
                lat_axis_deg=np.asarray(data["lat_axis_deg"], dtype=np.float64),
                lon_axis_deg=np.asarray(data["lon_axis_deg"], dtype=np.float64),
                total_field_grid_nt=np.asarray(data["total_field_grid_nt"], dtype=np.float64),
                anomaly_grid_nt=np.asarray(data["anomaly_grid_nt"], dtype=np.float64),
                reference_height_m=float(data["reference_height_m"]),
                default_method=str(np.asarray(data["default_method"]).item()),
                bounds_error=bool(np.asarray(data["bounds_error"]).item()),
                fill_value_nt=float(data["fill_value_nt"]),
                name=str(np.asarray(data["name"]).item()),
                metadata=json.loads(str(np.asarray(data["metadata_json"]).item())),
            )


@dataclass(frozen=True)
class RegionalMagneticMapManifest:
    region_name: str
    source_name: str
    source_kind: str
    raw_data_path: str
    processed_map_path: str
    manifest_path: str
    field_units: str
    anomaly_units: str
    reference_height_m: float
    lat_bounds_deg: tuple[float, float]
    lon_bounds_deg: tuple[float, float]
    spacing_deg: tuple[float, float]
    shape: tuple[int, int]
    interpolation: str
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "RegionalMagneticMapManifest":
        return cls(
            region_name=str(mapping["region_name"]),
            source_name=str(mapping["source_name"]),
            source_kind=str(mapping["source_kind"]),
            raw_data_path=str(mapping["raw_data_path"]),
            processed_map_path=str(mapping["processed_map_path"]),
            manifest_path=str(mapping["manifest_path"]),
            field_units=str(mapping["field_units"]),
            anomaly_units=str(mapping["anomaly_units"]),
            reference_height_m=float(mapping["reference_height_m"]),
            lat_bounds_deg=tuple(float(x) for x in mapping["lat_bounds_deg"]),
            lon_bounds_deg=tuple(float(x) for x in mapping["lon_bounds_deg"]),
            spacing_deg=tuple(float(x) for x in mapping["spacing_deg"]),
            shape=tuple(int(x) for x in mapping["shape"]),
            interpolation=str(mapping["interpolation"]),
            notes=[str(x) for x in mapping.get("notes", [])],
            metadata=dict(mapping.get("metadata", {})),
        )

    def write_json(self, path: str | Path) -> Path:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_mapping(), indent=2) + "\n", encoding="utf-8")
        return p


def load_magnetic_manifest(path: str | Path) -> RegionalMagneticMapManifest:
    p = Path(path).expanduser().resolve()
    return RegionalMagneticMapManifest.from_mapping(json.loads(p.read_text(encoding="utf-8")))


def load_magnetic_grid_from_manifest(
    path: str | Path,
) -> tuple[MagneticGrid, RegionalMagneticMapManifest, Path, Path]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = load_magnetic_manifest(manifest_path)
    processed_map_path = _resolve_manifest_ref(
        manifest.processed_map_path,
        manifest_path=manifest_path,
    )
    return (
        MagneticGrid.from_npz(processed_map_path),
        manifest,
        processed_map_path,
        manifest_path,
    )


def _validate_regular_grid(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    total_field_nt: np.ndarray,
    anomaly_nt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat_axis = np.unique(lat_deg)
    lon_axis = np.unique(lon_deg)
    expected_size = lat_axis.size * lon_axis.size
    if lat_deg.size != expected_size:
        raise ValueError(
            "Regular-grid CSV is incomplete or contains duplicates: "
            f"expected {expected_size} rows, got {lat_deg.size}."
        )
    row_lookup = {(float(la), float(lo)): idx for idx, (la, lo) in enumerate(zip(lat_deg, lon_deg))}
    if len(row_lookup) != lat_deg.size:
        raise ValueError("Regular-grid CSV contains duplicate lat/lon samples.")

    total_grid = np.empty((lat_axis.size, lon_axis.size), dtype=np.float64)
    anomaly_grid = np.empty((lat_axis.size, lon_axis.size), dtype=np.float64)
    for i, lat_val in enumerate(lat_axis):
        for j, lon_val in enumerate(lon_axis):
            idx = row_lookup[(float(lat_val), float(lon_val))]
            total_grid[i, j] = float(total_field_nt[idx])
            anomaly_grid[i, j] = float(anomaly_nt[idx])
    return lat_axis, lon_axis, total_grid, anomaly_grid


def load_regular_csv_magnetic_grid(
    raw_csv_path: str | Path,
    *,
    name: str,
    region_name: str,
    source_name: str,
    reference_height_m: float = 0.0,
    default_method: str = "linear",
    bounds_error: bool = False,
    fill_value_nt: float = np.nan,
    metadata_extra: Optional[dict[str, Any]] = None,
) -> MagneticGrid:
    p = Path(raw_csv_path).expanduser().resolve()
    rows: list[dict[str, str]] = []
    with p.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"lat_deg", "lon_deg", "total_field_nt", "anomaly_nt"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"Missing required CSV columns {sorted(missing)} in {p.name}.")
        rows.extend(row for row in reader if row)
    if not rows:
        raise ValueError(f"CSV magnetic grid {p} is empty.")

    lat_deg = np.asarray([float(row["lat_deg"]) for row in rows], dtype=np.float64)
    lon_deg = np.asarray([float(row["lon_deg"]) for row in rows], dtype=np.float64)
    total_field_nt = np.asarray([float(row["total_field_nt"]) for row in rows], dtype=np.float64)
    anomaly_nt = np.asarray([float(row["anomaly_nt"]) for row in rows], dtype=np.float64)
    lat_axis, lon_axis, total_grid, anomaly_grid = _validate_regular_grid(
        lat_deg,
        lon_deg,
        total_field_nt,
        anomaly_nt,
    )
    metadata = {
        "region_name": region_name,
        "source_name": source_name,
        "source_kind": "regular_grid_csv",
        "raw_data_path": str(p),
    }
    if metadata_extra:
        metadata.update(dict(metadata_extra))
    return MagneticGrid(
        lat_axis_deg=lat_axis,
        lon_axis_deg=lon_axis,
        total_field_grid_nt=total_grid,
        anomaly_grid_nt=anomaly_grid,
        reference_height_m=reference_height_m,
        default_method=default_method,
        bounds_error=bounds_error,
        fill_value_nt=fill_value_nt,
        name=name,
        metadata=metadata,
    )


def process_regular_csv_magnetic_grid(
    raw_csv_path: str | Path,
    *,
    region_name: str,
    source_name: str,
    processed_map_path: str | Path,
    manifest_path: str | Path,
    project_root: str | Path | None = None,
    reference_height_m: float = 0.0,
    default_method: str = "linear",
    bounds_error: bool = False,
    fill_value_nt: float = np.nan,
    notes: Optional[list[str]] = None,
    metadata_extra: Optional[dict[str, Any]] = None,
) -> tuple[MagneticGrid, RegionalMagneticMapManifest]:
    root = _project_root(project_root)
    raw_path = Path(raw_csv_path).expanduser().resolve()
    processed_path = Path(processed_map_path).expanduser().resolve()
    manifest_path_resolved = Path(manifest_path).expanduser().resolve()

    grid = load_regular_csv_magnetic_grid(
        raw_path,
        name=f"{region_name}_magnetic_grid",
        region_name=region_name,
        source_name=source_name,
        reference_height_m=reference_height_m,
        default_method=default_method,
        bounds_error=bounds_error,
        fill_value_nt=fill_value_nt,
        metadata_extra=metadata_extra,
    )
    grid.to_npz(processed_path)
    manifest = RegionalMagneticMapManifest(
        region_name=region_name,
        source_name=source_name,
        source_kind="regular_grid_csv",
        raw_data_path=_relative_to_root(raw_path, project_root=root),
        processed_map_path=_relative_to_root(processed_path, project_root=root),
        manifest_path=_relative_to_root(manifest_path_resolved, project_root=root),
        field_units="nT",
        anomaly_units="nT",
        reference_height_m=float(reference_height_m),
        lat_bounds_deg=(float(grid.lat_axis_deg[0]), float(grid.lat_axis_deg[-1])),
        lon_bounds_deg=(float(grid.lon_axis_deg[0]), float(grid.lon_axis_deg[-1])),
        spacing_deg=(
            float(np.mean(np.diff(grid.lat_axis_deg))),
            float(np.mean(np.diff(grid.lon_axis_deg))),
        ),
        shape=grid.shape,
        interpolation=str(default_method),
        notes=[] if notes is None else [str(x) for x in notes],
        metadata={
            "source_name": source_name,
            "source_kind": "regular_grid_csv",
            **({} if metadata_extra is None else dict(metadata_extra)),
        },
    )
    manifest.write_json(manifest_path_resolved)
    return grid, manifest


__all__ = [
    "MagneticGrid",
    "RegionalMagneticMapManifest",
    "load_magnetic_grid_from_manifest",
    "load_magnetic_manifest",
    "load_regular_csv_magnetic_grid",
    "process_regular_csv_magnetic_grid",
]
