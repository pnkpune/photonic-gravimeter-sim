"""
current_loader.py

Regional ocean-current field ingestion and cache helpers.

The first implementation is static in time and depth-aware. It is intended for
process-correction and water-track velocity correction rather than direct global
matching.
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
    if arr.ndim != 1 or arr.size < 1:
        raise ValueError(f"{name} must be one-dimensional with at least 1 value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if arr.size > 1 and not np.all(np.diff(arr) > 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return arr


def _grid3(x: ArrayLike, *, shape: tuple[int, int, int], name: str) -> FloatArray:
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
class CurrentFieldGrid:
    lat_axis_deg: FloatArray
    lon_axis_deg: FloatArray
    depth_axis_m: FloatArray
    north_current_grid_mps: FloatArray
    east_current_grid_mps: FloatArray
    name: str = "current_field_grid"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lat_axis_deg = _vec1_strictly_increasing(self.lat_axis_deg, name="lat_axis_deg")
        lon_axis_deg = _vec1_strictly_increasing(self.lon_axis_deg, name="lon_axis_deg")
        depth_axis_m = _vec1_strictly_increasing(self.depth_axis_m, name="depth_axis_m")
        shape = (lat_axis_deg.size, lon_axis_deg.size, depth_axis_m.size)
        object.__setattr__(self, "lat_axis_deg", lat_axis_deg)
        object.__setattr__(self, "lon_axis_deg", lon_axis_deg)
        object.__setattr__(self, "depth_axis_m", depth_axis_m)
        object.__setattr__(
            self,
            "north_current_grid_mps",
            _grid3(self.north_current_grid_mps, shape=shape, name="north_current_grid_mps"),
        )
        object.__setattr__(
            self,
            "east_current_grid_mps",
            _grid3(self.east_current_grid_mps, shape=shape, name="east_current_grid_mps"),
        )
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def _evaluate_depth_slice(self, grid: FloatArray, lat_deg: ArrayLike, lon_deg: ArrayLike, depth_m: ArrayLike):
        lat = np.asarray(lat_deg, dtype=np.float64)
        lon = np.asarray(lon_deg, dtype=np.float64)
        depth = np.asarray(depth_m, dtype=np.float64)
        lat_b, lon_b, depth_b = np.broadcast_arrays(lat, lon, depth)
        if self.depth_axis_m.size == 1:
            return bilinear_interpolate_rectilinear(
                self.lat_axis_deg,
                self.lon_axis_deg,
                grid[:, :, 0],
                lat_b,
                lon_b,
                bounds_error=False,
                fill_value=np.nan,
                method="linear",
            )
        out = np.empty(lat_b.shape, dtype=np.float64)
        flat_lat = lat_b.reshape(-1)
        flat_lon = lon_b.reshape(-1)
        flat_depth = depth_b.reshape(-1)
        for idx, depth_val in enumerate(flat_depth):
            if depth_val <= self.depth_axis_m[0]:
                low = high = 0
                weight_high = 0.0
            elif depth_val >= self.depth_axis_m[-1]:
                low = high = self.depth_axis_m.size - 1
                weight_high = 0.0
            else:
                high = int(np.searchsorted(self.depth_axis_m, depth_val, side="right"))
                low = high - 1
                denom = max(float(self.depth_axis_m[high] - self.depth_axis_m[low]), 1.0e-9)
                weight_high = float((depth_val - self.depth_axis_m[low]) / denom)
            low_val = float(
                bilinear_interpolate_rectilinear(
                    self.lat_axis_deg,
                    self.lon_axis_deg,
                    grid[:, :, low],
                    flat_lat[idx],
                    flat_lon[idx],
                    bounds_error=False,
                    fill_value=np.nan,
                    method="linear",
                )
            )
            if high == low:
                out.reshape(-1)[idx] = low_val
            else:
                high_val = float(
                    bilinear_interpolate_rectilinear(
                        self.lat_axis_deg,
                        self.lon_axis_deg,
                        grid[:, :, high],
                        flat_lat[idx],
                        flat_lon[idx],
                        bounds_error=False,
                        fill_value=np.nan,
                        method="linear",
                    )
                )
                out.reshape(-1)[idx] = (1.0 - weight_high) * low_val + weight_high * high_val
        return out

    def evaluate_current_ned_mps(
        self,
        lat_deg: ArrayLike,
        lon_deg: ArrayLike,
        depth_m: ArrayLike,
    ) -> FloatArray:
        north = np.asarray(
            self._evaluate_depth_slice(self.north_current_grid_mps, lat_deg, lon_deg, depth_m),
            dtype=np.float64,
        )
        east = np.asarray(
            self._evaluate_depth_slice(self.east_current_grid_mps, lat_deg, lon_deg, depth_m),
            dtype=np.float64,
        )
        out = np.stack([north, east, np.zeros_like(north)], axis=-1)
        return np.asarray(out, dtype=np.float64)

    def to_npz(self, path: str | Path) -> Path:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            lat_axis_deg=np.asarray(self.lat_axis_deg, dtype=np.float64),
            lon_axis_deg=np.asarray(self.lon_axis_deg, dtype=np.float64),
            depth_axis_m=np.asarray(self.depth_axis_m, dtype=np.float64),
            north_current_grid_mps=np.asarray(self.north_current_grid_mps, dtype=np.float64),
            east_current_grid_mps=np.asarray(self.east_current_grid_mps, dtype=np.float64),
            name=np.array(str(self.name)),
            metadata_json=np.array(json.dumps(_jsonable(self.metadata))),
        )
        return p

    @classmethod
    def from_npz(cls, path: str | Path) -> "CurrentFieldGrid":
        p = Path(path).expanduser().resolve()
        with np.load(p, allow_pickle=False) as data:
            return cls(
                lat_axis_deg=np.asarray(data["lat_axis_deg"], dtype=np.float64),
                lon_axis_deg=np.asarray(data["lon_axis_deg"], dtype=np.float64),
                depth_axis_m=np.asarray(data["depth_axis_m"], dtype=np.float64),
                north_current_grid_mps=np.asarray(data["north_current_grid_mps"], dtype=np.float64),
                east_current_grid_mps=np.asarray(data["east_current_grid_mps"], dtype=np.float64),
                name=str(np.asarray(data["name"]).item()),
                metadata=json.loads(str(np.asarray(data["metadata_json"]).item())),
            )


@dataclass(frozen=True)
class RegionalCurrentFieldManifest:
    region_name: str
    source_name: str
    source_kind: str
    raw_data_path: str
    processed_grid_path: str
    manifest_path: str
    current_units: str
    lat_bounds_deg: tuple[float, float]
    lon_bounds_deg: tuple[float, float]
    depth_bounds_m: tuple[float, float]
    spacing_deg: tuple[float, float]
    depth_spacing_m: float
    shape: tuple[int, int, int]
    interpolation: str
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "RegionalCurrentFieldManifest":
        return cls(
            region_name=str(mapping["region_name"]),
            source_name=str(mapping["source_name"]),
            source_kind=str(mapping["source_kind"]),
            raw_data_path=str(mapping["raw_data_path"]),
            processed_grid_path=str(mapping["processed_grid_path"]),
            manifest_path=str(mapping["manifest_path"]),
            current_units=str(mapping["current_units"]),
            lat_bounds_deg=tuple(float(x) for x in mapping["lat_bounds_deg"]),
            lon_bounds_deg=tuple(float(x) for x in mapping["lon_bounds_deg"]),
            depth_bounds_m=tuple(float(x) for x in mapping["depth_bounds_m"]),
            spacing_deg=tuple(float(x) for x in mapping["spacing_deg"]),
            depth_spacing_m=float(mapping["depth_spacing_m"]),
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


def load_current_manifest(path: str | Path) -> RegionalCurrentFieldManifest:
    p = Path(path).expanduser().resolve()
    return RegionalCurrentFieldManifest.from_mapping(json.loads(p.read_text(encoding="utf-8")))


def load_current_field_from_manifest(
    path: str | Path,
) -> tuple[CurrentFieldGrid, RegionalCurrentFieldManifest, Path, Path]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = load_current_manifest(manifest_path)
    processed_grid_path = _resolve_manifest_ref(
        manifest.processed_grid_path,
        manifest_path=manifest_path,
    )
    return (
        CurrentFieldGrid.from_npz(processed_grid_path),
        manifest,
        processed_grid_path,
        manifest_path,
    )


def _validate_regular_grid(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    depth_m: np.ndarray,
    north_mps: np.ndarray,
    east_mps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat_axis = np.unique(lat_deg)
    lon_axis = np.unique(lon_deg)
    depth_axis = np.unique(depth_m)
    expected_size = lat_axis.size * lon_axis.size * depth_axis.size
    if lat_deg.size != expected_size:
        raise ValueError(
            "Regular-grid CSV is incomplete or contains duplicates: "
            f"expected {expected_size} rows, got {lat_deg.size}."
        )
    row_lookup = {
        (float(la), float(lo), float(de)): idx
        for idx, (la, lo, de) in enumerate(zip(lat_deg, lon_deg, depth_m))
    }
    if len(row_lookup) != lat_deg.size:
        raise ValueError("Regular-grid CSV contains duplicate samples.")
    north_grid = np.empty((lat_axis.size, lon_axis.size, depth_axis.size), dtype=np.float64)
    east_grid = np.empty((lat_axis.size, lon_axis.size, depth_axis.size), dtype=np.float64)
    for i, lat_val in enumerate(lat_axis):
        for j, lon_val in enumerate(lon_axis):
            for k, depth_val in enumerate(depth_axis):
                idx = row_lookup[(float(lat_val), float(lon_val), float(depth_val))]
                north_grid[i, j, k] = float(north_mps[idx])
                east_grid[i, j, k] = float(east_mps[idx])
    return lat_axis, lon_axis, depth_axis, north_grid, east_grid


def load_regular_csv_current_field(
    raw_csv_path: str | Path,
    *,
    name: str,
    region_name: str,
    source_name: str,
    metadata_extra: Optional[dict[str, Any]] = None,
) -> CurrentFieldGrid:
    p = Path(raw_csv_path).expanduser().resolve()
    rows: list[dict[str, str]] = []
    with p.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"lat_deg", "lon_deg", "depth_m", "north_current_mps", "east_current_mps"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"Missing required CSV columns {sorted(missing)} in {p.name}.")
        rows.extend(row for row in reader if row)
    if not rows:
        raise ValueError(f"CSV current field {p} is empty.")
    lat_deg = np.asarray([float(row["lat_deg"]) for row in rows], dtype=np.float64)
    lon_deg = np.asarray([float(row["lon_deg"]) for row in rows], dtype=np.float64)
    depth_m = np.asarray([float(row["depth_m"]) for row in rows], dtype=np.float64)
    north_mps = np.asarray([float(row["north_current_mps"]) for row in rows], dtype=np.float64)
    east_mps = np.asarray([float(row["east_current_mps"]) for row in rows], dtype=np.float64)
    lat_axis, lon_axis, depth_axis, north_grid, east_grid = _validate_regular_grid(
        lat_deg,
        lon_deg,
        depth_m,
        north_mps,
        east_mps,
    )
    metadata = {
        "region_name": region_name,
        "source_name": source_name,
        "source_kind": "regular_grid_csv",
        "raw_data_path": str(p),
    }
    if metadata_extra:
        metadata.update(dict(metadata_extra))
    return CurrentFieldGrid(
        lat_axis_deg=lat_axis,
        lon_axis_deg=lon_axis,
        depth_axis_m=depth_axis,
        north_current_grid_mps=north_grid,
        east_current_grid_mps=east_grid,
        name=name,
        metadata=metadata,
    )


def process_regular_csv_current_field(
    raw_csv_path: str | Path,
    *,
    region_name: str,
    source_name: str,
    processed_grid_path: str | Path,
    manifest_path: str | Path,
    project_root: str | Path | None = None,
    notes: Optional[list[str]] = None,
    metadata_extra: Optional[dict[str, Any]] = None,
) -> tuple[CurrentFieldGrid, RegionalCurrentFieldManifest]:
    root = _project_root(project_root)
    raw_path = Path(raw_csv_path).expanduser().resolve()
    processed_path = Path(processed_grid_path).expanduser().resolve()
    manifest_path_resolved = Path(manifest_path).expanduser().resolve()

    grid = load_regular_csv_current_field(
        raw_path,
        name=f"{region_name}_current_field",
        region_name=region_name,
        source_name=source_name,
        metadata_extra=metadata_extra,
    )
    grid.to_npz(processed_path)
    manifest = RegionalCurrentFieldManifest(
        region_name=region_name,
        source_name=source_name,
        source_kind="regular_grid_csv",
        raw_data_path=_relative_to_root(raw_path, project_root=root),
        processed_grid_path=_relative_to_root(processed_path, project_root=root),
        manifest_path=_relative_to_root(manifest_path_resolved, project_root=root),
        current_units="m/s",
        lat_bounds_deg=(float(grid.lat_axis_deg[0]), float(grid.lat_axis_deg[-1])),
        lon_bounds_deg=(float(grid.lon_axis_deg[0]), float(grid.lon_axis_deg[-1])),
        depth_bounds_m=(float(grid.depth_axis_m[0]), float(grid.depth_axis_m[-1])),
        spacing_deg=(
            0.0 if grid.lat_axis_deg.size < 2 else float(np.mean(np.diff(grid.lat_axis_deg))),
            0.0 if grid.lon_axis_deg.size < 2 else float(np.mean(np.diff(grid.lon_axis_deg))),
        ),
        depth_spacing_m=(
            0.0 if grid.depth_axis_m.size < 2 else float(np.mean(np.diff(grid.depth_axis_m)))
        ),
        shape=(
            int(grid.lat_axis_deg.size),
            int(grid.lon_axis_deg.size),
            int(grid.depth_axis_m.size),
        ),
        interpolation="linear",
        notes=[] if notes is None else [str(x) for x in notes],
        metadata={} if metadata_extra is None else dict(metadata_extra),
    )
    manifest.write_json(manifest_path_resolved)
    return grid, manifest


__all__ = [
    "CurrentFieldGrid",
    "RegionalCurrentFieldManifest",
    "load_current_field_from_manifest",
    "load_current_manifest",
    "load_regular_csv_current_field",
    "process_regular_csv_current_field",
]
