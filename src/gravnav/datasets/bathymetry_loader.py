"""
bathymetry_loader.py

Regional bathymetry-grid ingestion and cache helpers.

This module mirrors the existing gravity dataset layer closely, but keeps the
runtime representation separate because bathymetry is not a gravity anomaly.
The navigation stack uses this grid only as an optional supporting passive aid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..physics.gravity_map import GravityGridMap
from ..physics.gravity_map import bilinear_interpolate_rectilinear
from .gravity_loader import (
    RegionalGravityMapManifest,
    load_regional_map_from_manifest,
)
from .current_loader import (
    CurrentFieldGrid,
    RegionalCurrentFieldManifest,
    load_current_field_from_manifest,
)
from .magnetic_loader import (
    MagneticGrid,
    RegionalMagneticMapManifest,
    load_magnetic_grid_from_manifest,
)
from ..utils.config import find_project_root

FloatArray = NDArray[np.float64]


RAW_RELATIVE_DIR = Path("data/bathymetry/raw/norwegian_margin_public")
RAW_FIXTURE_NAME = "gebco_norwegian_margin_fixture.csv"
PROCESSED_RELATIVE_DIR = Path("data/bathymetry/processed")
PROCESSED_GRID_NAME = "norwegian_margin_public_bathymetry_grid.npz"
PROCESSED_MANIFEST_NAME = "norwegian_margin_public_bathymetry_manifest.json"
DEMO_PACK_MANIFEST_NAME = "norwegian_margin_public_demo_pack.json"


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


def _axis_spacing_deg(axis: np.ndarray) -> float:
    if axis.size < 2:
        return float("nan")
    return float(np.mean(np.diff(axis)))


def _validate_regular_grid(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    elevation_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_axis = np.unique(lat_deg)
    lon_axis = np.unique(lon_deg)
    expected_size = lat_axis.size * lon_axis.size
    if lat_deg.size != expected_size:
        raise ValueError(
            "Regular-grid CSV is incomplete or contains duplicates: "
            f"expected {expected_size} rows from {lat_axis.size}x{lon_axis.size} "
            f"grid, got {lat_deg.size}."
        )

    row_lookup = {
        (float(la), float(lo)): idx
        for idx, (la, lo) in enumerate(zip(lat_deg, lon_deg))
    }
    if len(row_lookup) != lat_deg.size:
        raise ValueError("Regular-grid CSV contains duplicate lat/lon samples.")

    elevation_grid = np.empty((lat_axis.size, lon_axis.size), dtype=np.float64)
    for i, lat_val in enumerate(lat_axis):
        for j, lon_val in enumerate(lon_axis):
            key = (float(lat_val), float(lon_val))
            if key not in row_lookup:
                raise ValueError(
                    f"Missing regular-grid sample at lat={lat_val}, lon={lon_val}."
                )
            elevation_grid[i, j] = float(elevation_m[row_lookup[key]])

    return lat_axis, lon_axis, elevation_grid


@dataclass(frozen=True)
class BathymetryGrid:
    """
    Regular lat/lon bathymetry grid storing seafloor elevation [m].

    Elevation follows the standard topography convention:
    - negative: below the reference surface / sea level
    - positive: land elevation above the reference surface
    """

    lat_axis_deg: FloatArray
    lon_axis_deg: FloatArray
    elevation_grid_m: FloatArray
    reference_surface_height_m: float = 0.0
    default_method: str = "linear"
    bounds_error: bool = False
    fill_value_m: float = np.nan
    name: str = "bathymetry_grid"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lat_axis_deg = _vec1_strictly_increasing(
            self.lat_axis_deg,
            name="lat_axis_deg",
        )
        lon_axis_deg = _vec1_strictly_increasing(
            self.lon_axis_deg,
            name="lon_axis_deg",
        )
        elevation_grid_m = _grid2(
            self.elevation_grid_m,
            shape=(lat_axis_deg.size, lon_axis_deg.size),
            name="elevation_grid_m",
        )
        object.__setattr__(self, "lat_axis_deg", lat_axis_deg)
        object.__setattr__(self, "lon_axis_deg", lon_axis_deg)
        object.__setattr__(self, "elevation_grid_m", elevation_grid_m)
        object.__setattr__(
            self,
            "reference_surface_height_m",
            float(self.reference_surface_height_m),
        )
        object.__setattr__(self, "default_method", str(self.default_method))
        object.__setattr__(self, "bounds_error", bool(self.bounds_error))
        object.__setattr__(self, "fill_value_m", float(self.fill_value_m))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def shape(self) -> tuple[int, int]:
        return (
            int(self.lat_axis_deg.size),
            int(self.lon_axis_deg.size),
        )

    @property
    def lat_bounds_deg(self) -> tuple[float, float]:
        return (float(self.lat_axis_deg[0]), float(self.lat_axis_deg[-1]))

    @property
    def lon_bounds_deg(self) -> tuple[float, float]:
        return (float(self.lon_axis_deg[0]), float(self.lon_axis_deg[-1]))

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

    def evaluate_elevation_m(
        self,
        lat_deg: ArrayLike,
        lon_deg: ArrayLike,
        *,
        method: Optional[str] = None,
    ):
        return bilinear_interpolate_rectilinear(
            self.lat_axis_deg,
            self.lon_axis_deg,
            self.elevation_grid_m,
            lat_deg,
            lon_deg,
            bounds_error=self.bounds_error,
            fill_value=self.fill_value_m,
            method=self.default_method if method is None else str(method),
        )

    def evaluate_water_depth_m(
        self,
        lat_deg: ArrayLike,
        lon_deg: ArrayLike,
        *,
        reference_surface_height_m: Optional[float] = None,
        method: Optional[str] = None,
    ):
        href = (
            self.reference_surface_height_m
            if reference_surface_height_m is None
            else float(reference_surface_height_m)
        )
        elevation = np.asarray(
            self.evaluate_elevation_m(lat_deg, lon_deg, method=method),
            dtype=np.float64,
        )
        depth = np.maximum(0.0, href - elevation)
        if depth.ndim == 0:
            return float(depth)
        return depth

    def evaluate_water_depth_gradient_m_per_m(
        self,
        lat_deg: ArrayLike,
        lon_deg: ArrayLike,
        *,
        reference_surface_height_m: Optional[float] = None,
        delta_lat_deg: Optional[float] = None,
        delta_lon_deg: Optional[float] = None,
    ) -> FloatArray:
        lat = np.asarray(lat_deg, dtype=np.float64)
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat_b, lon_b = np.broadcast_arrays(lat, lon)
        lat_step = (
            float(delta_lat_deg)
            if delta_lat_deg is not None
            else max(1.0e-4, 0.5 * _axis_spacing_deg(self.lat_axis_deg))
        )
        lon_step = (
            float(delta_lon_deg)
            if delta_lon_deg is not None
            else max(1.0e-4, 0.5 * _axis_spacing_deg(self.lon_axis_deg))
        )
        north_m_per_deg = 111_320.0
        east_m_per_deg = np.maximum(
            north_m_per_deg * np.cos(np.deg2rad(lat_b)),
            1.0,
        )
        plus_n = np.asarray(
            self.evaluate_water_depth_m(
                lat_b + lat_step,
                lon_b,
                reference_surface_height_m=reference_surface_height_m,
            ),
            dtype=np.float64,
        )
        minus_n = np.asarray(
            self.evaluate_water_depth_m(
                lat_b - lat_step,
                lon_b,
                reference_surface_height_m=reference_surface_height_m,
            ),
            dtype=np.float64,
        )
        plus_e = np.asarray(
            self.evaluate_water_depth_m(
                lat_b,
                lon_b + lon_step,
                reference_surface_height_m=reference_surface_height_m,
            ),
            dtype=np.float64,
        )
        minus_e = np.asarray(
            self.evaluate_water_depth_m(
                lat_b,
                lon_b - lon_step,
                reference_surface_height_m=reference_surface_height_m,
            ),
            dtype=np.float64,
        )
        d_dn = (plus_n - minus_n) / (2.0 * lat_step * north_m_per_deg)
        d_de = (plus_e - minus_e) / (2.0 * lon_step * east_m_per_deg)
        return np.stack([d_dn, d_de], axis=-1).astype(np.float64)

    def evaluate_rugosity_m(
        self,
        lat_deg: ArrayLike,
        lon_deg: ArrayLike,
        *,
        reference_surface_height_m: Optional[float] = None,
    ):
        lat = np.asarray(lat_deg, dtype=np.float64)
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat_b, lon_b = np.broadcast_arrays(lat, lon)
        lat_step = max(1.0e-4, _axis_spacing_deg(self.lat_axis_deg))
        lon_step = max(1.0e-4, _axis_spacing_deg(self.lon_axis_deg))
        offsets = (
            (-lat_step, -lon_step),
            (-lat_step, 0.0),
            (-lat_step, lon_step),
            (0.0, -lon_step),
            (0.0, 0.0),
            (0.0, lon_step),
            (lat_step, -lon_step),
            (lat_step, 0.0),
            (lat_step, lon_step),
        )
        samples = [
            np.asarray(
                self.evaluate_water_depth_m(
                    lat_b + dlat,
                    lon_b + dlon,
                    reference_surface_height_m=reference_surface_height_m,
                ),
                dtype=np.float64,
            )
            for dlat, dlon in offsets
        ]
        stacked = np.stack(samples, axis=0)
        rugosity = np.std(stacked, axis=0)
        if rugosity.ndim == 0:
            return float(rugosity)
        return rugosity.astype(np.float64)

    def to_npz(self, path: str | Path) -> Path:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            lat_axis_deg=np.asarray(self.lat_axis_deg, dtype=np.float64),
            lon_axis_deg=np.asarray(self.lon_axis_deg, dtype=np.float64),
            elevation_grid_m=np.asarray(self.elevation_grid_m, dtype=np.float64),
            reference_surface_height_m=np.array(
                float(self.reference_surface_height_m),
                dtype=np.float64,
            ),
            default_method=np.array(str(self.default_method)),
            bounds_error=np.array(bool(self.bounds_error)),
            fill_value_m=np.array(float(self.fill_value_m), dtype=np.float64),
            name=np.array(str(self.name)),
            metadata_json=np.array(json.dumps(_jsonable(self.metadata))),
        )
        return p

    @classmethod
    def from_npz(cls, path: str | Path) -> "BathymetryGrid":
        p = Path(path).expanduser().resolve()
        with np.load(p, allow_pickle=False) as data:
            return cls(
                lat_axis_deg=np.asarray(data["lat_axis_deg"], dtype=np.float64),
                lon_axis_deg=np.asarray(data["lon_axis_deg"], dtype=np.float64),
                elevation_grid_m=np.asarray(data["elevation_grid_m"], dtype=np.float64),
                reference_surface_height_m=float(data["reference_surface_height_m"]),
                default_method=str(np.asarray(data["default_method"]).item()),
                bounds_error=bool(np.asarray(data["bounds_error"]).item()),
                fill_value_m=float(data["fill_value_m"]),
                name=str(np.asarray(data["name"]).item()),
                metadata=json.loads(str(np.asarray(data["metadata_json"]).item())),
            )


@dataclass(frozen=True)
class RegionalBathymetryManifest:
    """Compact metadata sidecar for one processed regional bathymetry grid."""

    region_name: str
    source_name: str
    source_kind: str
    raw_data_path: str
    processed_grid_path: str
    manifest_path: str
    elevation_units: str
    reference_surface_height_m: float
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
    def from_mapping(cls, mapping: dict[str, Any]) -> "RegionalBathymetryManifest":
        return cls(
            region_name=str(mapping["region_name"]),
            source_name=str(mapping["source_name"]),
            source_kind=str(mapping["source_kind"]),
            raw_data_path=str(mapping["raw_data_path"]),
            processed_grid_path=str(mapping["processed_grid_path"]),
            manifest_path=str(mapping["manifest_path"]),
            elevation_units=str(mapping["elevation_units"]),
            reference_surface_height_m=float(mapping["reference_surface_height_m"]),
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


@dataclass(frozen=True)
class RegionalDemoPackManifest:
    """Small manifest linking the regional gravity and bathymetry products."""

    region_name: str
    gravity_manifest_path: str
    bathymetry_manifest_path: str
    scenario_path: str
    sequence_profile_path: str
    magnetic_manifest_path: Optional[str] = None
    current_manifest_path: Optional[str] = None
    tide_config_path: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "RegionalDemoPackManifest":
        return cls(
            region_name=str(mapping["region_name"]),
            gravity_manifest_path=str(mapping["gravity_manifest_path"]),
            bathymetry_manifest_path=str(mapping["bathymetry_manifest_path"]),
            scenario_path=str(mapping["scenario_path"]),
            sequence_profile_path=str(mapping["sequence_profile_path"]),
            magnetic_manifest_path=(
                None
                if mapping.get("magnetic_manifest_path") is None
                else str(mapping["magnetic_manifest_path"])
            ),
            current_manifest_path=(
                None
                if mapping.get("current_manifest_path") is None
                else str(mapping["current_manifest_path"])
            ),
            tide_config_path=(
                None
                if mapping.get("tide_config_path") is None
                else str(mapping["tide_config_path"])
            ),
            notes=[str(x) for x in mapping.get("notes", [])],
            metadata=dict(mapping.get("metadata", {})),
        )

    def write_json(self, path: str | Path) -> Path:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_mapping(), indent=2) + "\n", encoding="utf-8")
        return p


@dataclass(frozen=True)
class ResolvedRegionalDemoPack:
    """Concrete assets resolved from one demo-pack manifest."""

    manifest: RegionalDemoPackManifest
    manifest_path: Path
    gravity_manifest: RegionalGravityMapManifest
    gravity_manifest_path: Path
    gravity_map: GravityGridMap
    gravity_map_path: Path
    bathymetry_manifest: RegionalBathymetryManifest
    bathymetry_manifest_path: Path
    bathymetry_grid: BathymetryGrid
    bathymetry_grid_path: Path
    magnetic_manifest: Optional[RegionalMagneticMapManifest]
    magnetic_manifest_path: Optional[Path]
    magnetic_grid: Optional[MagneticGrid]
    magnetic_grid_path: Optional[Path]
    current_manifest: Optional[RegionalCurrentFieldManifest]
    current_manifest_path: Optional[Path]
    current_field: Optional[CurrentFieldGrid]
    current_field_grid_path: Optional[Path]
    tide_config_path: Optional[Path]
    scenario_path: Path
    sequence_profile_path: Path


def load_bathymetry_manifest(path: str | Path) -> RegionalBathymetryManifest:
    p = Path(path).expanduser().resolve()
    return RegionalBathymetryManifest.from_mapping(
        json.loads(p.read_text(encoding="utf-8"))
    )


def load_demo_pack_manifest(path: str | Path) -> RegionalDemoPackManifest:
    p = Path(path).expanduser().resolve()
    return RegionalDemoPackManifest.from_mapping(
        json.loads(p.read_text(encoding="utf-8"))
    )


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
    if ref.parts and ref.parts[0] in {"configs", "data", "scripts", "src", "tests"}:
        return root_candidate
    return candidate


def load_bathymetry_grid_from_manifest(
    path: str | Path,
) -> tuple[BathymetryGrid, RegionalBathymetryManifest, Path, Path]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = load_bathymetry_manifest(manifest_path)
    processed_grid_path = _resolve_manifest_ref(
        manifest.processed_grid_path,
        manifest_path=manifest_path,
    )
    return (
        BathymetryGrid.from_npz(processed_grid_path),
        manifest,
        processed_grid_path,
        manifest_path,
    )


def resolve_regional_demo_pack(path: str | Path) -> ResolvedRegionalDemoPack:
    manifest_path = Path(path).expanduser().resolve()
    manifest = load_demo_pack_manifest(manifest_path)
    gravity_map, gravity_manifest, gravity_map_path, gravity_manifest_path = (
        load_regional_map_from_manifest(
            _resolve_manifest_ref(
                manifest.gravity_manifest_path,
                manifest_path=manifest_path,
            )
        )
    )
    (
        bathymetry_grid,
        bathymetry_manifest,
        bathymetry_grid_path,
        bathymetry_manifest_path,
    ) = load_bathymetry_grid_from_manifest(
        _resolve_manifest_ref(
            manifest.bathymetry_manifest_path,
            manifest_path=manifest_path,
        )
    )
    magnetic_grid: Optional[MagneticGrid] = None
    magnetic_manifest: Optional[RegionalMagneticMapManifest] = None
    magnetic_grid_path: Optional[Path] = None
    magnetic_manifest_path: Optional[Path] = None
    if manifest.magnetic_manifest_path is not None:
        (
            magnetic_grid,
            magnetic_manifest,
            magnetic_grid_path,
            magnetic_manifest_path,
        ) = load_magnetic_grid_from_manifest(
            _resolve_manifest_ref(
                manifest.magnetic_manifest_path,
                manifest_path=manifest_path,
            )
        )
    current_field: Optional[CurrentFieldGrid] = None
    current_manifest: Optional[RegionalCurrentFieldManifest] = None
    current_field_grid_path: Optional[Path] = None
    current_manifest_path: Optional[Path] = None
    if manifest.current_manifest_path is not None:
        (
            current_field,
            current_manifest,
            current_field_grid_path,
            current_manifest_path,
        ) = load_current_field_from_manifest(
            _resolve_manifest_ref(
                manifest.current_manifest_path,
                manifest_path=manifest_path,
            )
        )
    scenario_path = _resolve_manifest_ref(
        manifest.scenario_path,
        manifest_path=manifest_path,
    )
    sequence_profile_path = _resolve_manifest_ref(
        manifest.sequence_profile_path,
        manifest_path=manifest_path,
    )
    tide_config_path = (
        None
        if manifest.tide_config_path is None
        else _resolve_manifest_ref(
            manifest.tide_config_path,
            manifest_path=manifest_path,
        )
    )
    return ResolvedRegionalDemoPack(
        manifest=manifest,
        manifest_path=manifest_path,
        gravity_manifest=gravity_manifest,
        gravity_manifest_path=gravity_manifest_path,
        gravity_map=gravity_map,
        gravity_map_path=gravity_map_path,
        bathymetry_manifest=bathymetry_manifest,
        bathymetry_manifest_path=bathymetry_manifest_path,
        bathymetry_grid=bathymetry_grid,
        bathymetry_grid_path=bathymetry_grid_path,
        magnetic_manifest=magnetic_manifest,
        magnetic_manifest_path=magnetic_manifest_path,
        magnetic_grid=magnetic_grid,
        magnetic_grid_path=magnetic_grid_path,
        current_manifest=current_manifest,
        current_manifest_path=current_manifest_path,
        current_field=current_field,
        current_field_grid_path=current_field_grid_path,
        tide_config_path=tide_config_path,
        scenario_path=scenario_path,
        sequence_profile_path=sequence_profile_path,
    )


def load_regular_csv_bathymetry_grid(
    raw_csv_path: str | Path,
    *,
    name: str,
    region_name: str,
    source_name: str,
    reference_surface_height_m: float = 0.0,
    default_method: str = "linear",
    bounds_error: bool = False,
    fill_value_m: float = np.nan,
    metadata_extra: Optional[dict[str, Any]] = None,
) -> BathymetryGrid:
    p = Path(raw_csv_path).expanduser().resolve()
    rows: list[dict[str, str]] = []
    with p.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"lat_deg", "lon_deg", "elevation_m"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(
                f"Missing required CSV columns {sorted(missing)} in {p.name}."
            )
        for row in reader:
            if row:
                rows.append(row)
    if len(rows) == 0:
        raise ValueError(f"CSV bathymetry grid {p} is empty.")

    lat_deg = np.asarray([float(row["lat_deg"]) for row in rows], dtype=np.float64)
    lon_deg = np.asarray([float(row["lon_deg"]) for row in rows], dtype=np.float64)
    elevation_m = np.asarray([float(row["elevation_m"]) for row in rows], dtype=np.float64)
    lat_axis_deg, lon_axis_deg, elevation_grid_m = _validate_regular_grid(
        lat_deg,
        lon_deg,
        elevation_m,
    )

    metadata = {
        "region_name": region_name,
        "source_name": source_name,
        "source_kind": "regular_grid_csv",
        "raw_data_path": str(p),
    }
    if metadata_extra:
        metadata.update(dict(metadata_extra))

    return BathymetryGrid(
        lat_axis_deg=lat_axis_deg,
        lon_axis_deg=lon_axis_deg,
        elevation_grid_m=elevation_grid_m,
        reference_surface_height_m=reference_surface_height_m,
        default_method=default_method,
        bounds_error=bounds_error,
        fill_value_m=fill_value_m,
        name=name,
        metadata=metadata,
    )


def process_regular_csv_bathymetry_grid(
    raw_csv_path: str | Path,
    *,
    processed_npz_path: str | Path,
    manifest_path: str | Path,
    region_name: str,
    source_name: str,
    reference_surface_height_m: float = 0.0,
    default_method: str = "linear",
    bounds_error: bool = False,
    fill_value_m: float = np.nan,
    metadata_extra: Optional[dict[str, Any]] = None,
    project_root: str | Path | None = None,
) -> tuple[BathymetryGrid, RegionalBathymetryManifest]:
    root = _project_root(project_root)
    raw_path = Path(raw_csv_path).expanduser().resolve()
    grid = load_regular_csv_bathymetry_grid(
        raw_path,
        name=f"{region_name}_bathymetry_grid",
        region_name=region_name,
        source_name=source_name,
        reference_surface_height_m=reference_surface_height_m,
        default_method=default_method,
        bounds_error=bounds_error,
        fill_value_m=fill_value_m,
        metadata_extra=metadata_extra,
    )
    processed_path = grid.to_npz(processed_npz_path)
    manifest_target = Path(manifest_path).expanduser().resolve()
    manifest = RegionalBathymetryManifest(
        region_name=region_name,
        source_name=source_name,
        source_kind="regular_grid_csv",
        raw_data_path=_relative_to_root(raw_path, project_root=root),
        processed_grid_path=_relative_to_root(processed_path, project_root=root),
        manifest_path=_relative_to_root(manifest_target, project_root=root),
        elevation_units="m",
        reference_surface_height_m=float(reference_surface_height_m),
        lat_bounds_deg=grid.lat_bounds_deg,
        lon_bounds_deg=grid.lon_bounds_deg,
        spacing_deg=(
            _axis_spacing_deg(np.asarray(grid.lat_axis_deg, dtype=np.float64)),
            _axis_spacing_deg(np.asarray(grid.lon_axis_deg, dtype=np.float64)),
        ),
        shape=grid.shape,
        interpolation=str(grid.default_method),
        notes=[
            "Processed into BathymetryGrid NPZ cache for regional maritime demo use.",
            "Elevation is relative to the reference surface; water depth is max(0, href - elevation).",
        ],
        metadata={
            "grid_name": grid.name,
            "bounds_error": bool(grid.bounds_error),
            "fill_value_m": (
                None
                if not np.isfinite(float(grid.fill_value_m))
                else float(grid.fill_value_m)
            ),
            "raw_loader": "load_regular_csv_bathymetry_grid",
            **({} if metadata_extra is None else dict(metadata_extra)),
        },
    )
    manifest.write_json(manifest_target)
    return grid, manifest


def process_regular_array_bathymetry_grid(
    *,
    lat_axis_deg: ArrayLike,
    lon_axis_deg: ArrayLike,
    elevation_grid_m: ArrayLike,
    raw_data_path: str | Path,
    processed_npz_path: str | Path,
    manifest_path: str | Path,
    region_name: str,
    source_name: str,
    source_kind: str = "regular_grid_array",
    reference_surface_height_m: float = 0.0,
    default_method: str = "linear",
    bounds_error: bool = False,
    fill_value_m: float = np.nan,
    metadata_extra: Optional[dict[str, Any]] = None,
    project_root: str | Path | None = None,
) -> tuple[BathymetryGrid, RegionalBathymetryManifest]:
    root = _project_root(project_root)
    raw_path = Path(raw_data_path).expanduser().resolve()
    grid = BathymetryGrid(
        lat_axis_deg=np.asarray(lat_axis_deg, dtype=np.float64),
        lon_axis_deg=np.asarray(lon_axis_deg, dtype=np.float64),
        elevation_grid_m=np.asarray(elevation_grid_m, dtype=np.float64),
        reference_surface_height_m=reference_surface_height_m,
        default_method=default_method,
        bounds_error=bounds_error,
        fill_value_m=fill_value_m,
        name=f"{region_name}_bathymetry_grid",
        metadata={
            "region_name": region_name,
            "source_name": source_name,
            "source_kind": source_kind,
            "raw_data_path": str(raw_path),
            **({} if metadata_extra is None else dict(metadata_extra)),
        },
    )
    processed_path = grid.to_npz(processed_npz_path)
    manifest_target = Path(manifest_path).expanduser().resolve()
    manifest = RegionalBathymetryManifest(
        region_name=region_name,
        source_name=source_name,
        source_kind=source_kind,
        raw_data_path=_relative_to_root(raw_path, project_root=root),
        processed_grid_path=_relative_to_root(processed_path, project_root=root),
        manifest_path=_relative_to_root(manifest_target, project_root=root),
        elevation_units="m",
        reference_surface_height_m=float(reference_surface_height_m),
        lat_bounds_deg=grid.lat_bounds_deg,
        lon_bounds_deg=grid.lon_bounds_deg,
        spacing_deg=(
            _axis_spacing_deg(np.asarray(grid.lat_axis_deg, dtype=np.float64)),
            _axis_spacing_deg(np.asarray(grid.lon_axis_deg, dtype=np.float64)),
        ),
        shape=grid.shape,
        interpolation=str(grid.default_method),
        notes=[
            "Processed into BathymetryGrid NPZ cache for regional maritime demo use.",
            "Elevation is relative to the reference surface; water depth is max(0, href - elevation).",
        ],
        metadata={
            "grid_name": grid.name,
            "bounds_error": bool(grid.bounds_error),
            "fill_value_m": (
                None
                if not np.isfinite(float(grid.fill_value_m))
                else float(grid.fill_value_m)
            ),
            "raw_loader": "process_regular_array_bathymetry_grid",
            **({} if metadata_extra is None else dict(metadata_extra)),
        },
    )
    manifest.write_json(manifest_target)
    return grid, manifest


def ensure_regional_bathymetry_grid(
    region_name: str,
    *,
    project_root: str | Path | None = None,
    raw_path: str | Path | None = None,
    force_reprocess: bool = False,
) -> tuple[BathymetryGrid, RegionalBathymetryManifest, Path, Path]:
    root = _project_root(project_root)
    region = str(region_name).strip().lower()
    if region != "norwegian_margin_public":
        raise ValueError(
            f"Unsupported regional bathymetry grid {region_name!r}. "
            "Supported values: 'norwegian_margin_public'."
        )

    default_raw_path = root / RAW_RELATIVE_DIR / RAW_FIXTURE_NAME
    raw_target = default_raw_path if raw_path is None else Path(raw_path).expanduser().resolve()
    processed_path = (root / PROCESSED_RELATIVE_DIR / PROCESSED_GRID_NAME).resolve()
    manifest_path = (root / PROCESSED_RELATIVE_DIR / PROCESSED_MANIFEST_NAME).resolve()

    if processed_path.exists() and manifest_path.exists() and not force_reprocess:
        return (
            BathymetryGrid.from_npz(processed_path),
            load_bathymetry_manifest(manifest_path),
            processed_path,
            manifest_path,
        )

    if not raw_target.exists():
        raise FileNotFoundError(
            "Norwegian public bathymetry raw grid not found. Expected a regular-grid CSV "
            f"at {raw_target}. Place a clipped GEBCO-style product there or pass raw_path."
        )

    grid, manifest = process_regular_csv_bathymetry_grid(
        raw_target,
        processed_npz_path=processed_path,
        manifest_path=manifest_path,
        region_name="norwegian_margin_public",
        source_name="gebco_fixture" if raw_target == default_raw_path else raw_target.stem,
        reference_surface_height_m=0.0,
        default_method="linear",
        bounds_error=False,
        fill_value_m=np.nan,
        metadata_extra={
            "region_label": "Norwegian margin public bathymetry",
            "fixture": bool(raw_target == default_raw_path),
        },
        project_root=root,
    )
    return grid, manifest, processed_path, manifest_path


__all__ = [
    "BathymetryGrid",
    "DEMO_PACK_MANIFEST_NAME",
    "PROCESSED_GRID_NAME",
    "PROCESSED_MANIFEST_NAME",
    "RAW_FIXTURE_NAME",
    "RAW_RELATIVE_DIR",
    "RegionalBathymetryManifest",
    "RegionalDemoPackManifest",
    "ResolvedRegionalDemoPack",
    "ensure_regional_bathymetry_grid",
    "load_bathymetry_grid_from_manifest",
    "load_bathymetry_manifest",
    "load_demo_pack_manifest",
    "load_regular_csv_bathymetry_grid",
    "process_regular_array_bathymetry_grid",
    "process_regular_csv_bathymetry_grid",
    "resolve_regional_demo_pack",
]
