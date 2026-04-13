"""
gravity_loader.py

Small dataset/cache layer for regional gravity grids.

Design constraints
------------------
- The simulator keeps ``GravityGridMap`` as the only runtime map abstraction.
- This module only ingests external regional products and converts them into
  ``GravityGridMap`` plus a compact manifest.
- v1 intentionally supports one simple raw format: a regular-grid CSV table with
  columns ``lat_deg``, ``lon_deg``, and ``disturbance_mgal``. An optional
  ``vertical_gradient_mgal_per_m`` column is accepted when present.

The default repo fixture is a small Norwegian-margin grid under
``data/gravity_maps/raw/norwegian_margin``. Operators can replace that file with
their own public regional product as long as it follows the same schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..physics.gravity_map import GravityGridMap
from ..utils.config import find_project_root


RAW_RELATIVE_DIR = Path("data/gravity_maps/raw/norwegian_margin")
RAW_FIXTURE_NAME = "norwegian_margin_fixture.csv"
PROCESSED_RELATIVE_DIR = Path("data/gravity_maps/processed")
PROCESSED_MAP_NAME = "norwegian_margin_gravity_map.npz"
PROCESSED_MANIFEST_NAME = "norwegian_margin_gravity_map_manifest.json"


def _project_root(project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).expanduser().resolve()
    return find_project_root()


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


@dataclass(frozen=True)
class RegionalGravityMapManifest:
    """Compact metadata sidecar for one processed regional gravity map."""

    region_name: str
    source_name: str
    source_kind: str
    raw_data_path: str
    processed_map_path: str
    manifest_path: str
    disturbance_units: str
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
    def from_mapping(cls, mapping: dict[str, Any]) -> "RegionalGravityMapManifest":
        return cls(
            region_name=str(mapping["region_name"]),
            source_name=str(mapping["source_name"]),
            source_kind=str(mapping["source_kind"]),
            raw_data_path=str(mapping["raw_data_path"]),
            processed_map_path=str(mapping["processed_map_path"]),
            manifest_path=str(mapping["manifest_path"]),
            disturbance_units=str(mapping["disturbance_units"]),
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


def load_regional_manifest(path: str | Path) -> RegionalGravityMapManifest:
    p = Path(path).expanduser().resolve()
    return RegionalGravityMapManifest.from_mapping(
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


def load_regional_map_from_manifest(
    path: str | Path,
) -> tuple[GravityGridMap, RegionalGravityMapManifest, Path, Path]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = load_regional_manifest(manifest_path)
    processed_map_path = _resolve_manifest_ref(
        manifest.processed_map_path,
        manifest_path=manifest_path,
    )
    return (
        GravityGridMap.from_npz(processed_map_path),
        manifest,
        processed_map_path,
        manifest_path,
    )


def _validate_regular_grid(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    disturbance_mgal: np.ndarray,
    vertical_gradient_mgal_per_m: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | np.ndarray]:
    lat_axis = np.unique(lat_deg)
    lon_axis = np.unique(lon_deg)
    expected_size = lat_axis.size * lon_axis.size
    if lat_deg.size != expected_size:
        raise ValueError(
            "Regular-grid CSV is incomplete or contains duplicates: "
            f"expected {expected_size} rows from {lat_axis.size}x{lon_axis.size} "
            f"grid, got {lat_deg.size}."
        )

    row_lookup = {(float(la), float(lo)): idx for idx, (la, lo) in enumerate(zip(lat_deg, lon_deg))}
    if len(row_lookup) != lat_deg.size:
        raise ValueError("Regular-grid CSV contains duplicate lat/lon samples.")

    disturbance_grid = np.empty((lat_axis.size, lon_axis.size), dtype=np.float64)
    if vertical_gradient_mgal_per_m is None:
        vertical_gradient: float | np.ndarray = 0.0
    else:
        vertical_gradient = np.empty((lat_axis.size, lon_axis.size), dtype=np.float64)

    for i, lat_val in enumerate(lat_axis):
        for j, lon_val in enumerate(lon_axis):
            key = (float(lat_val), float(lon_val))
            if key not in row_lookup:
                raise ValueError(f"Missing regular-grid sample at lat={lat_val}, lon={lon_val}.")
            idx = row_lookup[key]
            disturbance_grid[i, j] = float(disturbance_mgal[idx])
            if isinstance(vertical_gradient, np.ndarray):
                vertical_gradient[i, j] = float(vertical_gradient_mgal_per_m[idx])

    return lat_axis, lon_axis, disturbance_grid, vertical_gradient


def load_regular_csv_gravity_map(
    raw_csv_path: str | Path,
    *,
    name: str,
    region_name: str,
    source_name: str,
    reference_height_m: float = 0.0,
    default_method: str = "linear",
    bounds_error: bool = False,
    fill_value_mgal: float = np.nan,
    metadata_extra: Optional[dict[str, Any]] = None,
) -> GravityGridMap:
    """
    Load a regular-grid CSV regional gravity map into ``GravityGridMap``.
    """
    p = Path(raw_csv_path).expanduser().resolve()
    rows: list[dict[str, str]] = []
    with p.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"lat_deg", "lon_deg", "disturbance_mgal"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(
                f"Missing required CSV columns {sorted(missing)} in {p.name}."
            )
        for row in reader:
            if not row:
                continue
            rows.append(row)
    if len(rows) == 0:
        raise ValueError(f"CSV gravity map {p} is empty.")

    lat_deg = np.asarray([float(row["lat_deg"]) for row in rows], dtype=np.float64)
    lon_deg = np.asarray([float(row["lon_deg"]) for row in rows], dtype=np.float64)
    disturbance_mgal = np.asarray(
        [float(row["disturbance_mgal"]) for row in rows],
        dtype=np.float64,
    )
    if "vertical_gradient_mgal_per_m" in rows[0]:
        vertical_gradient_mgal_per_m: Optional[np.ndarray] = np.asarray(
            [float(row["vertical_gradient_mgal_per_m"]) for row in rows],
            dtype=np.float64,
        )
    else:
        vertical_gradient_mgal_per_m = None

    lat_axis_deg, lon_axis_deg, disturbance_grid_mgal, vertical_gradient = _validate_regular_grid(
        lat_deg,
        lon_deg,
        disturbance_mgal,
        vertical_gradient_mgal_per_m,
    )

    metadata = {
        "region_name": region_name,
        "source_name": source_name,
        "source_kind": "regular_grid_csv",
        "raw_data_path": str(p),
    }
    if metadata_extra:
        metadata.update(dict(metadata_extra))

    return GravityGridMap.from_degrees(
        lat_axis_deg=lat_axis_deg,
        lon_axis_deg=lon_axis_deg,
        disturbance_grid_mps2=GravityGridMap.from_mgal_grid(
            lat_axis_rad=np.deg2rad(lat_axis_deg),
            lon_axis_rad=np.deg2rad(lon_axis_deg),
            disturbance_grid_mgal=disturbance_grid_mgal,
            vertical_gradient_mgal_per_m=vertical_gradient,
        ).disturbance_grid_mps2,
        reference_height_m=reference_height_m,
        vertical_gradient_mps2_per_m=(
            0.0
            if isinstance(vertical_gradient, float)
            else GravityGridMap.from_mgal_grid(
                lat_axis_rad=np.deg2rad(lat_axis_deg),
                lon_axis_rad=np.deg2rad(lon_axis_deg),
                disturbance_grid_mgal=np.zeros_like(disturbance_grid_mgal),
                vertical_gradient_mgal_per_m=vertical_gradient,
            ).vertical_gradient_mps2_per_m
        ),
        default_method=default_method,
        bounds_error=bounds_error,
        fill_value_mps2=(
            float(fill_value_mgal)
            if np.isnan(float(fill_value_mgal))
            else float(1.0e-5 * fill_value_mgal)
        ),
        name=name,
        metadata=metadata,
    )


def load_regular_xyz_gravity_map(
    raw_xyz_path: str | Path,
    *,
    name: str,
    region_name: str,
    source_name: str,
    column_order: tuple[str, str, str] = ("lon_deg", "lat_deg", "disturbance_mgal"),
    skiprows: int = 0,
    crop_bounds_deg: Optional[tuple[float, float, float, float]] = None,
    reference_height_m: float = 0.0,
    default_method: str = "linear",
    bounds_error: bool = False,
    fill_value_mgal: float = np.nan,
    metadata_extra: Optional[dict[str, Any]] = None,
) -> GravityGridMap:
    """
    Load a regular-grid XYZ text product into ``GravityGridMap``.

    The current supported semantic column names are:
    - ``lat_deg``
    - ``lon_deg``
    - ``disturbance_mgal``
    """
    p = Path(raw_xyz_path).expanduser().resolve()
    table = np.loadtxt(p, dtype=np.float64, ndmin=2, skiprows=int(skiprows))
    if table.ndim != 2 or table.shape[1] < 3:
        raise ValueError(
            f"XYZ gravity map {p} must contain at least 3 numeric columns."
        )

    semantic_names = tuple(str(name).strip().lower() for name in column_order)
    if set(semantic_names) != {"lat_deg", "lon_deg", "disturbance_mgal"}:
        raise ValueError(
            "column_order must be a permutation of "
            "('lat_deg', 'lon_deg', 'disturbance_mgal')."
        )

    column_index = {semantic_names[idx]: idx for idx in range(3)}
    lat_deg = np.asarray(table[:, column_index["lat_deg"]], dtype=np.float64)
    lon_deg = np.asarray(table[:, column_index["lon_deg"]], dtype=np.float64)
    disturbance_mgal = np.asarray(
        table[:, column_index["disturbance_mgal"]],
        dtype=np.float64,
    )
    if crop_bounds_deg is not None:
        lat_min_deg, lat_max_deg, lon_min_deg, lon_max_deg = crop_bounds_deg
        keep = (
            (lat_deg >= float(lat_min_deg))
            & (lat_deg <= float(lat_max_deg))
            & (lon_deg >= float(lon_min_deg))
            & (lon_deg <= float(lon_max_deg))
        )
        if not np.any(keep):
            raise ValueError("Requested crop bounds do not intersect the XYZ grid.")
        lat_deg = lat_deg[keep]
        lon_deg = lon_deg[keep]
        disturbance_mgal = disturbance_mgal[keep]
    lat_axis_deg, lon_axis_deg, disturbance_grid_mgal, vertical_gradient = _validate_regular_grid(
        lat_deg,
        lon_deg,
        disturbance_mgal,
        None,
    )

    metadata = {
        "region_name": region_name,
        "source_name": source_name,
        "source_kind": "regular_grid_xyz",
        "raw_data_path": str(p),
        "column_order": list(semantic_names),
        "skiprows": int(skiprows),
        "crop_bounds_deg": None if crop_bounds_deg is None else list(crop_bounds_deg),
    }
    if metadata_extra:
        metadata.update(dict(metadata_extra))

    return GravityGridMap.from_mgal_grid(
        lat_axis_rad=np.deg2rad(lat_axis_deg),
        lon_axis_rad=np.deg2rad(lon_axis_deg),
        disturbance_grid_mgal=disturbance_grid_mgal,
        reference_height_m=reference_height_m,
        vertical_gradient_mgal_per_m=vertical_gradient,
        default_method=default_method,
        bounds_error=bounds_error,
        fill_value_mgal=fill_value_mgal,
        name=name,
        metadata=metadata,
    )


def _axis_spacing_deg(axis: np.ndarray) -> float:
    if axis.size < 2:
        return float("nan")
    return float(np.mean(np.diff(axis)))


def process_regular_csv_gravity_map(
    raw_csv_path: str | Path,
    *,
    processed_npz_path: str | Path,
    manifest_path: str | Path,
    region_name: str,
    source_name: str,
    reference_height_m: float = 0.0,
    default_method: str = "linear",
    bounds_error: bool = False,
    fill_value_mgal: float = np.nan,
    metadata_extra: Optional[dict[str, Any]] = None,
    project_root: str | Path | None = None,
) -> tuple[GravityGridMap, RegionalGravityMapManifest]:
    """
    Convert one raw CSV gravity grid into a processed NPZ cache and manifest.
    """
    root = _project_root(project_root)
    raw_path = Path(raw_csv_path).expanduser().resolve()
    map_model = load_regular_csv_gravity_map(
        raw_path,
        name=f"{region_name}_gravity_map",
        region_name=region_name,
        source_name=source_name,
        reference_height_m=reference_height_m,
        default_method=default_method,
        bounds_error=bounds_error,
        fill_value_mgal=fill_value_mgal,
        metadata_extra=metadata_extra,
    )
    processed_path = map_model.to_npz(processed_npz_path)
    manifest_target = Path(manifest_path).expanduser().resolve()

    manifest = RegionalGravityMapManifest(
        region_name=region_name,
        source_name=source_name,
        source_kind="regular_grid_csv",
        raw_data_path=_relative_to_root(raw_path, project_root=root),
        processed_map_path=_relative_to_root(processed_path, project_root=root),
        manifest_path=_relative_to_root(manifest_target, project_root=root),
        disturbance_units="mGal",
        reference_height_m=float(reference_height_m),
        lat_bounds_deg=(
            float(map_model.lat_axis_deg[0]),
            float(map_model.lat_axis_deg[-1]),
        ),
        lon_bounds_deg=(
            float(map_model.lon_axis_deg[0]),
            float(map_model.lon_axis_deg[-1]),
        ),
        spacing_deg=(
            _axis_spacing_deg(np.asarray(map_model.lat_axis_deg, dtype=np.float64)),
            _axis_spacing_deg(np.asarray(map_model.lon_axis_deg, dtype=np.float64)),
        ),
        shape=tuple(int(x) for x in map_model.shape),
        interpolation=str(map_model.default_method),
        notes=[
            "Processed into GravityGridMap NPZ cache for direct simulator use.",
            "No upward/downward continuation beyond the map reference height is applied here.",
        ],
        metadata={
            "map_name": map_model.name,
            "bounds_error": bool(map_model.bounds_error),
            "fill_value_mps2": (
                None
                if not np.isfinite(float(map_model.fill_value_mps2))
                else float(map_model.fill_value_mps2)
            ),
            "raw_loader": "load_regular_csv_gravity_map",
            **({} if metadata_extra is None else dict(metadata_extra)),
        },
    )
    manifest.write_json(manifest_target)
    return map_model, manifest


def process_regular_xyz_gravity_map(
    raw_xyz_path: str | Path,
    *,
    processed_npz_path: str | Path,
    manifest_path: str | Path,
    region_name: str,
    source_name: str,
    column_order: tuple[str, str, str] = ("lon_deg", "lat_deg", "disturbance_mgal"),
    skiprows: int = 0,
    reference_height_m: float = 0.0,
    default_method: str = "linear",
    bounds_error: bool = False,
    fill_value_mgal: float = np.nan,
    metadata_extra: Optional[dict[str, Any]] = None,
    project_root: str | Path | None = None,
    crop_bounds_deg: Optional[tuple[float, float, float, float]] = None,
) -> tuple[GravityGridMap, RegionalGravityMapManifest]:
    """
    Convert one raw XYZ gravity grid into a processed NPZ cache and manifest.
    """
    root = _project_root(project_root)
    raw_path = Path(raw_xyz_path).expanduser().resolve()
    map_model = load_regular_xyz_gravity_map(
        raw_path,
        name=f"{region_name}_gravity_map",
        region_name=region_name,
        source_name=source_name,
        column_order=column_order,
        skiprows=skiprows,
        crop_bounds_deg=crop_bounds_deg,
        reference_height_m=reference_height_m,
        default_method=default_method,
        bounds_error=bounds_error,
        fill_value_mgal=fill_value_mgal,
        metadata_extra=metadata_extra,
    )

    processed_path = map_model.to_npz(processed_npz_path)
    manifest_target = Path(manifest_path).expanduser().resolve()
    manifest = RegionalGravityMapManifest(
        region_name=region_name,
        source_name=source_name,
        source_kind="regular_grid_xyz",
        raw_data_path=_relative_to_root(raw_path, project_root=root),
        processed_map_path=_relative_to_root(processed_path, project_root=root),
        manifest_path=_relative_to_root(manifest_target, project_root=root),
        disturbance_units="mGal",
        reference_height_m=float(reference_height_m),
        lat_bounds_deg=(
            float(map_model.lat_axis_deg[0]),
            float(map_model.lat_axis_deg[-1]),
        ),
        lon_bounds_deg=(
            float(map_model.lon_axis_deg[0]),
            float(map_model.lon_axis_deg[-1]),
        ),
        spacing_deg=(
            _axis_spacing_deg(np.asarray(map_model.lat_axis_deg, dtype=np.float64)),
            _axis_spacing_deg(np.asarray(map_model.lon_axis_deg, dtype=np.float64)),
        ),
        shape=tuple(int(x) for x in map_model.shape),
        interpolation=str(map_model.default_method),
        notes=[
            "Processed from a regular XYZ anomaly grid into GravityGridMap NPZ cache.",
            "This source is used as a regional scalar map proxy; check source semantics before interpreting it as same-point disturbance.",
        ],
        metadata={
            "map_name": map_model.name,
            "bounds_error": bool(map_model.bounds_error),
            "fill_value_mps2": (
                None
                if not np.isfinite(float(map_model.fill_value_mps2))
                else float(map_model.fill_value_mps2)
            ),
            "raw_loader": "load_regular_xyz_gravity_map",
            "column_order": list(column_order),
            "skiprows": int(skiprows),
            **({} if metadata_extra is None else dict(metadata_extra)),
        },
    )
    manifest.write_json(manifest_target)
    return map_model, manifest


def ensure_regional_gravity_map(
    region_name: str,
    *,
    project_root: str | Path | None = None,
    raw_path: str | Path | None = None,
    force_reprocess: bool = False,
) -> tuple[GravityGridMap, RegionalGravityMapManifest, Path, Path]:
    """
    Ensure that a processed NPZ cache exists for one supported regional map.
    """
    root = _project_root(project_root)
    region = str(region_name).strip().lower()
    if region != "norwegian_margin":
        raise ValueError(
            f"Unsupported regional gravity map {region_name!r}. "
            "Supported values: 'norwegian_margin'."
        )

    default_raw_path = root / RAW_RELATIVE_DIR / RAW_FIXTURE_NAME
    raw_target = default_raw_path if raw_path is None else Path(raw_path).expanduser().resolve()
    processed_path = (root / PROCESSED_RELATIVE_DIR / PROCESSED_MAP_NAME).resolve()
    manifest_path = (root / PROCESSED_RELATIVE_DIR / PROCESSED_MANIFEST_NAME).resolve()

    if processed_path.exists() and manifest_path.exists() and not force_reprocess:
        return (
            GravityGridMap.from_npz(processed_path),
            load_regional_manifest(manifest_path),
            processed_path,
            manifest_path,
        )

    if not raw_target.exists():
        raise FileNotFoundError(
            "Norwegian-margin raw gravity grid not found. Expected a regular-grid CSV "
            f"at {raw_target}. Place a regional product there or pass --regional-map-raw-path."
        )

    map_model, manifest = process_regular_csv_gravity_map(
        raw_target,
        processed_npz_path=processed_path,
        manifest_path=manifest_path,
        region_name="norwegian_margin",
        source_name="bundled_fixture" if raw_target == default_raw_path else raw_target.stem,
        reference_height_m=0.0,
        default_method="linear",
        bounds_error=False,
        fill_value_mgal=np.nan,
        metadata_extra={
            "region_label": "Norwegian margin",
            "fixture": bool(raw_target == default_raw_path),
        },
        project_root=root,
    )
    return map_model, manifest, processed_path, manifest_path


def ensure_norwegian_margin_map(
    *,
    project_root: str | Path | None = None,
    raw_path: str | Path | None = None,
    force_reprocess: bool = False,
) -> tuple[GravityGridMap, RegionalGravityMapManifest, Path, Path]:
    """Convenience wrapper for the built-in Norwegian-margin region name."""
    return ensure_regional_gravity_map(
        "norwegian_margin",
        project_root=project_root,
        raw_path=raw_path,
        force_reprocess=force_reprocess,
    )


__all__ = [
    "RegionalGravityMapManifest",
    "ensure_norwegian_margin_map",
    "ensure_regional_gravity_map",
    "load_regular_csv_gravity_map",
    "load_regular_xyz_gravity_map",
    "load_regional_manifest",
    "process_regular_csv_gravity_map",
    "process_regular_xyz_gravity_map",
]
