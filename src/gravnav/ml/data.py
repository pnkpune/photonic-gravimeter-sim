"""
Corpus builder for learned Earth-signature localization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..datasets.bathymetry_loader import ResolvedRegionalDemoPack, resolve_regional_demo_pack
from ..estimators.error_state_ins import (
    ErrorStateINSNominalState,
    ErrorStateINSProcessNoise,
    ErrorStateINSState,
)
from ..estimators.gravity_sequence_match import GravitySequenceMatcher, GravitySequenceMatcherSpec
from ..estimators.map_match_pf import (
    apply_ned_offsets_to_geodetic,
    evaluate_gravity_map_disturbance,
    evaluate_gravity_map_horizontal_gradient,
    geodetic_offsets_to_local_ned,
)
from ..physics.tides import TideCorrector, TideCorrectionSpec
from ..truth.scenarios import ScenarioSpec, build_truth_trajectory_from_scenario
from ..utils.config import load_config_mapping

FloatArray = NDArray[np.float64]

PATCH_CHANNEL_NAMES = (
    "gravity_disturbance_mps2",
    "gravity_gradient_n_per_s2",
    "gravity_gradient_e_per_s2",
    "bathymetry_clearance_m",
    "bathymetry_gradient_n_m_per_m",
    "bathymetry_gradient_e_m_per_m",
    "bathymetry_rugosity_m",
    "magnetic_total_nt",
    "magnetic_gradient_n_nt_per_m",
    "magnetic_gradient_e_nt_per_m",
    "current_north_mps",
    "current_east_mps",
    "tide_surface_height_m",
    "tide_gravity_correction_mps2",
)

QUERY_FEATURE_NAMES = PATCH_CHANNEL_NAMES

CANDIDATE_FEATURE_NAMES = PATCH_CHANNEL_NAMES


@dataclass(frozen=True)
class _PrecomputedRegionSample:
    index: int
    time_s: float
    lat_rad: float
    lon_rad: float
    height_m: float
    v_ned_mps: FloatArray
    track_unit_ned: FloatArray
    reference_surface_height_m: float
    query_feature: FloatArray
    query_window: FloatArray
    patch_tensor: FloatArray
    patch_summary: FloatArray


@dataclass(frozen=True)
class _AcceptedRouteVariant:
    scenario: ScenarioSpec
    truth: Any
    translation_ned_m: FloatArray


def _as_float_array(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _make_state(
    *,
    time_s: float,
    lat_rad: float,
    lon_rad: float,
    height_m: float,
    v_ned_mps: ArrayLike,
) -> ErrorStateINSState:
    nominal = ErrorStateINSNominalState(
        time_s=float(time_s),
        lat_rad=float(lat_rad),
        lon_rad=float(lon_rad),
        height_m=float(height_m),
        v_ned_mps=np.asarray(v_ned_mps, dtype=np.float64),
        C_n_b=np.eye(3, dtype=np.float64),
        gyro_bias_radps=np.zeros(3, dtype=np.float64),
        accel_bias_mps2=np.zeros(3, dtype=np.float64),
    )
    return ErrorStateINSState(
        nominal=nominal,
        P=np.eye(15, dtype=np.float64),
        process_noise=ErrorStateINSProcessNoise.perfect(),
    )


def _patch_grid_offsets(
    *,
    patch_size: int,
    patch_spacing_m: float,
) -> FloatArray:
    half = patch_size // 2
    axis = np.arange(-half, half + 1, dtype=np.float64) * float(patch_spacing_m)
    north, east = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack(
        [
            north.reshape(-1),
            east.reshape(-1),
            np.zeros(north.size, dtype=np.float64),
        ]
    ).astype(np.float64)


def _horizontal_track_unit_ned(
    truth_v_ned_mps: ArrayLike,
) -> FloatArray:
    v = np.asarray(truth_v_ned_mps, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(v[:2]))
    if norm < 1.0e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return np.array([v[0] / norm, v[1] / norm, 0.0], dtype=np.float64)


def _tide_corrector_from_demo_pack(pack: ResolvedRegionalDemoPack) -> Optional[TideCorrector]:
    if pack.tide_config_path is None:
        return None
    mapping = load_config_mapping(pack.tide_config_path)
    return TideCorrector(TideCorrectionSpec(**mapping))


def _current_vector_from_pack(
    pack: ResolvedRegionalDemoPack,
    *,
    lat_deg: ArrayLike,
    lon_deg: ArrayLike,
    depth_m: ArrayLike,
) -> FloatArray:
    if pack.current_field is None:
        lat = np.asarray(lat_deg, dtype=np.float64)
        lat_b = np.broadcast_to(lat, np.broadcast(lat, lon_deg, depth_m).shape)
        return np.zeros(lat_b.shape + (3,), dtype=np.float64)
    return np.asarray(
        pack.current_field.evaluate_current_ned_mps(lat_deg, lon_deg, depth_m),
        dtype=np.float64,
    )


def _candidate_feature_matrix(
    pack: ResolvedRegionalDemoPack,
    *,
    candidate_lat_rad: FloatArray,
    candidate_lon_rad: FloatArray,
    candidate_height_m: FloatArray,
    reference_surface_height_m: float,
    current_track_unit_ned: FloatArray,
    time_s: float,
    tide_corrector: Optional[TideCorrector],
) -> FloatArray:
    lat_deg = np.rad2deg(candidate_lat_rad)
    lon_deg = np.rad2deg(candidate_lon_rad)
    gravity = evaluate_gravity_map_disturbance(
        pack.gravity_map,
        candidate_lat_rad,
        candidate_lon_rad,
        candidate_height_m,
    )
    gravity_grad = evaluate_gravity_map_horizontal_gradient(
        pack.gravity_map,
        candidate_lat_rad,
        candidate_lon_rad,
        candidate_height_m,
    ).T

    if pack.bathymetry_grid is not None:
        clearance = np.asarray(
            pack.bathymetry_grid.evaluate_water_depth_m(
                lat_deg,
                lon_deg,
                reference_surface_height_m=reference_surface_height_m,
            ),
            dtype=np.float64,
        )
        bathy_grad = np.asarray(
            pack.bathymetry_grid.evaluate_water_depth_gradient_m_per_m(
                lat_deg,
                lon_deg,
                reference_surface_height_m=reference_surface_height_m,
            ),
            dtype=np.float64,
        )
        rugosity = np.asarray(
            pack.bathymetry_grid.evaluate_rugosity_m(
                lat_deg,
                lon_deg,
                reference_surface_height_m=reference_surface_height_m,
            ),
            dtype=np.float64,
        )
    else:
        clearance = np.zeros(candidate_lat_rad.shape, dtype=np.float64)
        bathy_grad = np.zeros((candidate_lat_rad.size, 2), dtype=np.float64)
        rugosity = np.zeros(candidate_lat_rad.shape, dtype=np.float64)

    if pack.magnetic_grid is not None:
        magnetic_total = np.asarray(
            pack.magnetic_grid.evaluate_total_field_nt(lat_deg, lon_deg),
            dtype=np.float64,
        )
        magnetic_grad = np.asarray(
            pack.magnetic_grid.evaluate_horizontal_gradient_nt_per_m(lat_deg, lon_deg),
            dtype=np.float64,
        )
    else:
        magnetic_total = np.zeros(candidate_lat_rad.shape, dtype=np.float64)
        magnetic_grad = np.zeros((candidate_lat_rad.size, 2), dtype=np.float64)

    depth_m = np.maximum(0.0, reference_surface_height_m - candidate_height_m)
    current = _current_vector_from_pack(
        pack,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        depth_m=depth_m,
    )

    tide_height = np.zeros(candidate_lat_rad.shape, dtype=np.float64)
    tide_gravity = np.zeros(candidate_lat_rad.shape, dtype=np.float64)
    if tide_corrector is not None:
        tide_samples = tide_corrector.evaluate(
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            time_s=float(time_s),
        )
        if isinstance(tide_samples, list):
            tide_height = np.asarray(
                [sample.sea_surface_height_m for sample in tide_samples],
                dtype=np.float64,
            ).reshape(candidate_lat_rad.shape)
            tide_gravity = np.asarray(
                [sample.total_gravity_correction_mps2 for sample in tide_samples],
                dtype=np.float64,
            ).reshape(candidate_lat_rad.shape)
        else:
            tide_height = np.full(
                candidate_lat_rad.shape,
                float(tide_samples.sea_surface_height_m),
                dtype=np.float64,
            )
            tide_gravity = np.full(
                candidate_lat_rad.shape,
                float(tide_samples.total_gravity_correction_mps2),
                dtype=np.float64,
            )

    return np.column_stack(
        [
            gravity.reshape(-1),
            gravity_grad[:, 0],
            gravity_grad[:, 1],
            clearance.reshape(-1),
            bathy_grad[:, 0],
            bathy_grad[:, 1],
            rugosity.reshape(-1),
            magnetic_total.reshape(-1),
            magnetic_grad[:, 0],
            magnetic_grad[:, 1],
            current.reshape(-1, 3)[:, 0],
            current.reshape(-1, 3)[:, 1],
            tide_height.reshape(-1),
            tide_gravity.reshape(-1),
        ]
    ).astype(np.float64)


def _measurement_feature_vector(
    pack: ResolvedRegionalDemoPack,
    *,
    lat_rad: float,
    lon_rad: float,
    height_m: float,
    reference_surface_height_m: float,
    time_s: float,
    tide_corrector: Optional[TideCorrector],
) -> FloatArray:
    candidate = _candidate_feature_matrix(
        pack,
        candidate_lat_rad=np.array([lat_rad], dtype=np.float64),
        candidate_lon_rad=np.array([lon_rad], dtype=np.float64),
        candidate_height_m=np.array([height_m], dtype=np.float64),
        reference_surface_height_m=reference_surface_height_m,
        current_track_unit_ned=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        time_s=time_s,
        tide_corrector=tide_corrector,
    )
    return candidate.reshape(-1).astype(np.float64)


def _patch_tensor(
    pack: ResolvedRegionalDemoPack,
    *,
    lat_rad: float,
    lon_rad: float,
    height_m: float,
    reference_surface_height_m: float,
    time_s: float,
    patch_offsets_ned_m: FloatArray,
    tide_corrector: Optional[TideCorrector],
) -> FloatArray:
    num = patch_offsets_ned_m.shape[0]
    lat = np.full(num, float(lat_rad), dtype=np.float64)
    lon = np.full(num, float(lon_rad), dtype=np.float64)
    h = np.full(num, float(height_m), dtype=np.float64)
    lat_p, lon_p, h_p = apply_ned_offsets_to_geodetic(lat, lon, h, patch_offsets_ned_m)
    features = _candidate_feature_matrix(
        pack,
        candidate_lat_rad=lat_p,
        candidate_lon_rad=lon_p,
        candidate_height_m=h_p,
        reference_surface_height_m=reference_surface_height_m,
        current_track_unit_ned=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        time_s=time_s,
        tide_corrector=tide_corrector,
    )
    patch_size = int(round(np.sqrt(features.shape[0])))
    return features.reshape(patch_size, patch_size, features.shape[1]).astype(np.float64)


def summarize_patch_tensor(patch_tensor: FloatArray) -> FloatArray:
    patch = np.asarray(patch_tensor, dtype=np.float64)
    center = patch[patch.shape[0] // 2, patch.shape[1] // 2, :]
    mean = np.mean(patch, axis=(0, 1))
    std = np.std(patch, axis=(0, 1))
    return np.concatenate([center, mean, std], axis=0).astype(np.float64)


@dataclass
class RealOceanCorpusSpec:
    window_size: int = 9
    patch_size: int = 9
    patch_spacing_m: float = 40.0
    max_examples_per_region: int = 96
    num_offset_realizations_per_region: int = 1
    num_route_variants_per_region: int = 1
    route_variant_max_attempts: int = 24
    route_variant_margin_m: float = 250.0
    route_variant_min_separation_m: float = 1_000.0
    initial_offset_std_m: float = 90.0
    offset_random_walk_std_m: float = 6.0
    random_seed: int = 42
    reference_surface_height_m: float = 0.0
    name: str = "real_ocean_corpus"


@dataclass
class RealOceanCorpus:
    spec: RealOceanCorpusSpec
    patch_tensors: FloatArray
    patch_summary_features: FloatArray
    query_windows: FloatArray
    candidate_features: FloatArray
    candidate_offsets_ned_m: FloatArray
    analytic_log_emission: FloatArray
    labels: NDArray[np.int64]
    truth_offsets_ned_m: FloatArray
    publishability_labels: NDArray[np.bool_]
    covariance_targets: FloatArray
    region_names: tuple[str, ...]
    region_index: NDArray[np.int64]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_examples(self) -> int:
        return int(self.query_windows.shape[0])

    def save_npz(self, path: str | Path) -> Path:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            spec_json=np.array(json.dumps(_jsonable(asdict(self.spec)))),
            patch_tensors=np.asarray(self.patch_tensors, dtype=np.float64),
            patch_summary_features=np.asarray(
                self.patch_summary_features,
                dtype=np.float64,
            ),
            query_windows=np.asarray(self.query_windows, dtype=np.float64),
            candidate_features=np.asarray(self.candidate_features, dtype=np.float64),
            candidate_offsets_ned_m=np.asarray(
                self.candidate_offsets_ned_m,
                dtype=np.float64,
            ),
            analytic_log_emission=np.asarray(
                self.analytic_log_emission,
                dtype=np.float64,
            ),
            labels=np.asarray(self.labels, dtype=np.int64),
            truth_offsets_ned_m=np.asarray(self.truth_offsets_ned_m, dtype=np.float64),
            publishability_labels=np.asarray(self.publishability_labels, dtype=bool),
            covariance_targets=np.asarray(self.covariance_targets, dtype=np.float64),
            region_names=np.asarray(self.region_names, dtype=object),
            region_index=np.asarray(self.region_index, dtype=np.int64),
            metadata_json=np.array(json.dumps(_jsonable(self.metadata))),
        )
        return p

    @classmethod
    def from_npz(cls, path: str | Path) -> "RealOceanCorpus":
        p = Path(path).expanduser().resolve()
        with np.load(p, allow_pickle=True) as data:
            return cls(
                spec=RealOceanCorpusSpec(**json.loads(str(data["spec_json"].item()))),
                patch_tensors=np.asarray(data["patch_tensors"], dtype=np.float64),
                patch_summary_features=np.asarray(
                    data["patch_summary_features"],
                    dtype=np.float64,
                ),
                query_windows=np.asarray(data["query_windows"], dtype=np.float64),
                candidate_features=np.asarray(
                    data["candidate_features"],
                    dtype=np.float64,
                ),
                candidate_offsets_ned_m=np.asarray(
                    data["candidate_offsets_ned_m"],
                    dtype=np.float64,
                ),
                analytic_log_emission=np.asarray(
                    data["analytic_log_emission"],
                    dtype=np.float64,
                ),
                labels=np.asarray(data["labels"], dtype=np.int64),
                truth_offsets_ned_m=np.asarray(
                    data["truth_offsets_ned_m"],
                    dtype=np.float64,
                ),
                publishability_labels=np.asarray(
                    data["publishability_labels"],
                    dtype=bool,
                ),
                covariance_targets=np.asarray(
                    data["covariance_targets"],
                    dtype=np.float64,
                ),
                region_names=tuple(str(x) for x in data["region_names"].tolist()),
                region_index=np.asarray(data["region_index"], dtype=np.int64),
                metadata=json.loads(str(data["metadata_json"].item())),
            )

    def region_example_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for idx, name in enumerate(self.region_names):
            counts[str(name)] = int(np.sum(self.region_index == idx))
        return counts

    def subset(
        self,
        mask: ArrayLike,
        *,
        name: str | None = None,
    ) -> "RealOceanCorpus":
        keep = np.asarray(mask, dtype=bool).reshape(-1)
        if keep.shape[0] != self.num_examples:
            raise ValueError(
                f"subset mask length {keep.shape[0]} does not match corpus size {self.num_examples}."
            )
        if not np.any(keep):
            raise ValueError("subset mask selects no examples.")

        selected_region_names = [self.region_names[int(i)] for i in self.region_index[keep]]
        unique_regions = tuple(dict.fromkeys(selected_region_names).keys())
        region_name_to_idx = {region: idx for idx, region in enumerate(unique_regions)}
        normalized_region_index = np.asarray(
            [region_name_to_idx[name] for name in selected_region_names],
            dtype=np.int64,
        )
        metadata = dict(self.metadata)
        metadata["subset_name"] = str(name) if name is not None else "subset"
        metadata["subset_region_example_counts"] = {
            region: int(sum(1 for x in selected_region_names if x == region))
            for region in unique_regions
        }
        return RealOceanCorpus(
            spec=self.spec,
            patch_tensors=self.patch_tensors[keep],
            patch_summary_features=self.patch_summary_features[keep],
            query_windows=self.query_windows[keep],
            candidate_features=self.candidate_features[keep],
            candidate_offsets_ned_m=self.candidate_offsets_ned_m[keep],
            analytic_log_emission=self.analytic_log_emission[keep],
            labels=self.labels[keep],
            truth_offsets_ned_m=self.truth_offsets_ned_m[keep],
            publishability_labels=self.publishability_labels[keep],
            covariance_targets=self.covariance_targets[keep],
            region_names=unique_regions,
            region_index=normalized_region_index,
            metadata=metadata,
        )

    def select_regions(
        self,
        *,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        name: str | None = None,
    ) -> "RealOceanCorpus":
        if include is not None and exclude is not None:
            raise ValueError("select_regions accepts include or exclude, not both.")

        if include is not None:
            wanted = {str(x) for x in include}
            mask = np.asarray(
                [self.region_names[int(idx)] in wanted for idx in self.region_index],
                dtype=bool,
            )
            subset_name = name or f"include:{','.join(sorted(wanted))}"
            return self.subset(mask, name=subset_name)

        if exclude is not None:
            blocked = {str(x) for x in exclude}
            mask = np.asarray(
                [self.region_names[int(idx)] not in blocked for idx in self.region_index],
                dtype=bool,
            )
            subset_name = name or f"exclude:{','.join(sorted(blocked))}"
            return self.subset(mask, name=subset_name)

        return self


def load_real_ocean_corpus(path: str | Path) -> RealOceanCorpus:
    return RealOceanCorpus.from_npz(path)


def _sample_indices(length: int, count: int, *, window_size: int) -> NDArray[np.int64]:
    if length <= window_size:
        return np.arange(window_size - 1, length, dtype=np.int64)
    start = window_size - 1
    stop = max(start + 1, length)
    count = max(1, int(count))
    if stop - start <= count:
        return np.arange(start, stop, dtype=np.int64)
    return np.linspace(start, stop - 1, count, dtype=np.int64)


def _pack_supported_bounds_deg(
    pack: ResolvedRegionalDemoPack,
) -> tuple[float, float, float, float]:
    lat_min = float(pack.gravity_map.lat_axis_deg[0])
    lat_max = float(pack.gravity_map.lat_axis_deg[-1])
    lon_min = float(pack.gravity_map.lon_axis_deg[0])
    lon_max = float(pack.gravity_map.lon_axis_deg[-1])

    bounds_deg = [
        (
            float(pack.bathymetry_grid.lat_axis_deg[0]),
            float(pack.bathymetry_grid.lat_axis_deg[-1]),
            float(pack.bathymetry_grid.lon_axis_deg[0]),
            float(pack.bathymetry_grid.lon_axis_deg[-1]),
        )
    ]
    if pack.magnetic_grid is not None:
        bounds_deg.append(
            (
                float(pack.magnetic_grid.lat_axis_deg[0]),
                float(pack.magnetic_grid.lat_axis_deg[-1]),
                float(pack.magnetic_grid.lon_axis_deg[0]),
                float(pack.magnetic_grid.lon_axis_deg[-1]),
            )
        )
    if pack.current_manifest is not None:
        bounds_deg.append(
            (
                float(pack.current_manifest.lat_bounds_deg[0]),
                float(pack.current_manifest.lat_bounds_deg[1]),
                float(pack.current_manifest.lon_bounds_deg[0]),
                float(pack.current_manifest.lon_bounds_deg[1]),
            )
        )

    for b_lat_min, b_lat_max, b_lon_min, b_lon_max in bounds_deg:
        lat_min = max(lat_min, float(b_lat_min))
        lat_max = min(lat_max, float(b_lat_max))
        lon_min = max(lon_min, float(b_lon_min))
        lon_max = min(lon_max, float(b_lon_max))

    if lat_min >= lat_max or lon_min >= lon_max:
        raise ValueError(
            f"No common support overlap found for demo pack {pack.manifest.region_name!r}."
        )
    return lat_min, lat_max, lon_min, lon_max


def _trajectory_supported_by_pack(
    pack: ResolvedRegionalDemoPack,
    truth: Any,
) -> bool:
    lat_rad = np.asarray(truth.lat_rad, dtype=np.float64)
    lon_rad = np.asarray(truth.lon_rad, dtype=np.float64)
    lat_deg = np.rad2deg(lat_rad)
    lon_deg = np.rad2deg(lon_rad)

    if not np.all(pack.gravity_map.contains(lat_rad, lon_rad)):
        return False
    if not np.all(pack.bathymetry_grid.contains(lat_deg, lon_deg)):
        return False
    if pack.magnetic_grid is not None and not np.all(pack.magnetic_grid.contains(lat_deg, lon_deg)):
        return False
    if pack.current_manifest is not None:
        if np.any(lat_deg < float(pack.current_manifest.lat_bounds_deg[0])):
            return False
        if np.any(lat_deg > float(pack.current_manifest.lat_bounds_deg[1])):
            return False
        if np.any(lon_deg < float(pack.current_manifest.lon_bounds_deg[0])):
            return False
        if np.any(lon_deg > float(pack.current_manifest.lon_bounds_deg[1])):
            return False
    return True


def _route_translation_limits_ned_m(
    pack: ResolvedRegionalDemoPack,
    truth: Any,
    *,
    margin_m: float,
) -> tuple[float, float, float, float]:
    lat_min_deg, lat_max_deg, lon_min_deg, lon_max_deg = _pack_supported_bounds_deg(pack)
    lat0 = float(truth.lat_rad[0])
    lon0 = float(truth.lon_rad[0])
    h0 = float(truth.height_m[0])

    corners_lat_rad = np.deg2rad(
        np.asarray([lat_min_deg, lat_min_deg, lat_max_deg, lat_max_deg], dtype=np.float64)
    )
    corners_lon_rad = np.deg2rad(
        np.asarray([lon_min_deg, lon_max_deg, lon_min_deg, lon_max_deg], dtype=np.float64)
    )
    corners_height_m = np.full(4, h0, dtype=np.float64)
    support_offsets = geodetic_offsets_to_local_ned(
        corners_lat_rad,
        corners_lon_rad,
        corners_height_m,
        lat_ref_rad=lat0,
        lon_ref_rad=lon0,
        height_ref_m=h0,
    )
    route_offsets = geodetic_offsets_to_local_ned(
        np.asarray(truth.lat_rad, dtype=np.float64),
        np.asarray(truth.lon_rad, dtype=np.float64),
        np.asarray(truth.height_m, dtype=np.float64),
        lat_ref_rad=lat0,
        lon_ref_rad=lon0,
        height_ref_m=h0,
    )

    north_low = float(np.min(support_offsets[:, 0]) - np.min(route_offsets[:, 0]) + margin_m)
    north_high = float(np.max(support_offsets[:, 0]) - np.max(route_offsets[:, 0]) - margin_m)
    east_low = float(np.min(support_offsets[:, 1]) - np.min(route_offsets[:, 1]) + margin_m)
    east_high = float(np.max(support_offsets[:, 1]) - np.max(route_offsets[:, 1]) - margin_m)
    return north_low, north_high, east_low, east_high


def _translate_scenario_origin(
    scenario: ScenarioSpec,
    *,
    north_m: float,
    east_m: float,
    variant_id: int,
) -> ScenarioSpec:
    lat_new, lon_new, h_new = apply_ned_offsets_to_geodetic(
        np.array([scenario.initial_lat_rad], dtype=np.float64),
        np.array([scenario.initial_lon_rad], dtype=np.float64),
        np.array([scenario.initial_height_m], dtype=np.float64),
        np.array([[north_m, east_m, 0.0]], dtype=np.float64),
    )
    metadata = dict(scenario.metadata)
    metadata.update(
        {
            "source_scenario_name": scenario.name,
            "route_variant_id": int(variant_id),
            "route_translation_ned_m": [float(north_m), float(east_m), 0.0],
        }
    )
    return ScenarioSpec(
        name=f"{scenario.name}__route_variant_{variant_id}",
        initial_lat_rad=float(lat_new[0]),
        initial_lon_rad=float(lon_new[0]),
        initial_height_m=float(h_new[0]),
        initial_heading_rad=float(scenario.initial_heading_rad),
        segments=scenario.segments,
        default_dt_s=float(scenario.default_dt_s),
        description=str(scenario.description),
        metadata=metadata,
    )


def _build_route_variants_for_pack(
    pack: ResolvedRegionalDemoPack,
    *,
    base_scenario: ScenarioSpec,
    base_truth: Any,
    spec: RealOceanCorpusSpec,
    rng: np.random.Generator,
) -> list[_AcceptedRouteVariant]:
    variants = [
        _AcceptedRouteVariant(
            scenario=base_scenario,
            truth=base_truth,
            translation_ned_m=np.zeros(3, dtype=np.float64),
        )
    ]
    target = max(1, int(spec.num_route_variants_per_region))
    if target <= 1:
        return variants

    north_low, north_high, east_low, east_high = _route_translation_limits_ned_m(
        pack,
        base_truth,
        margin_m=float(spec.route_variant_margin_m),
    )
    if north_low > north_high or east_low > east_high:
        return variants

    accepted_offsets = [np.zeros(2, dtype=np.float64)]
    attempts = 0
    while len(variants) < target and attempts < int(spec.route_variant_max_attempts):
        attempts += 1
        north_m = float(rng.uniform(north_low, north_high))
        east_m = float(rng.uniform(east_low, east_high))
        candidate_offset = np.array([north_m, east_m], dtype=np.float64)
        if any(
            np.linalg.norm(candidate_offset - prev) < float(spec.route_variant_min_separation_m)
            for prev in accepted_offsets
        ):
            continue
        scenario_variant = _translate_scenario_origin(
            base_scenario,
            north_m=north_m,
            east_m=east_m,
            variant_id=len(variants),
        )
        truth_variant = build_truth_trajectory_from_scenario(scenario_variant)
        if not _trajectory_supported_by_pack(pack, truth_variant):
            continue
        accepted_offsets.append(candidate_offset)
        variants.append(
            _AcceptedRouteVariant(
                scenario=scenario_variant,
                truth=truth_variant,
                translation_ned_m=np.array([north_m, east_m, 0.0], dtype=np.float64),
            )
        )
    return variants


def _precompute_region_samples(
    *,
    pack: ResolvedRegionalDemoPack,
    truth: Any,
    spec: RealOceanCorpusSpec,
    sample_indices: NDArray[np.int64],
    patch_offsets: FloatArray,
    tide_corrector: Optional[TideCorrector],
) -> tuple[list[_PrecomputedRegionSample], int]:
    window_size = int(spec.window_size)
    query_features: list[FloatArray] = []
    sample_index_set = {int(idx) for idx in sample_indices.tolist()}
    samples: list[_PrecomputedRegionSample] = []

    for k in range(len(truth.time_s)):
        lat_true = float(truth.lat_rad[k])
        lon_true = float(truth.lon_rad[k])
        height_true = float(truth.height_m[k])
        t_now = float(truth.time_s[k])
        v_now = np.asarray(truth.v_ned_mps[k], dtype=np.float64)
        track_unit = _horizontal_track_unit_ned(v_now)

        reference_surface_height_m = float(spec.reference_surface_height_m)
        if tide_corrector is not None:
            tide_sample = tide_corrector.evaluate(
                lat_deg=np.rad2deg(lat_true),
                lon_deg=np.rad2deg(lon_true),
                time_s=t_now,
            )
            if isinstance(tide_sample, list):
                raise TypeError("Scalar tide sample expected for truth precomputation.")
            reference_surface_height_m += float(tide_sample.sea_surface_height_m)

        query_feature = _measurement_feature_vector(
            pack,
            lat_rad=lat_true,
            lon_rad=lon_true,
            height_m=height_true,
            reference_surface_height_m=reference_surface_height_m,
            time_s=t_now,
            tide_corrector=tide_corrector,
        )
        query_features.append(query_feature)
        if k < window_size - 1 or k not in sample_index_set:
            continue

        query_window = np.stack(
            query_features[-window_size:],
            axis=0,
        ).astype(np.float64)
        patch = _patch_tensor(
            pack,
            lat_rad=lat_true,
            lon_rad=lon_true,
            height_m=height_true,
            reference_surface_height_m=reference_surface_height_m,
            time_s=t_now,
            patch_offsets_ned_m=patch_offsets,
            tide_corrector=tide_corrector,
        )
        if not np.all(np.isfinite(query_window)):
            continue
        if not np.all(np.isfinite(patch)):
            continue
        samples.append(
            _PrecomputedRegionSample(
                index=int(k),
                time_s=t_now,
                lat_rad=lat_true,
                lon_rad=lon_true,
                height_m=height_true,
                v_ned_mps=v_now,
                track_unit_ned=track_unit,
                reference_surface_height_m=reference_surface_height_m,
                query_feature=np.asarray(query_feature, dtype=np.float64),
                query_window=query_window,
                patch_tensor=np.asarray(patch, dtype=np.float64),
                patch_summary=summarize_patch_tensor(patch),
            )
        )

    return samples, int(len(truth.time_s))


def build_real_ocean_corpus(
    demo_pack_manifests: Sequence[str | Path],
    *,
    sequence_spec: GravitySequenceMatcherSpec,
    corpus_spec: Optional[RealOceanCorpusSpec] = None,
) -> RealOceanCorpus:
    spec = RealOceanCorpusSpec() if corpus_spec is None else corpus_spec
    rng = np.random.default_rng(int(spec.random_seed))
    patch_offsets = _patch_grid_offsets(
        patch_size=int(spec.patch_size),
        patch_spacing_m=float(spec.patch_spacing_m),
    )

    patch_tensors: list[FloatArray] = []
    patch_summary_features: list[FloatArray] = []
    query_windows: list[FloatArray] = []
    candidate_features: list[FloatArray] = []
    candidate_offsets: list[FloatArray] = []
    analytic_log_emission: list[FloatArray] = []
    labels: list[int] = []
    truth_offsets: list[FloatArray] = []
    publishability_labels: list[bool] = []
    covariance_targets: list[float] = []
    region_names: list[str] = []
    region_index: list[int] = []
    manifest_paths_resolved: list[str] = []
    route_variant_counts: dict[str, int] = {}
    skipped_nonfinite_examples_by_region: dict[str, int] = {}

    for region_id, manifest_path in enumerate(demo_pack_manifests):
        pack = resolve_regional_demo_pack(manifest_path)
        skipped_nonfinite_examples_by_region.setdefault(pack.manifest.region_name, 0)
        manifest_paths_resolved.append(str(Path(manifest_path).expanduser().resolve()))
        tide_corrector = _tide_corrector_from_demo_pack(pack)
        scenario = ScenarioSpec.from_mapping(load_config_mapping(pack.scenario_path))
        truth = build_truth_trajectory_from_scenario(scenario)
        route_variants = _build_route_variants_for_pack(
            pack,
            base_scenario=scenario,
            base_truth=truth,
            spec=spec,
            rng=rng,
        )
        route_variant_counts[pack.manifest.region_name] = int(len(route_variants))
        matcher = GravitySequenceMatcher(
            sequence_spec,
            pack.gravity_map,
            bathymetry_map=pack.bathymetry_grid,
            magnetic_map=pack.magnetic_grid,
        )
        for route_variant in route_variants:
            sample_indices = _sample_indices(
                len(route_variant.truth.time_s),
                spec.max_examples_per_region,
                window_size=int(spec.window_size),
            )
            precomputed_samples, num_truth_steps = _precompute_region_samples(
                pack=pack,
                truth=route_variant.truth,
                spec=spec,
                sample_indices=sample_indices,
                patch_offsets=patch_offsets,
                tide_corrector=tide_corrector,
            )
            for realization_idx in range(int(spec.num_offset_realizations_per_region)):
                initial_offset_ned = rng.normal(
                    0.0,
                    float(spec.initial_offset_std_m),
                    size=2,
                ).astype(np.float64)
                offset_walk = rng.normal(
                    0.0,
                    float(spec.offset_random_walk_std_m),
                    size=(num_truth_steps, 2),
                ).astype(np.float64)
                offset_series_ned = initial_offset_ned[None, :] + np.cumsum(
                    offset_walk,
                    axis=0,
                )

                for sample in precomputed_samples:
                    offset_ned = offset_series_ned[sample.index]
                    prior_lat, prior_lon, prior_h = apply_ned_offsets_to_geodetic(
                        np.array([sample.lat_rad], dtype=np.float64),
                        np.array([sample.lon_rad], dtype=np.float64),
                        np.array([sample.height_m], dtype=np.float64),
                        np.array([[offset_ned[0], offset_ned[1], 0.0]], dtype=np.float64),
                    )

                    state = _make_state(
                        time_s=sample.time_s,
                        lat_rad=float(prior_lat[0]),
                        lon_rad=float(prior_lon[0]),
                        height_m=float(prior_h[0]),
                        v_ned_mps=sample.v_ned_mps,
                    )

                    matcher._active_grid_mode = "nominal"
                    obs = matcher._build_observation(
                        measured_disturbance_mps2=float(sample.query_feature[0]),
                        gravity_meas_std_mps2=float(sequence_spec.gravity_meas_std_mps2),
                        ins_or_state=state,
                        search_center_offset_ned_m=np.zeros(3, dtype=np.float64),
                        current_track_unit_ned=sample.track_unit_ned,
                        measured_gradient_per_s2=sample.query_feature[1:3],
                        gradient_meas_std_per_s2=sequence_spec.gradient_meas_std_per_s2,
                        measured_bathymetry_m=(
                            None
                            if pack.bathymetry_grid is None
                            else float(sample.query_feature[3])
                        ),
                        bathymetry_meas_std_m=sequence_spec.bathymetry_meas_std_m,
                        measured_bathymetry_gradient_m_per_m=(
                            None
                            if pack.bathymetry_grid is None
                            else float(
                                np.dot(
                                    sample.track_unit_ned[:2],
                                    sample.query_feature[4:6],
                                )
                            )
                        ),
                        bathymetry_gradient_meas_std_m_per_m=(
                            sequence_spec.bathymetry_gradient_meas_std_m_per_m
                        ),
                        measured_bathymetry_rugosity_m=(
                            None
                            if pack.bathymetry_grid is None
                            else float(sample.query_feature[6])
                        ),
                        bathymetry_rugosity_meas_std_m=(
                            sequence_spec.bathymetry_rugosity_meas_std_m
                        ),
                        measured_magnetic_total_nt=(
                            None
                            if pack.magnetic_grid is None
                            else float(sample.query_feature[7])
                        ),
                        magnetic_meas_std_nt=sequence_spec.magnetic_meas_std_nt,
                        measured_magnetic_gradient_nt_per_m=(
                            None
                            if pack.magnetic_grid is None
                            else float(
                                np.dot(
                                    sample.track_unit_ned[:2],
                                    sample.query_feature[8:10],
                                )
                            )
                        ),
                        magnetic_gradient_meas_std_nt_per_m=(
                            sequence_spec.magnetic_gradient_meas_std_nt_per_m
                        ),
                        depth_measurement=None,
                        reference_surface_height_m=sample.reference_surface_height_m,
                        time_s=sample.time_s,
                    )

                    truth_offset = geodetic_offsets_to_local_ned(
                        np.array([sample.lat_rad], dtype=np.float64),
                        np.array([sample.lon_rad], dtype=np.float64),
                        np.array([sample.height_m], dtype=np.float64),
                        lat_ref_rad=float(prior_lat[0]),
                        lon_ref_rad=float(prior_lon[0]),
                        height_ref_m=float(prior_h[0]),
                    )[0]
                    deltas = obs.candidate_offsets_ned_m[:, :2] - truth_offset[None, :2]
                    label = int(np.argmin(np.sum(deltas**2, axis=1)))
                    horizontal_error = float(np.linalg.norm(deltas[label]))
                    nominal_cov = float(np.mean(sequence_spec.grid_spacing_m) ** 2)
                    candidate_matrix = _candidate_feature_matrix(
                        pack,
                        candidate_lat_rad=obs.candidate_lat_rad,
                        candidate_lon_rad=obs.candidate_lon_rad,
                        candidate_height_m=obs.candidate_height_m,
                        reference_surface_height_m=sample.reference_surface_height_m,
                        current_track_unit_ned=sample.track_unit_ned,
                        time_s=sample.time_s,
                        tide_corrector=tide_corrector,
                    )
                    if not np.all(np.isfinite(candidate_matrix)):
                        skipped_nonfinite_examples_by_region[pack.manifest.region_name] += 1
                        continue
                    if not np.all(np.isfinite(obs.log_emission)):
                        skipped_nonfinite_examples_by_region[pack.manifest.region_name] += 1
                        continue

                    patch_tensors.append(sample.patch_tensor)
                    patch_summary_features.append(sample.patch_summary)
                    query_windows.append(sample.query_window)
                    candidate_features.append(candidate_matrix)
                    candidate_offsets.append(
                        np.asarray(obs.candidate_offsets_ned_m, dtype=np.float64)
                    )
                    analytic_log_emission.append(np.asarray(obs.log_emission, dtype=np.float64))
                    labels.append(label)
                    truth_offsets.append(np.asarray(truth_offset, dtype=np.float64))
                    publishability_labels.append(
                        bool(
                            horizontal_error
                            <= max(0.5 * np.mean(sequence_spec.grid_spacing_m), 20.0)
                        )
                    )
                    covariance_targets.append(
                        float(max(horizontal_error**2 / max(nominal_cov, 1.0e-9), 0.25))
                    )
                    region_names.append(pack.manifest.region_name)
                    region_index.append(int(region_id))

    if len(query_windows) == 0:
        raise ValueError("No corpus examples were produced.")

    unique_regions = tuple(dict.fromkeys(region_names).keys())
    region_name_to_idx = {name: idx for idx, name in enumerate(unique_regions)}
    normalized_region_index = np.asarray(
        [region_name_to_idx[name] for name in region_names],
        dtype=np.int64,
    )

    return RealOceanCorpus(
        spec=spec,
        patch_tensors=np.asarray(patch_tensors, dtype=np.float64),
        patch_summary_features=np.asarray(
            patch_summary_features,
            dtype=np.float64,
        ),
        query_windows=np.asarray(query_windows, dtype=np.float64),
        candidate_features=np.asarray(candidate_features, dtype=np.float64),
        candidate_offsets_ned_m=np.asarray(candidate_offsets, dtype=np.float64),
        analytic_log_emission=np.asarray(analytic_log_emission, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        truth_offsets_ned_m=np.asarray(truth_offsets, dtype=np.float64),
        publishability_labels=np.asarray(publishability_labels, dtype=bool),
        covariance_targets=np.asarray(covariance_targets, dtype=np.float64),
        region_names=unique_regions,
        region_index=normalized_region_index,
        metadata={
            "query_feature_names": list(QUERY_FEATURE_NAMES),
            "candidate_feature_names": list(CANDIDATE_FEATURE_NAMES),
            "patch_channel_names": list(PATCH_CHANNEL_NAMES),
            "num_regions": len(unique_regions),
            "manifest_paths": manifest_paths_resolved,
            "region_example_counts": {
                region: int(sum(1 for x in region_names if x == region))
                for region in unique_regions
            },
            "num_offset_realizations_per_region": int(
                spec.num_offset_realizations_per_region
            ),
            "route_variant_counts": route_variant_counts,
            "skipped_nonfinite_examples_by_region": skipped_nonfinite_examples_by_region,
        },
    )
