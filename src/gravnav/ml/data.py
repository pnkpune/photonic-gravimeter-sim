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
        clearance = np.full(candidate_lat_rad.shape, np.nan, dtype=np.float64)
        bathy_grad = np.full((candidate_lat_rad.size, 2), np.nan, dtype=np.float64)
        rugosity = np.full(candidate_lat_rad.shape, np.nan, dtype=np.float64)

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
        magnetic_total = np.full(candidate_lat_rad.shape, np.nan, dtype=np.float64)
        magnetic_grad = np.full((candidate_lat_rad.size, 2), np.nan, dtype=np.float64)

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
        tide_samples = [
            tide_corrector.evaluate(
                lat_deg=float(la),
                lon_deg=float(lo),
                time_s=float(time_s),
            )
            for la, lo in zip(lat_deg.reshape(-1), lon_deg.reshape(-1))
        ]
        tide_height = np.asarray(
            [sample.sea_surface_height_m for sample in tide_samples],
            dtype=np.float64,
        ).reshape(candidate_lat_rad.shape)
        tide_gravity = np.asarray(
            [sample.total_gravity_correction_mps2 for sample in tide_samples],
            dtype=np.float64,
        ).reshape(candidate_lat_rad.shape)

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

    for region_id, manifest_path in enumerate(demo_pack_manifests):
        pack = resolve_regional_demo_pack(manifest_path)
        tide_corrector = _tide_corrector_from_demo_pack(pack)
        scenario = ScenarioSpec.from_mapping(load_config_mapping(pack.scenario_path))
        truth = build_truth_trajectory_from_scenario(scenario)
        matcher = GravitySequenceMatcher(
            sequence_spec,
            pack.gravity_map,
            bathymetry_map=pack.bathymetry_grid,
            magnetic_map=pack.magnetic_grid,
        )

        query_history: list[FloatArray] = []
        offset_ned = rng.normal(
            0.0,
            float(spec.initial_offset_std_m),
            size=2,
        ).astype(np.float64)
        sample_indices = _sample_indices(
            len(truth.time_s),
            spec.max_examples_per_region,
            window_size=int(spec.window_size),
        )

        for k in range(len(truth.time_s)):
            lat_true = float(truth.lat_rad[k])
            lon_true = float(truth.lon_rad[k])
            height_true = float(truth.height_m[k])
            t_now = float(truth.time_s[k])
            v_now = np.asarray(truth.v_ned_mps[k], dtype=np.float64)
            offset_ned += rng.normal(
                0.0,
                float(spec.offset_random_walk_std_m),
                size=2,
            )
            prior_lat, prior_lon, prior_h = apply_ned_offsets_to_geodetic(
                np.array([lat_true], dtype=np.float64),
                np.array([lon_true], dtype=np.float64),
                np.array([height_true], dtype=np.float64),
                np.array([[offset_ned[0], offset_ned[1], 0.0]], dtype=np.float64),
            )
            track_unit = _horizontal_track_unit_ned(v_now)
            reference_surface_height_m = float(spec.reference_surface_height_m)
            if tide_corrector is not None:
                tide_sample = tide_corrector.evaluate(
                    lat_deg=np.rad2deg(lat_true),
                    lon_deg=np.rad2deg(lon_true),
                    time_s=t_now,
                )
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
            query_history.append(query_feature)
            if len(query_history) < spec.window_size or k not in sample_indices:
                continue

            state = _make_state(
                time_s=t_now,
                lat_rad=float(prior_lat[0]),
                lon_rad=float(prior_lon[0]),
                height_m=float(prior_h[0]),
                v_ned_mps=v_now,
            )

            matcher._active_grid_mode = "nominal"
            obs = matcher._build_observation(
                measured_disturbance_mps2=float(query_feature[0]),
                gravity_meas_std_mps2=float(sequence_spec.gravity_meas_std_mps2),
                ins_or_state=state,
                search_center_offset_ned_m=np.zeros(3, dtype=np.float64),
                current_track_unit_ned=track_unit,
                measured_gradient_per_s2=query_feature[1:3],
                gradient_meas_std_per_s2=sequence_spec.gradient_meas_std_per_s2,
                measured_bathymetry_m=(
                    None if not np.isfinite(query_feature[3]) else float(query_feature[3])
                ),
                bathymetry_meas_std_m=sequence_spec.bathymetry_meas_std_m,
                measured_bathymetry_gradient_m_per_m=(
                    None
                    if not np.isfinite(query_feature[4])
                    else float(np.dot(track_unit[:2], query_feature[4:6]))
                ),
                bathymetry_gradient_meas_std_m_per_m=(
                    sequence_spec.bathymetry_gradient_meas_std_m_per_m
                ),
                measured_bathymetry_rugosity_m=(
                    None if not np.isfinite(query_feature[6]) else float(query_feature[6])
                ),
                bathymetry_rugosity_meas_std_m=(
                    sequence_spec.bathymetry_rugosity_meas_std_m
                ),
                measured_magnetic_total_nt=(
                    None if not np.isfinite(query_feature[7]) else float(query_feature[7])
                ),
                magnetic_meas_std_nt=sequence_spec.magnetic_meas_std_nt,
                measured_magnetic_gradient_nt_per_m=(
                    None
                    if not np.isfinite(query_feature[8])
                    else float(np.dot(track_unit[:2], query_feature[8:10]))
                ),
                magnetic_gradient_meas_std_nt_per_m=(
                    sequence_spec.magnetic_gradient_meas_std_nt_per_m
                ),
                depth_measurement=None,
                reference_surface_height_m=reference_surface_height_m,
                time_s=t_now,
            )

            truth_offset = geodetic_offsets_to_local_ned(
                np.array([lat_true], dtype=np.float64),
                np.array([lon_true], dtype=np.float64),
                np.array([height_true], dtype=np.float64),
                lat_ref_rad=float(prior_lat[0]),
                lon_ref_rad=float(prior_lon[0]),
                height_ref_m=float(prior_h[0]),
            )[0]
            deltas = obs.candidate_offsets_ned_m[:, :2] - truth_offset[None, :2]
            label = int(np.argmin(np.sum(deltas**2, axis=1)))
            horizontal_error = float(np.linalg.norm(deltas[label]))
            nominal_cov = float(np.mean(sequence_spec.grid_spacing_m) ** 2)

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
            patch_tensors.append(patch)
            patch_summary_features.append(summarize_patch_tensor(patch))
            query_windows.append(
                np.stack(query_history[-spec.window_size :], axis=0).astype(np.float64)
            )
            candidate_features.append(
                _candidate_feature_matrix(
                    pack,
                    candidate_lat_rad=obs.candidate_lat_rad,
                    candidate_lon_rad=obs.candidate_lon_rad,
                    candidate_height_m=obs.candidate_height_m,
                    reference_surface_height_m=reference_surface_height_m,
                    current_track_unit_ned=track_unit,
                    time_s=t_now,
                    tide_corrector=tide_corrector,
                )
            )
            candidate_offsets.append(
                np.asarray(obs.candidate_offsets_ned_m, dtype=np.float64)
            )
            analytic_log_emission.append(np.asarray(obs.log_emission, dtype=np.float64))
            labels.append(label)
            truth_offsets.append(np.asarray(truth_offset, dtype=np.float64))
            publishability_labels.append(
                bool(horizontal_error <= max(0.5 * np.mean(sequence_spec.grid_spacing_m), 20.0))
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
        },
    )
