from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gravnav.datasets.current_loader import (
    load_current_field_from_manifest,
    process_regular_csv_current_field,
)
from gravnav.datasets.magnetic_loader import (
    load_magnetic_grid_from_manifest,
    process_regular_csv_magnetic_grid,
)
from gravnav.estimators.error_state_ins import (
    ErrorStateINSNominalState,
    ErrorStateINSProcessNoise,
    ErrorStateINSState,
)
from gravnav.estimators.gravity_sequence_match import (
    GravitySequenceMatcher,
    GravitySequenceMatcherSpec,
)
from gravnav.estimators.map_match_pf import (
    apply_ned_offsets_to_geodetic,
    geodetic_offsets_to_local_ned,
)
from gravnav.physics.tides import TideCorrector, TideCorrectionSpec
from gravnav.sensors.current_profile import CurrentProfileSensor, CurrentProfileSensorSpec
from gravnav.sensors.magnetometer import MagnetometerSensorSpec, ScalarMagnetometerSensor
from gravnav.utils.config import load_config_mapping


def _make_state(
    *,
    time_s: float,
    lat_rad: float,
    lon_rad: float,
    height_m: float,
    v_ned_mps: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> ErrorStateINSState:
    nominal = ErrorStateINSNominalState(
        time_s=time_s,
        lat_rad=lat_rad,
        lon_rad=lon_rad,
        height_m=height_m,
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


def test_priority9_sensor_configs_load() -> None:
    root = Path(__file__).resolve().parents[1]
    mag = MagnetometerSensorSpec(
        **load_config_mapping(root / "configs/sensors/magnetometer_scalar.json")
    )
    current = CurrentProfileSensorSpec(
        **load_config_mapping(root / "configs/sensors/current_profile_sensor.json")
    )
    tide = TideCorrectionSpec(
        **load_config_mapping(root / "configs/environment/tide_correction_norway.json")
    )

    assert mag.noise_std_nt > 0.0
    assert current.max_speed_mps > 0.0
    assert len(tide.sea_surface_constituents) >= 1


def test_tide_corrector_and_public_loaders_round_trip(tmp_path: Path) -> None:
    magnetic_csv = tmp_path / "magnetic.csv"
    magnetic_csv.write_text(
        "\n".join(
            [
                "lat_deg,lon_deg,total_field_nt,anomaly_nt",
                "66.0,14.0,50000.0,15.0",
                "66.0,14.1,50010.0,20.0",
                "66.1,14.0,50020.0,25.0",
                "66.1,14.1,50030.0,30.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    current_csv = tmp_path / "current.csv"
    current_csv.write_text(
        "\n".join(
            [
                "lat_deg,lon_deg,depth_m,north_current_mps,east_current_mps",
                "66.0,14.0,20.0,0.10,0.05",
                "66.0,14.1,20.0,0.20,0.10",
                "66.1,14.0,20.0,0.30,0.15",
                "66.1,14.1,20.0,0.40,0.20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    magnetic_npz = tmp_path / "magnetic.npz"
    magnetic_manifest = tmp_path / "magnetic.json"
    current_npz = tmp_path / "current.npz"
    current_manifest = tmp_path / "current.json"

    process_regular_csv_magnetic_grid(
        magnetic_csv,
        region_name="fixture_mag",
        source_name="fixture",
        processed_map_path=magnetic_npz,
        manifest_path=magnetic_manifest,
        project_root=tmp_path,
    )
    process_regular_csv_current_field(
        current_csv,
        region_name="fixture_current",
        source_name="fixture",
        processed_grid_path=current_npz,
        manifest_path=current_manifest,
        project_root=tmp_path,
    )

    magnetic_grid, _, magnetic_grid_path, _ = load_magnetic_grid_from_manifest(
        magnetic_manifest
    )
    current_field, _, current_grid_path, _ = load_current_field_from_manifest(
        current_manifest
    )

    assert magnetic_grid_path == magnetic_npz.resolve()
    assert current_grid_path == current_npz.resolve()
    assert 50000.0 < float(magnetic_grid.evaluate_total_field_nt(66.05, 14.05)) < 50030.0
    current = np.asarray(
        current_field.evaluate_current_ned_mps(66.05, 14.05, 20.0),
        dtype=np.float64,
    )
    assert current.shape == (3,)
    assert 0.10 < float(current[0]) < 0.40
    assert 0.05 < float(current[1]) < 0.20

    tide_spec = TideCorrectionSpec(
        **json.loads(
            json.dumps(
                {
                    "name": "fixture_tide",
                    "sea_surface_constituents": [
                        {
                            "name": "M2",
                            "angular_frequency_rad_per_s": 0.0001405189,
                            "amplitude_at_equator": 0.5,
                            "phase_rad": 0.0,
                        }
                    ],
                    "ocean_loading_constituents": [
                        {
                            "name": "M2_load",
                            "angular_frequency_rad_per_s": 0.0001405189,
                            "amplitude_at_equator": 4.0e-07,
                        }
                    ],
                }
            )
        )
    )
    sample = TideCorrector(tide_spec).evaluate(lat_deg=66.0, lon_deg=14.0, time_s=600.0)
    assert not isinstance(sample, list)
    assert abs(sample.sea_surface_height_m) > 0.0
    assert abs(sample.total_gravity_correction_mps2) > 0.0


def test_magnetometer_and_current_profile_sensors_behave_reasonably() -> None:
    magnetometer = ScalarMagnetometerSensor(
        MagnetometerSensorSpec(
            noise_std_nt=0.0,
            turn_on_bias_std_nt=0.0,
            fixed_bias_nt=5.0,
            heading_disturbance_amplitude_nt=10.0,
            scale_factor_error_ppm=100.0,
        ),
        rng=np.random.default_rng(1),
    )
    meas = magnetometer.measure_total_field(50000.0, heading_rad=0.0, time_s=1.0)
    assert meas.value_nt != meas.ideal_value_nt
    assert meas.saturated is False

    current_sensor = CurrentProfileSensor(
        CurrentProfileSensorSpec(
            noise_std_mps=0.0,
            turn_on_bias_std_mps=0.0,
            fixed_bias_mps=(0.1, 0.0, 0.0),
            max_speed_mps=0.5,
        ),
        rng=np.random.default_rng(2),
    )
    current_meas = current_sensor.measure_current_profile_ned([1.0, 0.0, 0.0], time_s=1.0)
    assert current_meas.saturated is True
    assert float(np.linalg.norm(current_meas.value_ned_mps)) <= 0.5 + 1.0e-9


class _SimpleBathymetryMap:
    def __init__(self, lat_ref_rad: float, lon_ref_rad: float, height_ref_m: float) -> None:
        self.lat_ref_rad = lat_ref_rad
        self.lon_ref_rad = lon_ref_rad
        self.height_ref_m = height_ref_m

    def _ned(self, lat_deg, lon_deg):
        lat = np.asarray(np.deg2rad(lat_deg), dtype=np.float64)
        lon = np.asarray(np.deg2rad(lon_deg), dtype=np.float64)
        return geodetic_offsets_to_local_ned(
            lat,
            lon,
            np.zeros_like(lat),
            lat_ref_rad=self.lat_ref_rad,
            lon_ref_rad=self.lon_ref_rad,
            height_ref_m=self.height_ref_m,
        )

    def evaluate_water_depth_m(self, lat_deg, lon_deg, *, reference_surface_height_m=None):
        ned = self._ned(lat_deg, lon_deg)
        depth = 1200.0 + 0.015 * ned[..., 0] - 0.010 * ned[..., 1]
        return np.asarray(depth, dtype=np.float64)

    def evaluate_water_depth_gradient_m_per_m(self, lat_deg, lon_deg, *, reference_surface_height_m=None):
        lat = np.asarray(lat_deg, dtype=np.float64)
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat_b, lon_b = np.broadcast_arrays(lat, lon)
        grad = np.zeros(lat_b.shape + (2,), dtype=np.float64)
        grad[..., 0] = 0.015
        grad[..., 1] = -0.010
        return grad

    def evaluate_rugosity_m(self, lat_deg, lon_deg, *, reference_surface_height_m=None):
        lat = np.asarray(lat_deg, dtype=np.float64)
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat_b, lon_b = np.broadcast_arrays(lat, lon)
        return np.full(lat_b.shape, 1.5, dtype=np.float64)


class _SimpleMagneticMap:
    def __init__(self, lat_ref_rad: float, lon_ref_rad: float, height_ref_m: float) -> None:
        self.lat_ref_rad = lat_ref_rad
        self.lon_ref_rad = lon_ref_rad
        self.height_ref_m = height_ref_m

    def _ned(self, lat_deg, lon_deg):
        lat = np.asarray(np.deg2rad(lat_deg), dtype=np.float64)
        lon = np.asarray(np.deg2rad(lon_deg), dtype=np.float64)
        return geodetic_offsets_to_local_ned(
            lat,
            lon,
            np.zeros_like(lat),
            lat_ref_rad=self.lat_ref_rad,
            lon_ref_rad=self.lon_ref_rad,
            height_ref_m=self.height_ref_m,
        )

    def evaluate_total_field_nt(self, lat_deg, lon_deg):
        ned = self._ned(lat_deg, lon_deg)
        field = 50000.0 + 0.08 * ned[..., 0] + 0.04 * ned[..., 1]
        return np.asarray(field, dtype=np.float64)

    def evaluate_horizontal_gradient_nt_per_m(self, lat_deg, lon_deg):
        lat = np.asarray(lat_deg, dtype=np.float64)
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat_b, lon_b = np.broadcast_arrays(lat, lon)
        grad = np.zeros(lat_b.shape + (2,), dtype=np.float64)
        grad[..., 0] = 0.08
        grad[..., 1] = 0.04
        return grad


def test_sequence_matcher_supports_bathymetry_terrain_and_magnetics() -> None:
    lat0 = np.deg2rad(66.0)
    lon0 = np.deg2rad(14.0)
    h0 = 0.0

    def map_fn(lat_rad, lon_rad, height_m):
        ned = geodetic_offsets_to_local_ned(
            lat_rad,
            lon_rad,
            height_m,
            lat_ref_rad=lat0,
            lon_ref_rad=lon0,
            height_ref_m=h0,
        )
        return 2.0e-8 * ned[:, 0] - 1.0e-8 * ned[:, 1]

    matcher = GravitySequenceMatcher(
        GravitySequenceMatcherSpec(
            window_size=2,
            grid_half_span_m=(40.0, 40.0),
            grid_spacing_m=(10.0, 10.0),
            transition_std_m=(10.0, 10.0),
            center_prior_std_m=(30.0, 30.0),
            gravity_meas_std_mps2=2.0e-7,
            bathymetry_meas_std_m=1.0,
            bathymetry_gradient_meas_std_m_per_m=0.01,
            bathymetry_rugosity_meas_std_m=0.5,
            magnetic_meas_std_nt=2.0,
            magnetic_gradient_meas_std_nt_per_m=0.02,
        ),
        map_fn,
        bathymetry_map=_SimpleBathymetryMap(lat0, lon0, h0),
        magnetic_map=_SimpleMagneticMap(lat0, lon0, h0),
    )

    truth_offsets = np.array([[0.0, 0.0, 0.0], [15.0, 5.0, 0.0]], dtype=np.float64)
    ins_bias = np.array([20.0, -10.0, 0.0], dtype=np.float64)
    outputs = []
    for k, offset in enumerate(truth_offsets):
        lat_true, lon_true, h_true = apply_ned_offsets_to_geodetic(
            np.array([lat0], dtype=np.float64),
            np.array([lon0], dtype=np.float64),
            np.array([h0], dtype=np.float64),
            offset.reshape(1, 3),
        )
        lat_ins, lon_ins, h_ins = apply_ned_offsets_to_geodetic(
            np.array([lat0], dtype=np.float64),
            np.array([lon0], dtype=np.float64),
            np.array([h0], dtype=np.float64),
            (offset + ins_bias).reshape(1, 3),
        )

        g_meas = float(map_fn(lat_true, lon_true, h_true)[0])
        bathy_meas = float(
            np.asarray(
                matcher.bathymetry_map.evaluate_water_depth_m(
                    float(np.rad2deg(lat_true[0])),
                    float(np.rad2deg(lon_true[0])),
                ),
                dtype=np.float64,
            ).reshape(-1)[0]
        )
        magnetic_meas = float(
            np.asarray(
                matcher.magnetic_map.evaluate_total_field_nt(
                    float(np.rad2deg(lat_true[0])),
                    float(np.rad2deg(lon_true[0])),
                ),
                dtype=np.float64,
            ).reshape(-1)[0]
        )
        outputs.extend(
            matcher.update(
                g_meas,
                gravity_meas_std_mps2=2.0e-7,
                ins_or_state=_make_state(
                    time_s=float(k),
                    lat_rad=float(lat_ins[0]),
                    lon_rad=float(lon_ins[0]),
                    height_m=float(h_ins[0]),
                ),
                time_s=float(k),
                current_track_unit_ned=np.array([1.0, 0.0, 0.0], dtype=np.float64),
                measured_bathymetry_m=bathy_meas,
                bathymetry_meas_std_m=1.0,
                measured_bathymetry_gradient_m_per_m=0.015,
                bathymetry_gradient_meas_std_m_per_m=0.01,
                measured_bathymetry_rugosity_m=1.5,
                bathymetry_rugosity_meas_std_m=0.5,
                measured_magnetic_total_nt=magnetic_meas,
                magnetic_meas_std_nt=2.0,
                measured_magnetic_gradient_nt_per_m=0.08,
                magnetic_gradient_meas_std_nt_per_m=0.02,
            )
        )
    outputs.extend(matcher.finalize())

    assert outputs
    result = outputs[-1]
    assert result.used_bathymetry is True
    assert result.used_magnetics is True
    assert result.estimate.predicted_magnetic_total_nt is not None
    assert result.ambiguity_diagnostics.magnetic_information_ratio is not None
