"""
tides.py

Compact tide and datum-correction helpers for Norway-first maritime demos.

This module deliberately stays lightweight and NumPy-only. It provides a small
harmonic correction surface that can be driven by tracked configuration files or
later replaced by higher-fidelity public-model backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike


def _as_float_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


@dataclass(frozen=True)
class TideConstituent:
    """
    One harmonic constituent used for sea-surface or gravity corrections.
    """

    name: str
    angular_frequency_rad_per_s: float
    amplitude_at_equator: float
    phase_rad: float = 0.0
    lat_scale_sin_power: float = 0.0
    lon_wavenumber: float = 0.0
    time_offset_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(
            self,
            "angular_frequency_rad_per_s",
            float(self.angular_frequency_rad_per_s),
        )
        object.__setattr__(self, "amplitude_at_equator", float(self.amplitude_at_equator))
        object.__setattr__(self, "phase_rad", float(self.phase_rad))
        object.__setattr__(self, "lat_scale_sin_power", float(self.lat_scale_sin_power))
        object.__setattr__(self, "lon_wavenumber", float(self.lon_wavenumber))
        object.__setattr__(self, "time_offset_s", float(self.time_offset_s))

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "TideConstituent":
        return cls(
            name=str(mapping["name"]),
            angular_frequency_rad_per_s=float(mapping["angular_frequency_rad_per_s"]),
            amplitude_at_equator=float(mapping["amplitude_at_equator"]),
            phase_rad=float(mapping.get("phase_rad", 0.0)),
            lat_scale_sin_power=float(mapping.get("lat_scale_sin_power", 0.0)),
            lon_wavenumber=float(mapping.get("lon_wavenumber", 0.0)),
            time_offset_s=float(mapping.get("time_offset_s", 0.0)),
        )


@dataclass(frozen=True)
class TideCorrectionSample:
    """
    Tide/datum correction evaluated at one space-time point.
    """

    time_s: float
    sea_surface_height_m: float
    ocean_loading_gravity_mps2: float
    solid_earth_gravity_mps2: float
    total_gravity_correction_mps2: float


@dataclass
class TideCorrectionSpec:
    """
    Lightweight harmonic tide/datum correction model.

    Sea-surface-height terms shift the effective ocean reference surface.
    Gravity terms are additive corrections to the gravimeter measurement.
    """

    name: str = "tide_correction"
    sea_surface_constituents: tuple[TideConstituent, ...] = field(default_factory=tuple)
    ocean_loading_constituents: tuple[TideConstituent, ...] = field(default_factory=tuple)
    solid_earth_constituents: tuple[TideConstituent, ...] = field(default_factory=tuple)
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.sea_surface_constituents = tuple(
            c if isinstance(c, TideConstituent) else TideConstituent.from_mapping(c)
            for c in self.sea_surface_constituents
        )
        self.ocean_loading_constituents = tuple(
            c if isinstance(c, TideConstituent) else TideConstituent.from_mapping(c)
            for c in self.ocean_loading_constituents
        )
        self.solid_earth_constituents = tuple(
            c if isinstance(c, TideConstituent) else TideConstituent.from_mapping(c)
            for c in self.solid_earth_constituents
        )
        self.notes = [str(x) for x in self.notes]
        self.metadata = dict(self.metadata)


def _evaluate_constituent_sum(
    constituents: Sequence[TideConstituent],
    *,
    lat_deg: ArrayLike,
    lon_deg: ArrayLike,
    time_s: float,
) -> np.ndarray:
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    lat_b, lon_b = np.broadcast_arrays(lat, lon)
    out = np.zeros_like(lat_b, dtype=np.float64)
    lat_scale_base = np.abs(np.sin(np.deg2rad(lat_b)))
    lon_rad = np.deg2rad(lon_b)
    t = float(time_s)
    for c in constituents:
        lat_scale = np.power(lat_scale_base, c.lat_scale_sin_power)
        phase = (
            c.angular_frequency_rad_per_s * (t + c.time_offset_s)
            + c.lon_wavenumber * lon_rad
            + c.phase_rad
        )
        out += c.amplitude_at_equator * lat_scale * np.cos(phase)
    return out


class TideCorrector:
    """
    Evaluate harmonic sea-surface and gravity tide corrections.
    """

    def __init__(self, spec: TideCorrectionSpec) -> None:
        self.spec = spec

    def evaluate(
        self,
        *,
        lat_deg: ArrayLike,
        lon_deg: ArrayLike,
        time_s: float,
    ) -> TideCorrectionSample | list[TideCorrectionSample]:
        lat = np.asarray(lat_deg, dtype=np.float64)
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat_b, lon_b = np.broadcast_arrays(lat, lon)
        ssh = _evaluate_constituent_sum(
            self.spec.sea_surface_constituents,
            lat_deg=lat_b,
            lon_deg=lon_b,
            time_s=time_s,
        )
        ocean_loading = _evaluate_constituent_sum(
            self.spec.ocean_loading_constituents,
            lat_deg=lat_b,
            lon_deg=lon_b,
            time_s=time_s,
        )
        solid_earth = _evaluate_constituent_sum(
            self.spec.solid_earth_constituents,
            lat_deg=lat_b,
            lon_deg=lon_b,
            time_s=time_s,
        )
        total = ocean_loading + solid_earth
        if ssh.ndim == 0:
            return TideCorrectionSample(
                time_s=float(time_s),
                sea_surface_height_m=float(ssh),
                ocean_loading_gravity_mps2=float(ocean_loading),
                solid_earth_gravity_mps2=float(solid_earth),
                total_gravity_correction_mps2=float(total),
            )
        samples: list[TideCorrectionSample] = []
        for idx in np.ndindex(ssh.shape):
            samples.append(
                TideCorrectionSample(
                    time_s=float(time_s),
                    sea_surface_height_m=float(ssh[idx]),
                    ocean_loading_gravity_mps2=float(ocean_loading[idx]),
                    solid_earth_gravity_mps2=float(solid_earth[idx]),
                    total_gravity_correction_mps2=float(total[idx]),
                )
            )
        return samples

    def effective_reference_surface_height_m(
        self,
        *,
        base_reference_surface_height_m: float,
        lat_deg: float,
        lon_deg: float,
        time_s: float,
    ) -> float:
        sample = self.evaluate(lat_deg=lat_deg, lon_deg=lon_deg, time_s=time_s)
        if isinstance(sample, list):
            raise TypeError("Scalar lat/lon inputs are required.")
        return float(base_reference_surface_height_m) + float(sample.sea_surface_height_m)

    def gravity_correction_mps2(
        self,
        *,
        lat_deg: float,
        lon_deg: float,
        time_s: float,
    ) -> float:
        sample = self.evaluate(lat_deg=lat_deg, lon_deg=lon_deg, time_s=time_s)
        if isinstance(sample, list):
            raise TypeError("Scalar lat/lon inputs are required.")
        return float(sample.total_gravity_correction_mps2)


__all__ = [
    "TideConstituent",
    "TideCorrectionSample",
    "TideCorrectionSpec",
    "TideCorrector",
]
