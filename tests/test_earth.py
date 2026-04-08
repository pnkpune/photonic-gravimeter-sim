from __future__ import annotations

import numpy as np

from gravnav.physics.earth import ecef_to_geodetic, geodetic_to_ecef


def test_scalar_geodetic_ecef_roundtrip() -> None:
    lat_rad = np.deg2rad(18.25)
    lon_rad = np.deg2rad(72.75)
    height_m = 123.4

    x_m, y_m, z_m = geodetic_to_ecef(lat_rad, lon_rad, height_m)
    lat_out, lon_out, height_out = ecef_to_geodetic(x_m, y_m, z_m)

    assert np.isclose(lat_out, lat_rad, atol=1.0e-12)
    assert np.isclose(lon_out, lon_rad, atol=1.0e-12)
    assert np.isclose(height_out, height_m, atol=1.0e-6)
