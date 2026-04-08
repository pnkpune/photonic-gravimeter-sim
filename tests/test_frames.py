from __future__ import annotations

import numpy as np

from gravnav.physics.earth import geodetic_to_ecef
from gravnav.physics.frames import (
    LocalLevelFrame,
    dcm_ecef_to_ned,
    dcm_ecef_to_ned_from_ecef_position,
)


def test_scalar_ecef_position_helpers() -> None:
    lat_rad = np.deg2rad(14.5)
    lon_rad = np.deg2rad(74.0)
    height_m = -50.0

    x_m, y_m, z_m = geodetic_to_ecef(lat_rad, lon_rad, height_m)

    C_direct = dcm_ecef_to_ned(lat_rad, lon_rad)
    C_from_ecef = dcm_ecef_to_ned_from_ecef_position(x_m, y_m, z_m)
    frame = LocalLevelFrame.from_ecef_position(x_m, y_m, z_m)

    assert np.allclose(C_from_ecef, C_direct, atol=1.0e-12)
    assert np.isclose(frame.lat_rad, lat_rad, atol=1.0e-12)
    assert np.isclose(frame.lon_rad, lon_rad, atol=1.0e-12)
    assert np.isclose(frame.height_m, height_m, atol=1.0e-6)
