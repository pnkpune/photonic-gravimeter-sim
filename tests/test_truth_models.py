from __future__ import annotations

import numpy as np

from gravnav.truth.scenarios import ScenarioSpec


def test_constant_rate_turn_mapping_accepts_heading_rate_degps() -> None:
    scenario = ScenarioSpec.from_mapping(
        {
            "name": "degps_turn_case",
            "initial_lat_deg": 0.0,
            "initial_lon_deg": 0.0,
            "initial_height_m": 0.0,
            "initial_heading_deg": 0.0,
            "segments": [
                {
                    "type": "constant_rate_turn",
                    "duration_s": 10.0,
                    "speed_mps": 5.0,
                    "heading_rate_degps": 3.0,
                    "label": "turn",
                }
            ],
        }
    )

    segment = scenario.segments[0]
    assert np.isclose(segment.heading_rate_radps, np.deg2rad(3.0), atol=1.0e-15)
