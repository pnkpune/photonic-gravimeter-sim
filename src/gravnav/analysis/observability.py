"""
observability.py

Quantitative observability analysis for gravity-aided navigation.

This module computes the local observability Gramian and derived metrics
to answer the central question: "How much position information does the
gravity measurement actually provide along this trajectory?"

Why this file exists
--------------------
Scalar gravity is a 1D measurement in 3D position space.  Whether map
matching can actually constrain position depends on:

1) The local gravity map gradient (how fast the map value changes with
   position).
2) The trajectory geometry (do successive measurements sample different
   gradient directions?).
3) The measurement noise level.

The observability Gramian captures all three in a single 3x3 matrix
whose eigenvalues indicate how much position information has been
accumulated along each principal direction.

Theory
------
At each measurement time k, the gravity map provides a linearized
scalar measurement of the form:

    z_k = delta_g(lat, lon, h) + eps_k

with Jacobian (the local gravity gradient):

    G_k = [d(delta_g)/d(lat), d(delta_g)/d(lon), d(delta_g)/d(h)]

measured in appropriate units (e.g., m/s^2 per radian or per metre).

The position-block observability Gramian accumulated over a window
of K measurements from time t_0 to t_K is:

    O = sum_{k=1}^{K}  Phi(t_0, t_k)^T  G_k^T  R_k^{-1}  G_k  Phi(t_0, t_k)

where Phi(t_0, t_k) is the position-block state transition matrix.

For short windows or when treating position as approximately constant
(since INS drift is slow relative to measurement rate), Phi ≈ I and:

    O ≈ sum_{k=1}^{K}  G_k^T  R_k^{-1}  G_k

The eigenvalues of O indicate:
- rank(O) = 3  : full 3D position observability
- rank(O) = 2  : position constrained in a plane
- rank(O) = 1  : position constrained along a line
- rank(O) = 0  : no position information from gravity

For a single scalar measurement, rank(G_k^T G_k) = 1 always.  You need
multiple measurements with different gradient directions to achieve
rank > 1.  This happens via:
- Trajectory curvature (turns sample different gradient orientations)
- Map curvature (gradient direction rotates along a straight path)
- Multi-modal measurements (gravity + gradient + magnetics each add rank)

Conventions
-----------
- Gradients are computed in local NED coordinates [m/s^2 per m].
- The Gramian is accumulated in NED position space [m^{-2} s^{-4}]
  (if using SI units for gradient and noise) or equivalently as a
  dimensionless information measure when normalized.
- Eigenvalues are sorted ascending (smallest first).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..estimators.map_match_pf import evaluate_gravity_map_disturbance
from ..physics.earth import meridian_radius, prime_vertical_radius

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Local gravity gradient computation
# ---------------------------------------------------------------------------


def gravity_map_gradient_ned(
    map_model: Any,
    lat_rad: float,
    lon_rad: float,
    height_m: float,
    *,
    delta_north_m: float = 50.0,
    delta_east_m: float = 50.0,
    delta_down_m: float = 5.0,
) -> FloatArray:
    """
    Compute the local gravity disturbance gradient in NED via central
    finite differences.

    Parameters
    ----------
    map_model : object
        Gravity map backend (same interface as used by the PF).
    lat_rad : float
        Latitude [rad].
    lon_rad : float
        Longitude [rad].
    height_m : float
        Ellipsoidal height [m].
    delta_north_m, delta_east_m, delta_down_m : float
        Step sizes for finite differences [m].

    Returns
    -------
    np.ndarray, shape (3,)
        Gradient [d(delta_g)/dN, d(delta_g)/dE, d(delta_g)/dD] in
        units of [m/s^2 per m].
    """
    phi = float(lat_rad)
    lam = float(lon_rad)
    h = float(height_m)

    M = float(meridian_radius(phi))
    N_pv = float(prime_vertical_radius(phi))
    cos_phi = max(abs(float(np.cos(phi))), 1.0e-8)

    # Convert NED step sizes to geodetic increments
    dlat = delta_north_m / (M + h)
    dlon = delta_east_m / ((N_pv + h) * cos_phi)
    dh = delta_down_m  # Down = -height, so dh in height space

    def _eval(la: float, lo: float, ht: float) -> float:
        return float(
            evaluate_gravity_map_disturbance(
                map_model,
                np.array([la]),
                np.array([lo]),
                np.array([ht]),
            )[0]
        )

    # Central differences
    g_north = (_eval(phi + dlat, lam, h) - _eval(phi - dlat, lam, h)) / (2.0 * delta_north_m)
    g_east = (_eval(phi, lam + dlon, h) - _eval(phi, lam - dlon, h)) / (2.0 * delta_east_m)
    g_down = (_eval(phi, lam, h - dh) - _eval(phi, lam, h + dh)) / (2.0 * delta_down_m)
    # Note: Down = -height, so d(delta_g)/dD = -d(delta_g)/dh
    # We want the gradient w.r.t. moving in the Down direction.
    # Moving Down by dD means height decreases by dD.

    return np.array([g_north, g_east, g_down], dtype=np.float64)


def gravity_map_gradient_norm(
    map_model: Any,
    lat_rad: float,
    lon_rad: float,
    height_m: float,
    **kwargs: Any,
) -> float:
    """
    Magnitude of the local horizontal gravity gradient.

    Returns
    -------
    float
        ||[dg/dN, dg/dE]|| in [m/s^2 per m].
    """
    grad = gravity_map_gradient_ned(map_model, lat_rad, lon_rad, height_m, **kwargs)
    return float(np.sqrt(grad[0] ** 2 + grad[1] ** 2))


# ---------------------------------------------------------------------------
# Observability Gramian
# ---------------------------------------------------------------------------


@dataclass
class ObservabilitySnapshot:
    """
    Observability analysis at a single trajectory point.

    Attributes
    ----------
    time_s : float
        Timestamp.
    gradient_ned : np.ndarray, shape (3,)
        Local gravity gradient in NED [m/s^2 per m].
    gradient_norm_horizontal : float
        Horizontal gradient magnitude.
    gramian_eigenvalues : np.ndarray, shape (3,)
        Eigenvalues of the accumulated Gramian, sorted ascending.
    gramian_eigenvectors : np.ndarray, shape (3, 3)
        Eigenvectors as columns.
    information_density : float
        Scalar summary of local navigation information (log-det of Gramian).
    observable_rank : int
        Effective rank of the Gramian (number of eigenvalues above threshold).
    feedback_recommended : bool
        Whether the observability is sufficient for feedback injection.
    """

    time_s: float
    gradient_ned: FloatArray
    gradient_norm_horizontal: float
    gramian_eigenvalues: FloatArray
    gramian_eigenvectors: FloatArray
    information_density: float
    observable_rank: int
    feedback_recommended: bool


def summarize_observability_snapshot(
    snapshot: ObservabilitySnapshot,
) -> dict[str, Any]:
    """
    Convert an observability snapshot into a JSON-friendly summary.

    Notes
    -----
    `information_density` can be non-finite during the earliest steps before the
    Gramian accumulates meaningful rank. Those cases are exported as ``None`` so
    downstream JSON consumers do not need to special-case infinities.
    """
    info_density = float(snapshot.information_density)
    if not np.isfinite(info_density):
        info_value: float | None = None
    else:
        info_value = info_density

    return {
        "time_s": float(snapshot.time_s),
        "gradient_ned_mps2_per_m": np.asarray(
            snapshot.gradient_ned,
            dtype=np.float64,
        ).tolist(),
        "gradient_norm_horizontal_mps2_per_m": float(
            snapshot.gradient_norm_horizontal,
        ),
        "gramian_eigenvalues": np.asarray(
            snapshot.gramian_eigenvalues,
            dtype=np.float64,
        ).tolist(),
        "gramian_eigenvectors": np.asarray(
            snapshot.gramian_eigenvectors,
            dtype=np.float64,
        ).tolist(),
        "information_density": info_value,
        "observable_rank": int(snapshot.observable_rank),
        "feedback_recommended": bool(snapshot.feedback_recommended),
    }


class ObservabilityAnalyzer:
    """
    Sliding-window observability Gramian tracker.

    Accumulates the position-block observability Gramian over a sliding
    window and provides real-time observability diagnostics.
    """

    def __init__(
        self,
        map_model: Any,
        *,
        window_size: int = 30,
        gravity_noise_std_mps2: float = 1.0e-5,
        eigenvalue_threshold: float = 1.0e-20,
        min_rank_for_feedback: int = 2,
        min_gradient_norm: float = 1.0e-10,
    ) -> None:
        """
        Parameters
        ----------
        map_model : object
            Gravity map backend.
        window_size : int
            Number of measurements in the sliding window.
        gravity_noise_std_mps2 : float
            Gravity measurement noise standard deviation [m/s^2].
        eigenvalue_threshold : float
            Eigenvalues below this are treated as zero.
        min_rank_for_feedback : int
            Minimum Gramian rank to recommend feedback (1, 2, or 3).
        min_gradient_norm : float
            Minimum horizontal gradient norm to count a measurement as
            informative.
        """
        self.map_model = map_model
        self.window_size = int(window_size)
        self.noise_variance = float(gravity_noise_std_mps2) ** 2
        self.eigenvalue_threshold = float(eigenvalue_threshold)
        self.min_rank_for_feedback = int(min_rank_for_feedback)
        self.min_gradient_norm = float(min_gradient_norm)

        # Sliding window of gradient outer products (G^T R^{-1} G terms)
        self._window: list[FloatArray] = []

        # Running accumulated Gramian
        self._gramian = np.zeros((3, 3), dtype=np.float64)

    @property
    def gramian(self) -> FloatArray:
        """Current accumulated observability Gramian."""
        return self._gramian.copy()

    def reset(self) -> None:
        """Clear the window and Gramian."""
        self._window.clear()
        self._gramian = np.zeros((3, 3), dtype=np.float64)

    def update(
        self,
        lat_rad: float,
        lon_rad: float,
        height_m: float,
        time_s: float,
        **gradient_kwargs: Any,
    ) -> ObservabilitySnapshot:
        """
        Add one measurement point and return observability diagnostics.

        Parameters
        ----------
        lat_rad, lon_rad, height_m : float
            Current position.
        time_s : float
            Timestamp.
        **gradient_kwargs
            Passed to `gravity_map_gradient_ned` (e.g., step sizes).

        Returns
        -------
        ObservabilitySnapshot
            Current observability state.
        """
        grad = gravity_map_gradient_ned(
            self.map_model,
            lat_rad,
            lon_rad,
            height_m,
            **gradient_kwargs,
        )
        grad_h_norm = float(np.sqrt(grad[0] ** 2 + grad[1] ** 2))

        # Compute the information contribution: G^T R^{-1} G
        # G is shape (1, 3), R is scalar noise_variance
        # G^T R^{-1} G = (1/sigma^2) * grad @ grad^T
        info_contribution = np.outer(grad, grad) / self.noise_variance

        # Add to sliding window
        self._window.append(info_contribution)
        self._gramian += info_contribution

        # Remove oldest if window exceeded
        if len(self._window) > self.window_size:
            oldest = self._window.pop(0)
            self._gramian -= oldest

        # Symmetrize for numerical stability
        self._gramian = 0.5 * (self._gramian + self._gramian.T)

        # Eigendecompose
        eigenvalues, eigenvectors = np.linalg.eigh(self._gramian)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Observable rank
        rank = int(np.sum(eigenvalues > self.eigenvalue_threshold))

        # Information density: log-det of Gramian (or sum of log eigenvalues)
        pos_eigs = eigenvalues[eigenvalues > self.eigenvalue_threshold]
        if len(pos_eigs) > 0:
            info_density = float(np.sum(np.log(pos_eigs)))
        else:
            info_density = -np.inf

        feedback_ok = rank >= self.min_rank_for_feedback and grad_h_norm >= self.min_gradient_norm

        return ObservabilitySnapshot(
            time_s=time_s,
            gradient_ned=grad,
            gradient_norm_horizontal=grad_h_norm,
            gramian_eigenvalues=eigenvalues,
            gramian_eigenvectors=eigenvectors,
            information_density=info_density,
            observable_rank=rank,
            feedback_recommended=feedback_ok,
        )


def compute_trajectory_observability(
    map_model: Any,
    lat_rad_array: ArrayLike,
    lon_rad_array: ArrayLike,
    height_m_array: ArrayLike,
    time_s_array: ArrayLike,
    *,
    window_size: int = 30,
    gravity_noise_std_mps2: float = 1.0e-5,
    **kwargs: Any,
) -> list[ObservabilitySnapshot]:
    """
    Compute observability snapshots along an entire trajectory.

    Parameters
    ----------
    map_model : object
        Gravity map backend.
    lat_rad_array, lon_rad_array, height_m_array : array-like
        Trajectory positions.
    time_s_array : array-like
        Timestamps.
    window_size : int
        Sliding window size.
    gravity_noise_std_mps2 : float
        Measurement noise standard deviation.

    Returns
    -------
    list[ObservabilitySnapshot]
        One snapshot per trajectory point.
    """
    lat = np.asarray(lat_rad_array, dtype=np.float64).ravel()
    lon = np.asarray(lon_rad_array, dtype=np.float64).ravel()
    h = np.asarray(height_m_array, dtype=np.float64).ravel()
    t = np.asarray(time_s_array, dtype=np.float64).ravel()

    n = lat.size
    if not (lon.size == n and h.size == n and t.size == n):
        raise ValueError("All input arrays must have the same length.")

    analyzer = ObservabilityAnalyzer(
        map_model,
        window_size=window_size,
        gravity_noise_std_mps2=gravity_noise_std_mps2,
        **kwargs,
    )

    results = []
    for k in range(n):
        snap = analyzer.update(
            lat_rad=float(lat[k]),
            lon_rad=float(lon[k]),
            height_m=float(h[k]),
            time_s=float(t[k]),
        )
        results.append(snap)

    return results


__all__ = [
    "ObservabilityAnalyzer",
    "ObservabilitySnapshot",
    "compute_trajectory_observability",
    "gravity_map_gradient_ned",
    "gravity_map_gradient_norm",
    "summarize_observability_snapshot",
]
