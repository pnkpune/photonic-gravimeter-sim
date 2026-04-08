"""
nav_plots.py

Navigation-result plotting helpers for the gravity-aided navigation simulator.

Why this file exists
--------------------
The repository now has:
- truth trajectories with geodetic position, NED velocity, and body->NED attitude
- logged sensor histories
- logged INS, PF, and integrity estimator histories
- persistence helpers that expose these histories as NumPy-friendly arrays

What is still missing is a navigation-focused plotting layer that can:
1) compare truth and estimator trajectories,
2) visualize position / velocity / attitude histories,
3) show INS error growth against truth,
4) inspect PF and integrity diagnostics,
5) save figures for reports and notebooks.

Design notes
------------
- This module uses Matplotlib's explicit object-oriented API.
- It returns `(fig, ax)` or `(fig, axes)` from every plotting helper so callers
  can continue customizing the figure.
- It avoids hard-coding styling beyond simple line styles and labels.
- It accepts the repository's `ScenarioSimulationResult` directly.

Conventions
-----------
- Ground tracks are plotted either:
  - in geodetic latitude/longitude [deg], or
  - in local NED offsets [m] relative to the truth start.
- Position errors are always plotted in local NED:
      [dN, dE, dD]
- Attitude is shown as yaw-pitch-roll using the repository's standard 3-2-1
  body->NED convention.
- Signed depth uses:
      depth = h_ref - h
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..estimators.map_match_pf import geodetic_offsets_to_local_ned
from ..simulation.metrics import ins_position_error_history_from_truth
from ..simulation.results import ScenarioSimulationResult
from ..truth.trajectory import TruthTrajectory, ypr_from_dcm_body_to_ned

FloatArray = NDArray[np.float64]


# -----------------------------------------------------------------------------
# Low-level helpers
# -----------------------------------------------------------------------------


def _as_float_array(x: ArrayLike) -> FloatArray:
    """Convert input to a NumPy float64 array."""
    return np.asarray(x, dtype=np.float64)


def _wrap_angle_pi(angle_rad: ArrayLike) -> FloatArray:
    """
    Wrap angle(s) to [-pi, pi).
    """
    ang = _as_float_array(angle_rad)
    return ((ang + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float64)


def _rad2deg(x: ArrayLike) -> FloatArray:
    """Radians to degrees."""
    return np.rad2deg(_as_float_array(x)).astype(np.float64)


def _new_figure_and_axes(
    *,
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] = (7.5, 4.5),
    squeeze: bool = True,
    sharex: bool | str = False,
):
    """
    Create a constrained-layout figure and axes grid.
    """
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=squeeze,
        sharex=sharex,
        constrained_layout=True,
    )
    return fig, axes


def _ensure_result_has_truth(result: ScenarioSimulationResult) -> None:
    """
    Validate that the result contains at least one truth sample.
    """
    if len(result.truth) == 0:
        raise ValueError("ScenarioSimulationResult.truth is empty.")


def _truth_time_s(result: ScenarioSimulationResult) -> FloatArray:
    """
    Return truth time history [s].
    """
    _ensure_result_has_truth(result)
    return result.truth_time_s()


def _truth_origin(truth: TruthTrajectory) -> tuple[float, float, float]:
    """
    Return the first truth geodetic sample as `(lat0, lon0, h0)`.
    """
    if len(truth) == 0:
        raise ValueError("truth is empty.")
    return (
        float(truth.lat_rad[0]),
        float(truth.lon_rad[0]),
        float(truth.height_m[0]),
    )


def _truth_local_ned_offsets(truth: TruthTrajectory) -> FloatArray:
    """
    Return local NED truth offsets [m] relative to the truth start.
    """
    lat0, lon0, h0 = _truth_origin(truth)
    return geodetic_offsets_to_local_ned(
        truth.lat_rad,
        truth.lon_rad,
        truth.height_m,
        lat_ref_rad=lat0,
        lon_ref_rad=lon0,
        height_ref_m=h0,
    ).astype(np.float64)


def _safe_time_axis(
    raw_time_s: ArrayLike,
    *,
    truth_time_s: Optional[ArrayLike] = None,
) -> FloatArray:
    """
    Return a usable time axis.

    Strategy
    --------
    - If all values in `raw_time_s` are finite, use them.
    - Else if `truth_time_s` is provided and has at least the same length, use
      its leading segment.
    - Else fall back to sample index.
    """
    t = _as_float_array(raw_time_s).reshape(-1)
    if np.all(np.isfinite(t)):
        return t.astype(np.float64)

    if truth_time_s is not None:
        tt = _as_float_array(truth_time_s).reshape(-1)
        if tt.size >= t.size:
            return tt[: t.size].astype(np.float64)

    return np.arange(t.size, dtype=np.float64)


def _ypr_history_from_dcm_stack(C_hist: ArrayLike) -> FloatArray:
    """
    Convert a DCM history into yaw-pitch-roll history.

    Parameters
    ----------
    C_hist : array-like, shape (N, 3, 3)
        Body->NED DCM history.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Columns `[yaw, pitch, roll]` [rad].
    """
    C_arr = _as_float_array(C_hist)
    if C_arr.ndim != 3 or C_arr.shape[1:] != (3, 3):
        raise ValueError(f"C_hist must have shape (N, 3, 3), got {C_arr.shape}.")
    out = np.empty((C_arr.shape[0], 3), dtype=np.float64)
    for k in range(C_arr.shape[0]):
        out[k] = ypr_from_dcm_body_to_ned(C_arr[k])
    return out


def _ins_history_arrays(
    result: ScenarioSimulationResult,
) -> Optional[dict[str, np.ndarray]]:
    """
    Return INS history arrays, or None if no INS history exists.
    """
    if len(result.estimators.ins_states) == 0:
        return None
    return result.estimators.ins_history_arrays()


def _pf_history_arrays(
    result: ScenarioSimulationResult,
) -> Optional[dict[str, np.ndarray]]:
    """
    Return PF history arrays, or None if no PF history exists.
    """
    if len(result.estimators.pf_updates) == 0:
        return None
    return result.estimators.pf_history_arrays()


def _integrity_history_arrays(
    result: ScenarioSimulationResult,
) -> Optional[dict[str, np.ndarray]]:
    """
    Return integrity history arrays, or None if no integrity history exists.
    """
    if len(result.estimators.integrity_snapshots) == 0:
        return None
    return result.estimators.integrity_history_arrays()


def _ins_local_ned_offsets(
    result: ScenarioSimulationResult,
) -> Optional[FloatArray]:
    """
    Convert logged INS geodetic states into local NED offsets [m] relative to the
    truth start.
    """
    arrays = _ins_history_arrays(result)
    if arrays is None:
        return None

    lat0, lon0, h0 = _truth_origin(result.truth)
    return geodetic_offsets_to_local_ned(
        arrays["ins_lat_rad"],
        arrays["ins_lon_rad"],
        arrays["ins_height_m"],
        lat_ref_rad=lat0,
        lon_ref_rad=lon0,
        height_ref_m=h0,
    ).astype(np.float64)


def _pf_local_ned_offsets(
    result: ScenarioSimulationResult,
) -> Optional[FloatArray]:
    """
    Convert PF geodetic estimates into local NED offsets [m] relative to the
    truth start.
    """
    arrays = _pf_history_arrays(result)
    if arrays is None:
        return None

    lat0, lon0, h0 = _truth_origin(result.truth)
    return geodetic_offsets_to_local_ned(
        arrays["pf_lat_rad"],
        arrays["pf_lon_rad"],
        arrays["pf_height_m"],
        lat_ref_rad=lat0,
        lon_ref_rad=lon0,
        height_ref_m=h0,
    ).astype(np.float64)


def _plot_vec3_history(
    axes: Sequence[Axes],
    time_s: ArrayLike,
    values: ArrayLike,
    *,
    component_labels: Sequence[str],
    ylabel: str,
    series_label: Optional[str] = None,
    linestyle: str = "-",
) -> None:
    """
    Plot a 3-component history on three aligned axes.
    """
    t = _as_float_array(time_s).reshape(-1)
    x = _as_float_array(values)
    if x.shape != (t.size, 3):
        raise ValueError(
            f"values must have shape ({t.size}, 3), got {x.shape}."
        )
    if len(axes) != 3:
        raise ValueError("axes must contain exactly three Axes.")

    for k, ax in enumerate(axes):
        ax.plot(t, x[:, k], linestyle=linestyle, label=series_label)
        ax.set_ylabel(f"{component_labels[k]}\n{ylabel}")
        ax.grid(True, alpha=0.25)


def save_figure(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int = 150,
    transparent: bool = False,
) -> Path:
    """
    Save a Matplotlib figure to disk.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str or pathlib.Path
        Output file path.
    dpi : int, default=150
        Save DPI.
    transparent : bool, default=False
        Whether to save with transparent background.

    Returns
    -------
    pathlib.Path
        Resolved output path.
    """
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi), transparent=bool(transparent))
    return out


# -----------------------------------------------------------------------------
# Ground track plots
# -----------------------------------------------------------------------------


def plot_ground_track_geodetic(
    result: ScenarioSimulationResult,
    *,
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (7.2, 6.2),
    title: Optional[str] = None,
    show_ins: bool = True,
    show_pf: bool = True,
    show_start_finish: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot ground track in geodetic latitude/longitude.

    Parameters
    ----------
    result : ScenarioSimulationResult
        Run result.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into.
    figsize : tuple, default=(7.2, 6.2)
        Figure size used when `ax is None`.
    title : str, optional
        Title override.
    show_ins : bool, default=True
        Whether to overlay logged INS states.
    show_pf : bool, default=True
        Whether to overlay PF position estimates.
    show_start_finish : bool, default=True
        Whether to mark the truth start and end locations.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes.
    """
    _ensure_result_has_truth(result)
    if ax is None:
        fig, ax = _new_figure_and_axes(figsize=figsize)
    else:
        fig = ax.figure

    truth_lat_deg = _rad2deg(result.truth.lat_rad)
    truth_lon_deg = _rad2deg(result.truth.lon_rad)

    ax.plot(truth_lon_deg, truth_lat_deg, label="truth")

    if show_ins:
        ins = _ins_history_arrays(result)
        if ins is not None:
            ax.plot(
                _rad2deg(ins["ins_lon_rad"]),
                _rad2deg(ins["ins_lat_rad"]),
                linestyle="--",
                label="INS",
            )

    if show_pf:
        pf = _pf_history_arrays(result)
        if pf is not None:
            ax.plot(
                _rad2deg(pf["pf_lon_rad"]),
                _rad2deg(pf["pf_lat_rad"]),
                linestyle=":",
                label="PF",
            )

    if show_start_finish:
        ax.scatter(
            [truth_lon_deg[0], truth_lon_deg[-1]],
            [truth_lat_deg[0], truth_lat_deg[-1]],
            marker="o",
        )

    ax.set_xlabel("longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    ax.set_title("Ground track (geodetic)" if title is None else str(title))
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig, ax


def plot_ground_track_local_ned(
    result: ScenarioSimulationResult,
    *,
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (7.2, 6.2),
    title: Optional[str] = None,
    show_ins: bool = True,
    show_pf: bool = True,
    equal_aspect: bool = True,
    show_start_finish: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot ground track in local East/North coordinates.

    Parameters
    ----------
    result : ScenarioSimulationResult
        Run result.
    ax : matplotlib.axes.Axes, optional
        Existing axes.
    figsize : tuple, default=(7.2, 6.2)
        Figure size used when `ax is None`.
    title : str, optional
        Title override.
    show_ins : bool, default=True
        Whether to overlay logged INS positions.
    show_pf : bool, default=True
        Whether to overlay PF position estimates.
    equal_aspect : bool, default=True
        Whether to request equal axis scaling.
    show_start_finish : bool, default=True
        Whether to mark the truth start and end locations.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes.
    """
    _ensure_result_has_truth(result)
    truth_ned = _truth_local_ned_offsets(result.truth)

    if ax is None:
        fig, ax = _new_figure_and_axes(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(truth_ned[:, 1], truth_ned[:, 0], label="truth")

    if show_ins:
        ins_ned = _ins_local_ned_offsets(result)
        if ins_ned is not None:
            ax.plot(ins_ned[:, 1], ins_ned[:, 0], linestyle="--", label="INS")

    if show_pf:
        pf_ned = _pf_local_ned_offsets(result)
        if pf_ned is not None:
            ax.plot(pf_ned[:, 1], pf_ned[:, 0], linestyle=":", label="PF")

    if show_start_finish:
        ax.scatter(
            [truth_ned[0, 1], truth_ned[-1, 1]],
            [truth_ned[0, 0], truth_ned[-1, 0]],
            marker="o",
        )

    ax.set_xlabel("east offset [m]")
    ax.set_ylabel("north offset [m]")
    ax.set_title("Ground track (local NED)" if title is None else str(title))
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig, ax


# -----------------------------------------------------------------------------
# Vertical / velocity / attitude plots
# -----------------------------------------------------------------------------


def plot_altitude_history(
    result: ScenarioSimulationResult,
    *,
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (8.5, 4.5),
    title: Optional[str] = None,
    show_ins: bool = True,
    show_pf: bool = False,
) -> tuple[Figure, Axes]:
    """
    Plot ellipsoidal height history.

    Parameters
    ----------
    result : ScenarioSimulationResult
        Run result.
    ax : matplotlib.axes.Axes, optional
        Existing axes.
    figsize : tuple, default=(8.5, 4.5)
        Figure size used when `ax is None`.
    title : str, optional
        Title override.
    show_ins : bool, default=True
        Whether to overlay INS height.
    show_pf : bool, default=False
        Whether to overlay PF height estimates.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes.
    """
    tt = _truth_time_s(result)

    if ax is None:
        fig, ax = _new_figure_and_axes(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(tt, result.truth.height_m, label="truth")

    if show_ins:
        ins = _ins_history_arrays(result)
        if ins is not None:
            t_ins = _safe_time_axis(ins["ins_time_s"], truth_time_s=tt)
            ax.plot(t_ins, ins["ins_height_m"], linestyle="--", label="INS")

    if show_pf:
        pf = _pf_history_arrays(result)
        if pf is not None:
            t_pf = _safe_time_axis(pf["pf_time_s"], truth_time_s=tt)
            ax.plot(t_pf, pf["pf_height_m"], linestyle=":", label="PF")

    ax.set_xlabel("time [s]")
    ax.set_ylabel("height [m]")
    ax.set_title("Ellipsoidal height history" if title is None else str(title))
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig, ax


def plot_depth_history(
    result: ScenarioSimulationResult,
    *,
    reference_surface_height_m: float = 0.0,
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (8.5, 4.5),
    title: Optional[str] = None,
    show_measurements: bool = True,
    show_truth: bool = True,
    show_ideal_measurement: bool = False,
) -> tuple[Figure, Axes]:
    """
    Plot signed depth history.

    Parameters
    ----------
    result : ScenarioSimulationResult
        Run result.
    reference_surface_height_m : float, default=0.0
        Reference surface used to convert truth height to signed depth:
            depth = h_ref - h
    ax : matplotlib.axes.Axes, optional
        Existing axes.
    figsize : tuple, default=(8.5, 4.5)
        Figure size used when `ax is None`.
    title : str, optional
        Title override.
    show_measurements : bool, default=True
        Whether to overlay logged depth measurements.
    show_truth : bool, default=True
        Whether to show truth depth derived from the truth trajectory.
    show_ideal_measurement : bool, default=False
        Whether to show the ideal depth channel recorded in the depth sensor log.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes.
    """
    if ax is None:
        fig, ax = _new_figure_and_axes(figsize=figsize)
    else:
        fig = ax.figure

    tt = _truth_time_s(result)
    truth_depth = float(reference_surface_height_m) - _as_float_array(result.truth.height_m)

    if show_truth:
        ax.plot(tt, truth_depth, label="truth depth")

    if show_measurements and len(result.sensors.depth_samples) > 0:
        depth = result.sensors.depth_history_arrays()
        t_depth = _safe_time_axis(depth["depth_time_s"], truth_time_s=tt)
        ax.plot(t_depth, depth["depth_value_m"], linestyle="--", label="depth measurement")

        if show_ideal_measurement:
            ax.plot(
                t_depth,
                depth["depth_ideal_depth_m"],
                linestyle=":",
                label="depth ideal",
            )

    ax.set_xlabel("time [s]")
    ax.set_ylabel("signed depth [m]")
    ax.set_title("Depth history" if title is None else str(title))
    ax.grid(True, alpha=0.25)
    if show_truth or show_measurements:
        ax.legend()
    return fig, ax


def plot_velocity_ned(
    result: ScenarioSimulationResult,
    *,
    axes: Optional[Sequence[Axes]] = None,
    figsize: tuple[float, float] = (8.8, 7.2),
    title: Optional[str] = None,
    show_ins: bool = True,
    show_velocity_aid: bool = True,
) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    """
    Plot NED velocity components over time.

    Parameters
    ----------
    result : ScenarioSimulationResult
        Run result.
    axes : sequence of matplotlib.axes.Axes, optional
        Existing 3-axis stack.
    figsize : tuple, default=(8.8, 7.2)
        Figure size used when `axes is None`.
    title : str, optional
        Title override.
    show_ins : bool, default=True
        Whether to overlay INS velocity.
    show_velocity_aid : bool, default=True
        Whether to overlay velocity-aid measurements when they are logged in the
        NED frame.

    Returns
    -------
    tuple[Figure, tuple[Axes, Axes, Axes]]
        Figure and three axes.
    """
    tt = _truth_time_s(result)
    truth_v = result.truth_velocity_ned()

    if axes is None:
        fig, axes_arr = _new_figure_and_axes(
            nrows=3,
            ncols=1,
            figsize=figsize,
            squeeze=False,
            sharex=True,
        )
        ax_tuple = (axes_arr[0, 0], axes_arr[1, 0], axes_arr[2, 0])
    else:
        if len(axes) != 3:
            raise ValueError("axes must contain exactly three Axes.")
        ax_tuple = (axes[0], axes[1], axes[2])
        fig = ax_tuple[0].figure

    _plot_vec3_history(
        ax_tuple,
        tt,
        truth_v,
        component_labels=("north", "east", "down"),
        ylabel="[m/s]",
        series_label="truth",
        linestyle="-",
    )

    if show_ins:
        ins = _ins_history_arrays(result)
        if ins is not None:
            t_ins = _safe_time_axis(ins["ins_time_s"], truth_time_s=tt)
            _plot_vec3_history(
                ax_tuple,
                t_ins,
                ins["ins_v_ned_mps"],
                component_labels=("north", "east", "down"),
                ylabel="[m/s]",
                series_label="INS",
                linestyle="--",
            )

    if show_velocity_aid and len(result.sensors.velocity_aid_samples) > 0:
        vel = result.sensors.velocity_aid_history_arrays()
        frame = np.asarray(vel["velocity_aid_frame"], dtype=str)
        ned_mask = frame == "ned"
        if np.any(ned_mask):
            t_vel = _safe_time_axis(vel["velocity_aid_time_s"], truth_time_s=tt)[ned_mask]
            _plot_vec3_history(
                ax_tuple,
                t_vel,
                vel["velocity_aid_value_mps"][ned_mask],
                component_labels=("north", "east", "down"),
                ylabel="[m/s]",
                series_label="velocity aid",
                linestyle=":",
            )

    ax_tuple[-1].set_xlabel("time [s]")
    ax_tuple[0].set_title("Velocity history (NED)" if title is None else str(title))

    handles, labels = ax_tuple[0].get_legend_handles_labels()
    if len(handles) > 0:
        ax_tuple[0].legend(handles, labels)

    return fig, ax_tuple


def plot_attitude_ypr(
    result: ScenarioSimulationResult,
    *,
    axes: Optional[Sequence[Axes]] = None,
    figsize: tuple[float, float] = (8.8, 7.2),
    title: Optional[str] = None,
    degrees: bool = True,
    show_ins: bool = True,
) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    """
    Plot yaw, pitch, roll histories.

    Parameters
    ----------
    result : ScenarioSimulationResult
        Run result.
    axes : sequence of matplotlib.axes.Axes, optional
        Existing 3-axis stack.
    figsize : tuple, default=(8.8, 7.2)
        Figure size used when `axes is None`.
    title : str, optional
        Title override.
    degrees : bool, default=True
        If True, convert angles to degrees.
    show_ins : bool, default=True
        Whether to overlay INS attitude.

    Returns
    -------
    tuple[Figure, tuple[Axes, Axes, Axes]]
        Figure and three axes.
    """
    tt = _truth_time_s(result)
    truth_ypr = np.asarray(result.truth.yaw_pitch_roll_rad, dtype=np.float64)

    if axes is None:
        fig, axes_arr = _new_figure_and_axes(
            nrows=3,
            ncols=1,
            figsize=figsize,
            squeeze=False,
            sharex=True,
        )
        ax_tuple = (axes_arr[0, 0], axes_arr[1, 0], axes_arr[2, 0])
    else:
        if len(axes) != 3:
            raise ValueError("axes must contain exactly three Axes.")
        ax_tuple = (axes[0], axes[1], axes[2])
        fig = ax_tuple[0].figure

    unit = "[deg]" if degrees else "[rad]"
    truth_plot = _rad2deg(truth_ypr) if degrees else truth_ypr

    _plot_vec3_history(
        ax_tuple,
        tt,
        truth_plot,
        component_labels=("yaw", "pitch", "roll"),
        ylabel=unit,
        series_label="truth",
        linestyle="-",
    )

    if show_ins:
        ins = _ins_history_arrays(result)
        if ins is not None:
            t_ins = _safe_time_axis(ins["ins_time_s"], truth_time_s=tt)
            ins_ypr = _ypr_history_from_dcm_stack(ins["ins_C_n_b"])
            ins_ypr[:, 0] = _wrap_angle_pi(ins_ypr[:, 0])
            ins_ypr[:, 2] = _wrap_angle_pi(ins_ypr[:, 2])
            ins_plot = _rad2deg(ins_ypr) if degrees else ins_ypr

            _plot_vec3_history(
                ax_tuple,
                t_ins,
                ins_plot,
                component_labels=("yaw", "pitch", "roll"),
                ylabel=unit,
                series_label="INS",
                linestyle="--",
            )

    ax_tuple[-1].set_xlabel("time [s]")
    ax_tuple[0].set_title("Attitude history (YPR)" if title is None else str(title))

    handles, labels = ax_tuple[0].get_legend_handles_labels()
    if len(handles) > 0:
        ax_tuple[0].legend(handles, labels)

    return fig, ax_tuple


# -----------------------------------------------------------------------------
# Error and estimator-diagnostic plots
# -----------------------------------------------------------------------------


def plot_position_error_ned(
    result: ScenarioSimulationResult,
    *,
    axes: Optional[Sequence[Axes]] = None,
    figsize: tuple[float, float] = (9.0, 8.2),
    title: Optional[str] = None,
) -> tuple[Figure, tuple[Axes, Axes, Axes, Axes]]:
    """
    Plot INS position error in local NED coordinates against truth.

    Parameters
    ----------
    result : ScenarioSimulationResult
        Run result.
    axes : sequence of matplotlib.axes.Axes, optional
        Existing 4-axis stack.
    figsize : tuple, default=(9.0, 8.2)
        Figure size used when `axes is None`.
    title : str, optional
        Title override.

    Returns
    -------
    tuple[Figure, tuple[Axes, Axes, Axes, Axes]]
        Figure and four axes for `(dN, dE, dD, horizontal_norm)`.
    """
    if len(result.estimators.ins_states) == 0:
        raise ValueError("No INS state history is available in the result.")

    err_ned = ins_position_error_history_from_truth(
        result.truth,
        result.estimators.ins_states,
    )
    ins = result.estimators.ins_history_arrays()
    tt = _truth_time_s(result)
    t_ins = _safe_time_axis(ins["ins_time_s"], truth_time_s=tt)

    if axes is None:
        fig, axes_arr = _new_figure_and_axes(
            nrows=4,
            ncols=1,
            figsize=figsize,
            squeeze=False,
            sharex=True,
        )
        ax_tuple = (
            axes_arr[0, 0],
            axes_arr[1, 0],
            axes_arr[2, 0],
            axes_arr[3, 0],
        )
    else:
        if len(axes) != 4:
            raise ValueError("axes must contain exactly four Axes.")
        ax_tuple = (axes[0], axes[1], axes[2], axes[3])
        fig = ax_tuple[0].figure

    labels = ("dN", "dE", "dD")
    for k in range(3):
        ax_tuple[k].plot(t_ins, err_ned[:, k], label="INS error")
        ax_tuple[k].set_ylabel(f"{labels[k]}\n[m]")
        ax_tuple[k].grid(True, alpha=0.25)

    horiz = np.linalg.norm(err_ned[:, :2], axis=1)
    ax_tuple[3].plot(t_ins, horiz, label="horizontal")
    ax_tuple[3].set_ylabel("horizontal\n[m]")
    ax_tuple[3].set_xlabel("time [s]")
    ax_tuple[3].grid(True, alpha=0.25)

    ax_tuple[0].set_title("INS position error (NED)" if title is None else str(title))
    return fig, ax_tuple


def plot_pf_diagnostics(
    result: ScenarioSimulationResult,
    *,
    axes: Optional[Sequence[Axes]] = None,
    figsize: tuple[float, float] = (9.0, 8.0),
    title: Optional[str] = None,
) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    """
    Plot particle-filter diagnostics over time.

    Panels
    ------
    1) Effective sample size before/after update
    2) Predicted disturbance mean with +/-1 sigma band
    3) Horizontal PF offset in local NED

    Parameters
    ----------
    result : ScenarioSimulationResult
        Run result.
    axes : sequence of matplotlib.axes.Axes, optional
        Existing 3-axis stack.
    figsize : tuple, default=(9.0, 8.0)
        Figure size used when `axes is None`.
    title : str, optional
        Title override.

    Returns
    -------
    tuple[Figure, tuple[Axes, Axes, Axes]]
        Figure and three axes.
    """
    pf = _pf_history_arrays(result)
    if pf is None:
        raise ValueError("No PF updates are available in the result.")

    tt = _truth_time_s(result)
    t_pf = _safe_time_axis(pf["pf_time_s"], truth_time_s=tt)
    pf_ned = _pf_local_ned_offsets(result)

    if axes is None:
        fig, axes_arr = _new_figure_and_axes(
            nrows=3,
            ncols=1,
            figsize=figsize,
            squeeze=False,
            sharex=True,
        )
        ax_tuple = (axes_arr[0, 0], axes_arr[1, 0], axes_arr[2, 0])
    else:
        if len(axes) != 3:
            raise ValueError("axes must contain exactly three Axes.")
        ax_tuple = (axes[0], axes[1], axes[2])
        fig = ax_tuple[0].figure

    ax_tuple[0].plot(t_pf, pf["pf_effective_sample_size_before"], label="ESS before")
    ax_tuple[0].plot(t_pf, pf["pf_effective_sample_size_after"], linestyle="--", label="ESS after")
    ax_tuple[0].set_ylabel("ESS")
    ax_tuple[0].grid(True, alpha=0.25)
    ax_tuple[0].legend()

    mean_g = pf["pf_predicted_disturbance_mean_mps2"]
    std_g = pf["pf_predicted_disturbance_std_mps2"]
    ax_tuple[1].plot(t_pf, mean_g, label="predicted disturbance mean")
    ax_tuple[1].fill_between(
        t_pf,
        mean_g - std_g,
        mean_g + std_g,
        alpha=0.2,
        label="±1σ",
    )
    ax_tuple[1].set_ylabel("disturbance\n[m/s²]")
    ax_tuple[1].grid(True, alpha=0.25)
    ax_tuple[1].legend()

    if pf_ned is not None:
        horiz = np.linalg.norm(pf_ned[:, :2], axis=1)
        ax_tuple[2].plot(t_pf, horiz, label="PF horizontal offset")
        ax_tuple[2].set_ylabel("horizontal\n[m]")
        ax_tuple[2].legend()
    ax_tuple[2].set_xlabel("time [s]")
    ax_tuple[2].grid(True, alpha=0.25)

    ax_tuple[0].set_title("PF diagnostics" if title is None else str(title))
    return fig, ax_tuple


def plot_integrity_history(
    result: ScenarioSimulationResult,
    *,
    axes: Optional[Sequence[Axes]] = None,
    figsize: tuple[float, float] = (9.0, 8.0),
    title: Optional[str] = None,
) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    """
    Plot integrity-monitor history.

    Panels
    ------
    1) Horizontal error vs horizontal protection level
    2) Vertical error vs vertical protection level
    3) HMI flags over time

    Parameters
    ----------
    result : ScenarioSimulationResult
        Run result.
    axes : sequence of matplotlib.axes.Axes, optional
        Existing 3-axis stack.
    figsize : tuple, default=(9.0, 8.0)
        Figure size used when `axes is None`.
    title : str, optional
        Title override.

    Returns
    -------
    tuple[Figure, tuple[Axes, Axes, Axes]]
        Figure and three axes.
    """
    integ = _integrity_history_arrays(result)
    if integ is None:
        raise ValueError("No integrity snapshots are available in the result.")

    tt = _truth_time_s(result)
    t = _safe_time_axis(integ["integrity_time_s"], truth_time_s=tt)

    if axes is None:
        fig, axes_arr = _new_figure_and_axes(
            nrows=3,
            ncols=1,
            figsize=figsize,
            squeeze=False,
            sharex=True,
        )
        ax_tuple = (axes_arr[0, 0], axes_arr[1, 0], axes_arr[2, 0])
    else:
        if len(axes) != 3:
            raise ValueError("axes must contain exactly three Axes.")
        ax_tuple = (axes[0], axes[1], axes[2])
        fig = ax_tuple[0].figure

    ax_tuple[0].plot(
        t,
        integ["integrity_horizontal_error_m"],
        label="horizontal error",
    )
    ax_tuple[0].plot(
        t,
        integ["integrity_horizontal_protection_m"],
        linestyle="--",
        label="HPL",
    )
    ax_tuple[0].set_ylabel("[m]")
    ax_tuple[0].grid(True, alpha=0.25)
    ax_tuple[0].legend()

    ax_tuple[1].plot(
        t,
        integ["integrity_vertical_error_m"],
        label="vertical error",
    )
    ax_tuple[1].plot(
        t,
        integ["integrity_vertical_protection_m"],
        linestyle="--",
        label="VPL",
    )
    ax_tuple[1].set_ylabel("[m]")
    ax_tuple[1].grid(True, alpha=0.25)
    ax_tuple[1].legend()

    ax_tuple[2].step(
        t,
        integ["integrity_hmi_horizontal"].astype(np.float64),
        where="post",
        label="HMI horizontal",
    )
    ax_tuple[2].step(
        t,
        integ["integrity_hmi_vertical"].astype(np.float64),
        where="post",
        linestyle="--",
        label="HMI vertical",
    )
    ax_tuple[2].set_ylabel("flag")
    ax_tuple[2].set_xlabel("time [s]")
    ax_tuple[2].grid(True, alpha=0.25)
    ax_tuple[2].legend()

    ax_tuple[0].set_title("Integrity history" if title is None else str(title))
    return fig, ax_tuple


# -----------------------------------------------------------------------------
# Overview panel
# -----------------------------------------------------------------------------


def plot_navigation_overview(
    result: ScenarioSimulationResult,
    *,
    figsize: tuple[float, float] = (13.0, 10.0),
    title: Optional[str] = None,
    reference_surface_height_m: float = 0.0,
) -> tuple[Figure, NDArray[np.object_]]:
    """
    Plot a compact navigation overview dashboard.

    Layout
    ------
    2 x 3 panel:
    - local NED ground track
    - altitude history
    - depth history
    - velocity NED
    - attitude YPR
    - position error NED

    Parameters
    ----------
    result : ScenarioSimulationResult
        Run result.
    figsize : tuple, default=(13, 10)
        Figure size.
    title : str, optional
        Figure suptitle override.
    reference_surface_height_m : float, default=0.0
        Reference surface used for the depth panel.

    Returns
    -------
    tuple[Figure, np.ndarray]
        Figure and 2D axes array.
    """
    fig, axes = _new_figure_and_axes(
        nrows=2,
        ncols=3,
        figsize=figsize,
        squeeze=False,
    )

    plot_ground_track_local_ned(
        result,
        ax=axes[0, 0],
        title="Ground track",
    )
    plot_altitude_history(
        result,
        ax=axes[0, 1],
        title="Height",
    )
    plot_depth_history(
        result,
        ax=axes[0, 2],
        title="Depth",
        reference_surface_height_m=reference_surface_height_m,
    )

    # Lower row: create nested 3-axis stacks by replacing the placeholder axes.
    for j in range(3):
        axes[1, j].remove()

    gs = fig.add_gridspec(2, 3)

    vel_sub = gs[1, 0].subgridspec(3, 1, hspace=0.05)
    vel_axes = tuple(fig.add_subplot(vel_sub[k, 0]) for k in range(3))
    plot_velocity_ned(
        result,
        axes=vel_axes,
        title="Velocity",
    )

    att_sub = gs[1, 1].subgridspec(3, 1, hspace=0.05)
    att_axes = tuple(fig.add_subplot(att_sub[k, 0]) for k in range(3))
    plot_attitude_ypr(
        result,
        axes=att_axes,
        title="Attitude",
        degrees=True,
    )

    err_sub = gs[1, 2].subgridspec(4, 1, hspace=0.05)
    err_axes = tuple(fig.add_subplot(err_sub[k, 0]) for k in range(4))
    if len(result.estimators.ins_states) > 0:
        plot_position_error_ned(
            result,
            axes=err_axes,
            title="Position error",
        )
    else:
        for ax in err_axes:
            ax.text(
                0.5,
                0.5,
                "No INS history",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()

    if title is not None:
        fig.suptitle(str(title))

    # Return a dense object array mirroring the visual layout.
    out = np.empty((2, 3), dtype=object)
    out[0, 0] = axes[0, 0]
    out[0, 1] = axes[0, 1]
    out[0, 2] = axes[0, 2]
    out[1, 0] = vel_axes
    out[1, 1] = att_axes
    out[1, 2] = err_axes
    return fig, out


__all__ = [
    "FloatArray",
    "plot_altitude_history",
    "plot_attitude_ypr",
    "plot_depth_history",
    "plot_ground_track_geodetic",
    "plot_ground_track_local_ned",
    "plot_integrity_history",
    "plot_navigation_overview",
    "plot_pf_diagnostics",
    "plot_position_error_ned",
    "plot_velocity_ned",
    "save_figure",
]