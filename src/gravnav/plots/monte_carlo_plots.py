"""
monte_carlo_plots.py

Monte Carlo plotting helpers for the gravity-aided navigation simulator.

Why this file exists
--------------------
The simulation stack now has:
- one-run result containers
- one-run metric summaries
- Monte Carlo orchestration and aggregate summaries

What is still missing is a plotting layer that can:
1) inspect per-run metric distributions,
2) compare studies against each other,
3) visualize aggregate summaries,
4) save publication-friendly figures without depending on notebooks.

Design notes
------------
- This module intentionally uses Matplotlib's explicit object-oriented API.
- It stays NumPy + Matplotlib only.
- It accepts the repository's Monte Carlo result/summary objects directly.
- It does not mutate the study objects it plots.
- Missing / unavailable metrics raise clear KeyError or ValueError exceptions.

Conventions
-----------
- Metric names are the flattened scalar keys produced by:
      MonteCarloStudyResult.metric_series()
  Examples include:
      "ins_position_error.horizontal_rmse_m"
      "gravimeter_error.rmse"
      "integrity.fraction_nis_passed"

- Figure-returning helpers all return `(fig, ax)` or `(fig, axes)` so callers can
  further customize titles, labels, or save the figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..simulation.monte_carlo import (
    MonteCarloScalarAggregate,
    MonteCarloStudyResult,
    MonteCarloStudySummary,
)

FloatArray = NDArray[np.float64]


# -----------------------------------------------------------------------------
# Low-level helpers
# -----------------------------------------------------------------------------


def _as_float_array(x: ArrayLike) -> FloatArray:
    """Convert input to a NumPy float64 array."""
    return np.asarray(x, dtype=np.float64)


def _finite_1d(x: ArrayLike, *, name: str) -> FloatArray:
    """
    Return a flattened finite-valued 1D array.

    Parameters
    ----------
    x : array-like
        Input values.
    name : str
        Name for error messages.

    Returns
    -------
    np.ndarray, shape (N,)
        Finite values only.

    Raises
    ------
    ValueError
        If no finite values remain.
    """
    arr = _as_float_array(x).reshape(-1)
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        raise ValueError(f"{name} contains no finite values.")
    return vals


def _metric_display_name(metric_name: str) -> str:
    """
    Convert a flattened metric key into a readable plot label.

    Example
    -------
        "ins_position_error.horizontal_rmse_m"
            -> "ins position error / horizontal rmse m"
    """
    return str(metric_name).replace(".", " / ").replace("_", " ")


def _new_figure_and_axes(
    *,
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] = (7.5, 4.5),
    squeeze: bool = True,
):
    """
    Create a constrained-layout Matplotlib figure and axes grid.
    """
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=squeeze,
        constrained_layout=True,
    )
    return fig, axes


def _coerce_summary(
    study_or_summary: MonteCarloStudyResult | MonteCarloStudySummary,
) -> MonteCarloStudySummary:
    """
    Return a `MonteCarloStudySummary` from either a full study result or summary.
    """
    if isinstance(study_or_summary, MonteCarloStudySummary):
        return study_or_summary
    if isinstance(study_or_summary, MonteCarloStudyResult):
        return study_or_summary.summary()
    raise TypeError(
        "Expected MonteCarloStudyResult or MonteCarloStudySummary, "
        f"got {type(study_or_summary).__name__}."
    )


def _metric_aggregate(
    study_or_summary: MonteCarloStudyResult | MonteCarloStudySummary,
    metric_name: str,
) -> MonteCarloScalarAggregate:
    """
    Return one scalar aggregate by metric name.
    """
    summary = _coerce_summary(study_or_summary)
    agg = summary.metric_aggregates.get(str(metric_name))
    if agg is None:
        available = sorted(summary.metric_aggregates.keys())
        preview = ", ".join(available[:10])
        more = "" if len(available) <= 10 else ", ..."
        raise KeyError(
            f"Metric {metric_name!r} not present in study summary. "
            f"Available metrics include: {preview}{more}"
        )
    return agg


def _metric_series(study: MonteCarloStudyResult, metric_name: str) -> FloatArray:
    """
    Extract one scalar metric series from a study result.

    Parameters
    ----------
    study : MonteCarloStudyResult
        Full Monte Carlo study result.
    metric_name : str
        Flattened metric name.

    Returns
    -------
    np.ndarray, shape (N,)
        Finite scalar values across successful runs.

    Raises
    ------
    KeyError
        If the metric is absent.
    ValueError
        If the metric exists but contains no finite values.
    """
    series = study.metric_series()
    if metric_name not in series:
        available = sorted(series.keys())
        preview = ", ".join(available[:10])
        more = "" if len(available) <= 10 else ", ..."
        raise KeyError(
            f"Metric {metric_name!r} not present in study result. "
            f"Available metrics include: {preview}{more}"
        )
    return _finite_1d(series[metric_name], name=metric_name)


def _named_studies(
    studies: (
        Mapping[str, MonteCarloStudyResult | MonteCarloStudySummary]
        | Sequence[MonteCarloStudyResult | MonteCarloStudySummary]
    ),
) -> list[tuple[str, MonteCarloStudyResult | MonteCarloStudySummary]]:
    """
    Normalize studies into a list of `(label, study)` pairs.
    """
    if isinstance(studies, Mapping):
        return [(str(k), v) for k, v in studies.items()]

    if isinstance(studies, Sequence):
        out: list[tuple[str, MonteCarloStudyResult | MonteCarloStudySummary]] = []
        for i, item in enumerate(studies):
            out.append((f"study_{i}", item))
        return out

    raise TypeError(
        "studies must be a mapping of label -> study/summary or a sequence of studies."
    )


def _aggregate_center_and_error(
    agg: MonteCarloScalarAggregate,
    *,
    stat: str = "mean",
    error_band: str = "p05_p95",
) -> tuple[float, float, float]:
    """
    Convert one aggregate into `(center, lower_err, upper_err)`.

    Parameters
    ----------
    agg : MonteCarloScalarAggregate
        Aggregate metric summary.
    stat : {"mean", "median"}
        Central statistic.
    error_band : {"p05_p95", "std", "none"}
        Error-bar construction rule.

    Returns
    -------
    tuple[float, float, float]
        `(center, lower, upper)` suitable for asymmetric y-error bars.
    """
    key = str(stat).strip().lower()
    if key == "mean":
        center = float(agg.mean)
    elif key == "median":
        center = float(agg.median)
    else:
        raise ValueError("stat must be 'mean' or 'median'.")

    band = str(error_band).strip().lower()
    if band == "none":
        return center, 0.0, 0.0
    if band == "std":
        spread = float(agg.std)
        spread = 0.0 if not np.isfinite(spread) else max(spread, 0.0)
        return center, spread, spread
    if band == "p05_p95":
        lower = 0.0 if not np.isfinite(agg.p05) else max(center - float(agg.p05), 0.0)
        upper = 0.0 if not np.isfinite(agg.p95) else max(float(agg.p95) - center, 0.0)
        return center, lower, upper

    raise ValueError("error_band must be one of {'p05_p95', 'std', 'none'}.")


# -----------------------------------------------------------------------------
# Public data access helpers
# -----------------------------------------------------------------------------


def available_metric_names(study: MonteCarloStudyResult) -> tuple[str, ...]:
    """
    Return all flattened scalar metric names available in a study result.

    Parameters
    ----------
    study : MonteCarloStudyResult
        Study result whose successful runs already contain metric summaries.

    Returns
    -------
    tuple[str, ...]
        Sorted flattened metric names.
    """
    return tuple(sorted(study.metric_series().keys()))


def metric_series(study: MonteCarloStudyResult, metric_name: str) -> FloatArray:
    """
    Public wrapper returning one finite scalar metric series.
    """
    return _metric_series(study, metric_name)


# -----------------------------------------------------------------------------
# One-study distribution plots
# -----------------------------------------------------------------------------


def plot_metric_histogram(
    study: MonteCarloStudyResult,
    metric_name: str,
    *,
    bins: int | str | Sequence[float] = "auto",
    density: bool = False,
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (7.5, 4.5),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    annotate_summary: bool = True,
    show_mean: bool = True,
    show_median: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot a histogram of one Monte Carlo metric across successful runs.

    Parameters
    ----------
    study : MonteCarloStudyResult
        Study result.
    metric_name : str
        Flattened scalar metric name.
    bins : int, str, or sequence, default="auto"
        Histogram bin rule passed to Matplotlib.
    density : bool, default=False
        Whether to normalize to probability density.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into.
    figsize : tuple, default=(7.5, 4.5)
        Figure size used when `ax is None`.
    title : str, optional
        Optional title override.
    xlabel : str, optional
        Optional x-axis label override.
    annotate_summary : bool, default=True
        Whether to add a summary textbox.
    show_mean : bool, default=True
        Draw the sample mean as a vertical reference line.
    show_median : bool, default=True
        Draw the sample median as a vertical reference line.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes.
    """
    values = _metric_series(study, metric_name)

    if ax is None:
        fig, ax = _new_figure_and_axes(figsize=figsize)
    else:
        fig = ax.figure

    ax.hist(values, bins=bins, density=density, alpha=0.8)
    ax.set_xlabel(_metric_display_name(metric_name) if xlabel is None else str(xlabel))
    ax.set_ylabel("density" if density else "count")
    ax.set_title(
        _metric_display_name(metric_name) if title is None else str(title)
    )
    ax.grid(True, alpha=0.25)

    mean_v = float(np.mean(values))
    median_v = float(np.median(values))
    p95_v = float(np.percentile(values, 95.0))

    if show_mean:
        ax.axvline(mean_v, linestyle="--", linewidth=1.5, label="mean")
    if show_median:
        ax.axvline(median_v, linestyle=":", linewidth=1.8, label="median")

    if show_mean or show_median:
        ax.legend()

    if annotate_summary:
        summary_text = "\n".join(
            [
                f"n = {values.size}",
                f"mean = {mean_v:.6g}",
                f"median = {median_v:.6g}",
                f"p95 = {p95_v:.6g}",
            ]
        )
        ax.text(
            0.98,
            0.98,
            summary_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round", "alpha": 0.15},
        )

    return fig, ax


def plot_metric_ecdf(
    study: MonteCarloStudyResult,
    metric_name: str,
    *,
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (7.5, 4.5),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "empirical CDF",
) -> tuple[Figure, Axes]:
    """
    Plot the empirical CDF of one Monte Carlo metric.

    Parameters
    ----------
    study : MonteCarloStudyResult
        Study result.
    metric_name : str
        Flattened scalar metric name.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into.
    figsize : tuple, default=(7.5, 4.5)
        Figure size used when `ax is None`.
    title : str, optional
        Optional title override.
    xlabel : str, optional
        Optional x-axis label override.
    ylabel : str, default="empirical CDF"
        Y-axis label.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes.
    """
    values = np.sort(_metric_series(study, metric_name))
    y = np.arange(1, values.size + 1, dtype=np.float64) / float(values.size)

    if ax is None:
        fig, ax = _new_figure_and_axes(figsize=figsize)
    else:
        fig = ax.figure

    ax.step(values, y, where="post")
    ax.set_xlabel(_metric_display_name(metric_name) if xlabel is None else str(xlabel))
    ax.set_ylabel(str(ylabel))
    ax.set_title(
        f"ECDF: {_metric_display_name(metric_name)}"
        if title is None
        else str(title)
    )
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    return fig, ax


def plot_metric_distribution_panel(
    study: MonteCarloStudyResult,
    metric_name: str,
    *,
    bins: int | str | Sequence[float] = "auto",
    figsize: tuple[float, float] = (11.0, 4.2),
    title_prefix: Optional[str] = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """
    Plot both histogram and ECDF for one metric on a 1x2 panel.

    Parameters
    ----------
    study : MonteCarloStudyResult
        Study result.
    metric_name : str
        Flattened scalar metric name.
    bins : int, str, or sequence, default="auto"
        Histogram bin rule.
    figsize : tuple, default=(11.0, 4.2)
        Figure size.
    title_prefix : str, optional
        Optional title prefix.

    Returns
    -------
    tuple[Figure, tuple[Axes, Axes]]
        Figure and `(ax_hist, ax_ecdf)`.
    """
    fig, axes = _new_figure_and_axes(
        nrows=1,
        ncols=2,
        figsize=figsize,
        squeeze=True,
    )
    ax_hist, ax_ecdf = axes

    prefix = "" if title_prefix is None else f"{title_prefix}: "
    label = _metric_display_name(metric_name)

    plot_metric_histogram(
        study,
        metric_name,
        bins=bins,
        ax=ax_hist,
        title=f"{prefix}{label}",
    )
    plot_metric_ecdf(
        study,
        metric_name,
        ax=ax_ecdf,
        title=f"{prefix}ECDF",
        xlabel=label,
    )
    return fig, (ax_hist, ax_ecdf)


def plot_metric_grid(
    study: MonteCarloStudyResult,
    metric_names: Sequence[str],
    *,
    bins: int | str | Sequence[float] = "auto",
    ncols: int = 2,
    figsize_per_panel: tuple[float, float] = (5.2, 3.8),
    density: bool = False,
) -> tuple[Figure, NDArray[np.object_]]:
    """
    Plot histograms for multiple metrics on a regular grid.

    Parameters
    ----------
    study : MonteCarloStudyResult
        Study result.
    metric_names : sequence of str
        Flattened scalar metric names to plot.
    bins : int, str, or sequence, default="auto"
        Histogram bin rule.
    ncols : int, default=2
        Number of columns in the panel grid.
    figsize_per_panel : tuple, default=(5.2, 3.8)
        Per-panel size used to scale the full figure.
    density : bool, default=False
        Whether to normalize histograms to density.

    Returns
    -------
    tuple[Figure, np.ndarray]
        Figure and 2D axes array.
    """
    names = [str(m) for m in metric_names]
    if len(names) == 0:
        raise ValueError("metric_names must contain at least one entry.")

    ncols = max(int(ncols), 1)
    nrows = int(np.ceil(len(names) / ncols))

    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = _new_figure_and_axes(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    flat_axes = axes.reshape(-1)
    for ax, metric_name in zip(flat_axes, names):
        plot_metric_histogram(
            study,
            metric_name,
            bins=bins,
            density=density,
            ax=ax,
            annotate_summary=False,
            title=_metric_display_name(metric_name),
        )

    for ax in flat_axes[len(names):]:
        ax.set_visible(False)

    return fig, axes


# -----------------------------------------------------------------------------
# Aggregate plots for one study
# -----------------------------------------------------------------------------


def plot_metric_aggregate_bars(
    study_or_summary: MonteCarloStudyResult | MonteCarloStudySummary,
    metric_names: Sequence[str],
    *,
    stat: str = "mean",
    error_band: str = "p05_p95",
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (9.0, 4.8),
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    rotate_xticks_deg: float = 20.0,
) -> tuple[Figure, Axes]:
    """
    Plot aggregate bars for several metrics from one study.

    Parameters
    ----------
    study_or_summary : MonteCarloStudyResult or MonteCarloStudySummary
        Study source.
    metric_names : sequence of str
        Metrics to include.
    stat : {"mean", "median"}, default="mean"
        Central statistic to plot.
    error_band : {"p05_p95", "std", "none"}, default="p05_p95"
        Error-bar rule.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into.
    figsize : tuple, default=(9.0, 4.8)
        Figure size used when `ax is None`.
    title : str, optional
        Plot title override.
    ylabel : str, optional
        Y-axis label override.
    rotate_xticks_deg : float, default=20
        X tick rotation angle.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes.
    """
    names = [str(m) for m in metric_names]
    if len(names) == 0:
        raise ValueError("metric_names must contain at least one entry.")

    if ax is None:
        fig, ax = _new_figure_and_axes(figsize=figsize)
    else:
        fig = ax.figure

    centers: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    for metric_name in names:
        agg = _metric_aggregate(study_or_summary, metric_name)
        c, lo, hi = _aggregate_center_and_error(
            agg,
            stat=stat,
            error_band=error_band,
        )
        centers.append(c)
        lower.append(lo)
        upper.append(hi)

    x = np.arange(len(names), dtype=np.float64)
    yerr = np.vstack([lower, upper])

    ax.bar(x, centers)
    if error_band != "none":
        ax.errorbar(
            x,
            centers,
            yerr=yerr,
            fmt="none",
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_metric_display_name(v) for v in names], rotation=rotate_xticks_deg, ha="right")
    ax.set_ylabel(
        str(ylabel) if ylabel is not None else str(stat).lower()
    )
    ax.set_title(
        "Monte Carlo metric aggregates" if title is None else str(title)
    )
    ax.grid(True, axis="y", alpha=0.25)
    return fig, ax


def plot_failure_summary(
    study: MonteCarloStudyResult,
    *,
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (6.5, 4.0),
    title: Optional[str] = None,
    annotate: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot successful vs failed run counts for a study.

    Parameters
    ----------
    study : MonteCarloStudyResult
        Study result.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into.
    figsize : tuple, default=(6.5, 4.0)
        Figure size used when `ax is None`.
    title : str, optional
        Title override.
    annotate : bool, default=True
        Whether to annotate counts above the bars.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes.
    """
    summary = study.summary()
    values = np.array(
        [summary.n_runs_completed, summary.n_runs_failed],
        dtype=np.float64,
    )
    labels = ["completed", "failed"]

    if ax is None:
        fig, ax = _new_figure_and_axes(figsize=figsize)
    else:
        fig = ax.figure

    bars = ax.bar(labels, values)
    ax.set_ylabel("count")
    ax.set_title("Monte Carlo run outcomes" if title is None else str(title))
    ax.grid(True, axis="y", alpha=0.25)

    if annotate:
        for rect, value in zip(bars, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height(),
                f"{int(value)}",
                ha="center",
                va="bottom",
            )

    return fig, ax


# -----------------------------------------------------------------------------
# Multi-study comparison plots
# -----------------------------------------------------------------------------


def compare_studies_metric_boxplot(
    studies: (
        Mapping[str, MonteCarloStudyResult]
        | Sequence[MonteCarloStudyResult]
    ),
    metric_name: str,
    *,
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (8.5, 4.8),
    title: Optional[str] = None,
    showmeans: bool = True,
    rotate_xticks_deg: float = 15.0,
) -> tuple[Figure, Axes]:
    """
    Compare one metric across multiple studies using a boxplot.

    Parameters
    ----------
    studies : mapping or sequence of MonteCarloStudyResult
        Studies to compare.
    metric_name : str
        Flattened scalar metric name.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into.
    figsize : tuple, default=(8.5, 4.8)
        Figure size used when `ax is None`.
    title : str, optional
        Title override.
    showmeans : bool, default=True
        Whether to show sample means on the boxplot.
    rotate_xticks_deg : float, default=15
        X tick rotation angle.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes.
    """
    named = _named_studies(studies)
    if len(named) == 0:
        raise ValueError("At least one study must be provided.")

    labels: list[str] = []
    data: list[FloatArray] = []

    for label, study in named:
        if not isinstance(study, MonteCarloStudyResult):
            raise TypeError(
                "compare_studies_metric_boxplot requires full MonteCarloStudyResult "
                "objects because it plots per-run distributions."
            )
        labels.append(label)
        data.append(_metric_series(study, metric_name))

    if ax is None:
        fig, ax = _new_figure_and_axes(figsize=figsize)
    else:
        fig = ax.figure

    ax.boxplot(data, labels=labels, showmeans=showmeans)
    ax.set_ylabel(_metric_display_name(metric_name))
    ax.set_title(
        f"Study comparison: {_metric_display_name(metric_name)}"
        if title is None
        else str(title)
    )
    ax.grid(True, axis="y", alpha=0.25)
    for tick in ax.get_xticklabels():
        tick.set_rotation(rotate_xticks_deg)
        tick.set_ha("right")
    return fig, ax


def compare_studies_metric_summary(
    studies: (
        Mapping[str, MonteCarloStudyResult | MonteCarloStudySummary]
        | Sequence[MonteCarloStudyResult | MonteCarloStudySummary]
    ),
    metric_name: str,
    *,
    stat: str = "mean",
    error_band: str = "p05_p95",
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (8.5, 4.8),
    title: Optional[str] = None,
    rotate_xticks_deg: float = 15.0,
) -> tuple[Figure, Axes]:
    """
    Compare one aggregate metric across multiple studies using bars + error bars.

    Parameters
    ----------
    studies : mapping or sequence of study results/summaries
        Studies to compare.
    metric_name : str
        Flattened scalar metric name.
    stat : {"mean", "median"}, default="mean"
        Aggregate center to plot.
    error_band : {"p05_p95", "std", "none"}, default="p05_p95"
        Error-bar rule.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into.
    figsize : tuple, default=(8.5, 4.8)
        Figure size used when `ax is None`.
    title : str, optional
        Title override.
    rotate_xticks_deg : float, default=15
        X tick rotation angle.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes.
    """
    named = _named_studies(studies)
    if len(named) == 0:
        raise ValueError("At least one study must be provided.")

    labels: list[str] = []
    centers: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    for label, study_or_summary in named:
        agg = _metric_aggregate(study_or_summary, metric_name)
        center, lo, hi = _aggregate_center_and_error(
            agg,
            stat=stat,
            error_band=error_band,
        )
        labels.append(label)
        centers.append(center)
        lower.append(lo)
        upper.append(hi)

    if ax is None:
        fig, ax = _new_figure_and_axes(figsize=figsize)
    else:
        fig = ax.figure

    x = np.arange(len(labels), dtype=np.float64)
    ax.bar(x, centers)

    if error_band != "none":
        ax.errorbar(
            x,
            centers,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate_xticks_deg, ha="right")
    ax.set_ylabel(_metric_display_name(metric_name))
    ax.set_title(
        f"Study comparison: {_metric_display_name(metric_name)}"
        if title is None
        else str(title)
    )
    ax.grid(True, axis="y", alpha=0.25)
    return fig, ax


# -----------------------------------------------------------------------------
# Persistence helper
# -----------------------------------------------------------------------------


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


__all__ = [
    "FloatArray",
    "available_metric_names",
    "compare_studies_metric_boxplot",
    "compare_studies_metric_summary",
    "metric_series",
    "plot_failure_summary",
    "plot_metric_aggregate_bars",
    "plot_metric_distribution_panel",
    "plot_metric_ecdf",
    "plot_metric_grid",
    "plot_metric_histogram",
    "save_figure",
]