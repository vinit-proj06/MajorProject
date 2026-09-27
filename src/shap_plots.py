"""All-feature native-SHAP plots with bounded column-wise density aggregation.

SHAP values are additive raw log-odds contributions, not probability changes.
Global summaries use every supplied observation. Binning is display aggregation,
not row sampling, feature selection, or clipping of attribution outliers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

POSITIVE = "#c93e51"
NEGATIVE = "#287bb5"
NEUTRAL = "#777777"
DPI = 160


def _names_and_order(feature_names, order):
    names = list(feature_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("feature_names must contain nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    ordered = np.asarray(order)
    if (ordered.ndim != 1 or ordered.size != len(names) or ordered.dtype.kind not in "iu"
            or not np.array_equal(np.sort(ordered), np.arange(len(names)))):
        raise ValueError("order must be a complete permutation of every feature; no cutoff is allowed")
    return names, ordered.astype(np.int64, copy=False)


def _vector(values, length, name):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector with one value per feature")
    return array


def _scalar(value, name):
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a finite scalar")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def _row_figure(n_features, title, subtitle, *, local=False):
    height = max(6.0, 0.30 * (n_features + (2 if local else 0)) + 2.4)
    fig, ax = plt.subplots(figsize=(18, height))
    fig.subplots_adjust(left=0.39 if local else 0.35, right=0.94,
                        bottom=0.85 / height, top=1 - 1.05 / height)
    fig.text(0.025, 1 - 0.24 / height, title, ha="left", va="top", fontsize=17, weight="bold")
    fig.text(0.025, 1 - 0.57 / height, subtitle, ha="left", va="top", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(7))
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    return fig, ax, height


def _finish(path, fig, names, order, **metadata):
    path = Path(path)
    if path.suffix.lower() not in {".png", ".pdf"}:
        path = path.with_suffix(".png")
    png, pdf = path.with_suffix(".png"), path.with_suffix(".pdf")
    png.parent.mkdir(parents=True, exist_ok=True)
    try:
        # PDF preserves every label as vector text. PNG supports ordinary viewers.
        fig.savefig(pdf, metadata={"Title": "All-feature SHAP explanation"})
        fig.savefig(png, dpi=DPI)
    finally:
        plt.close(fig)
    return {"png": str(png), "pdf": str(pdf), "feature_count": len(names),
            "feature_order": order.tolist(),
            "feature_names": [names[index] for index in order], **metadata}


def _percentile_ranks(values):
    """Average tied ranks, mapped to [0, 1]; a constant feature is neutral."""
    if len(values) == 1:
        return np.array([0.5])
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    average_ranks = (np.cumsum(counts) - (counts + 1) / 2) / (len(values) - 1)
    return average_ranks[inverse]


def plot_global_summary(path, feature_names, X, shap_values, order, bins=160):
    """Plot every feature and observation as a binned mirrored density.

    Each feature is read separately, including when inputs are memory maps.
    Mirrored height is proportional to sqrt(bin count / largest count in row).
    Exact zeros and any other constant attribution mass are drawn at their exact
    x coordinate as vertical stems, avoiding a misleading artificial bin width.
    Colors are mean within-feature value percentile ranks in each bin/mass.
    """
    names, order = _names_and_order(feature_names, order)
    X, shap_values = np.asarray(X), np.asarray(shap_values)
    if (X.ndim != 2 or X.shape != shap_values.shape or X.shape[1] != len(names)
            or X.shape[0] == 0):
        raise ValueError("X and shap_values must have the same nonempty (rows, features) shape")
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    fig, ax, height = _row_figure(
        len(names), f"Global SHAP summary — all {len(names)} features",
        f"All test rows; binned density; raw log-odds | {len(X):,} rows per feature | no feature cutoff or row subsampling")
    # Leave space between the axis label and the two-line density explanation.
    fig.subplots_adjust(bottom=1.12 / height)
    cmap, norm = plt.get_cmap("coolwarm"), Normalize(0, 1)
    represented, zero_counts = [], []
    min_value, max_value = 0.0, 0.0
    try:
        for row, column in enumerate(order):
            values = np.asarray(X[:, column], dtype=np.float64)
            contributions = np.asarray(shap_values[:, column], dtype=np.float64)
            if not np.isfinite(values).all() or not np.isfinite(contributions).all():
                raise ValueError(f"Nonfinite feature/SHAP values for {names[column]}; no rows are silently dropped")
            ranks = _percentile_ranks(values)
            min_value = min(min_value, float(contributions.min()))
            max_value = max(max_value, float(contributions.max()))
            zero = contributions == 0
            zero_count = int(zero.sum())
            zero_counts.append(zero_count)
            masses = []
            if zero_count:
                masses.append((0.0, zero_count, float(ranks[zero].mean())))
            nonzero = contributions[~zero]
            nonzero_ranks = ranks[~zero]
            counts = np.zeros(0, dtype=np.int64)
            edges = np.zeros(0)
            average_colors = np.zeros(0)
            if len(nonzero):
                low, high = float(nonzero.min()), float(nonzero.max())
                if low == high:
                    masses.append((low, len(nonzero), float(nonzero_ranks.mean())))
                else:
                    edges = np.linspace(low, high, int(bins) + 1)
                    # Rightmost values belong to the final bin, including maximum.
                    indexes = np.clip(np.searchsorted(edges, nonzero, side="right") - 1, 0, int(bins) - 1)
                    counts = np.bincount(indexes, minlength=int(bins))
                    sums = np.bincount(indexes, weights=nonzero_ranks, minlength=int(bins))
                    average_colors = np.divide(sums, counts, out=np.full(int(bins), 0.5), where=counts > 0)
            covered = int(counts.sum()) + sum(mass[1] for mass in masses)
            if covered != len(X):
                raise RuntimeError("Density aggregation failed to represent every observation")
            represented.append(covered)
            largest = max([int(counts.max()) if counts.size else 0, *[mass[1] for mass in masses]])
            nonempty = np.flatnonzero(counts)
            if nonempty.size:
                half_width = 0.40 * np.sqrt(counts[nonempty] / largest)
                polygons = [
                    [(edges[i], row - width), (edges[i + 1], row - width),
                     (edges[i + 1], row + width), (edges[i], row + width)]
                    for i, width in zip(nonempty, half_width)
                ]
                ax.add_collection(PolyCollection(polygons, facecolors=cmap(norm(average_colors[nonempty])),
                                                 edgecolors="none", linewidths=0))
            for position, count, color_rank in masses:
                width = 0.40 * np.sqrt(count / largest)
                ax.vlines(position, row - width, row + width,
                          color=cmap(norm(color_rank)), linewidth=2.4, zorder=4)
        extent = max(abs(min_value), abs(max_value)) or 1.0
        ax.set_xlim(-1.04 * extent, 1.04 * extent)
        ax.set_ylim(len(names) - 0.4, -0.6)
        ax.set_yticks(np.arange(len(names)), [names[index] for index in order], fontsize=9)
        ax.axvline(0, color="#666666", linewidth=0.7, alpha=0.65, zorder=0)
        ax.set_xlabel("SHAP contribution to raw model log-odds", fontsize=11)
        color_ax = fig.add_axes([0.965, 0.5 - 1.1 / height, 0.009, 2.2 / height])
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_ax)
        colorbar.set_ticks([0, 1], labels=["Low", "High"])
        colorbar.ax.tick_params(labelsize=8)
        colorbar.ax.set_title("Feature\nvalue", fontsize=8, pad=8)
        fig.text(0.35, 0.18 / height,
                 "Bin height uses square-root counts within each feature; color = mean value percentile. Exact-value stems preserve zero/constant mass.\n"
                 "All attributions are represented; no x-axis clipping. Order follows the complete supplied global importance ranking.",
                 fontsize=8, va="bottom")
    except Exception:
        plt.close(fig)
        raise
    return _finish(path, fig, names, order, sample_count=len(X),
                   represented_values_per_feature=represented,
                   zero_shap_values_per_feature=zero_counts,
                   binned_density=True, row_subsampling=False, bins=int(bins),
                   color_encoding="mean within-feature percentile rank per bin",
                   density_encoding="square root of bin count relative to largest count in each feature",
                   exact_zero_mass_preserved=True, attribution_units="raw log-odds")


def plot_importance(path, feature_names, mean_abs, order):
    """Full horizontal ranking; retain zero-importance features and every label."""
    names, order = _names_and_order(feature_names, order)
    importance = _vector(mean_abs, len(names), "mean_abs")
    if np.any(importance < 0):
        raise ValueError("mean_abs cannot contain negative values")
    displayed = importance[order]
    fig, ax, _ = _row_figure(
        len(names), f"Mean absolute SHAP importance — all {len(names)} features",
        "Complete ranking; no selected cutoff | Contributions measured in raw model log-odds")
    y = np.arange(len(names))
    ax.barh(y, displayed, height=0.68, color=NEGATIVE)
    maximum = float(displayed.max()) or 1.0
    for row, value in enumerate(displayed):
        ax.text(value + 0.012 * maximum, row, f"{value:.6g}", va="center", fontsize=8)
        if value == 0:
            ax.plot(0, row, marker="|", color=NEUTRAL, markersize=8, clip_on=False)
    ax.set_yticks(y, [f"{rank + 1}. {names[index]}" for rank, index in enumerate(order)], fontsize=9)
    ax.set_ylim(len(names) - 0.4, -0.6)
    ax.set_xlim(0, maximum * 1.20)
    ax.set_xlabel("Mean absolute SHAP contribution (raw log-odds)", fontsize=11)
    return _finish(path, fig, names, order, mean_abs=displayed.tolist(),
                   zero_importance_count=int(np.count_nonzero(importance == 0)), selected_cutoff=None)


def plot_elbow(path, feature_names, mean_abs, order):
    """Importance by rank, displaying every rank including zero-valued entries."""
    names, order = _names_and_order(feature_names, order)
    importance = _vector(mean_abs, len(names), "mean_abs")
    if np.any(importance < 0):
        raise ValueError("mean_abs cannot contain negative values")
    displayed, ranks = importance[order], np.arange(1, len(names) + 1)
    fig, ax = plt.subplots(figsize=(13, 6.5), layout="constrained")
    ax.plot(ranks, displayed, color=NEGATIVE, linewidth=1.5, marker="o", markersize=3)
    ax.set(title=f"SHAP importance by rank — all {len(names)} features, no cutoff selected",
           xlabel="Feature rank in the complete importance table", ylabel="Mean absolute SHAP contribution (raw log-odds)",
           xlim=(0.5, len(names) + 0.5), ylim=(0, (float(displayed.max()) or 1.0) * 1.08))
    ax.xaxis.set_major_locator(MaxNLocator(12, integer=True))
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Every feature is plotted, including zeros. Feature names appear in the full ranking plot/table. No automatic feature selection.", fontsize=9)
    return _finish(path, fig, names, order, ranks=ranks.tolist(),
                   mean_abs=displayed.tolist(), selected_cutoff=None)


def plot_local_waterfall(path, feature_names, feature_values, shap_values, base_value,
                         margin, probability, threshold, patient_id, hour):
    """Add every feature term, including zeros, from baseline to model margin."""
    supplied_names = list(feature_names)
    names, _ = _names_and_order(supplied_names, np.arange(len(supplied_names)))
    values = _vector(feature_values, len(names), "feature_values")
    contributions = _vector(shap_values, len(names), "shap_values")
    base = _scalar(base_value, "base_value")
    margin = _scalar(margin, "margin")
    probability = _scalar(probability, "probability")
    threshold = _scalar(threshold, "threshold")
    hour_label = "not supplied" if hour is None else f"{_scalar(hour, 'hour'):g}"
    if not 0 <= probability <= 1 or not 0 <= threshold <= 1:
        raise ValueError("probability and threshold must lie in [0, 1]")
    reconstructed = float(base + contributions.sum(dtype=np.float64))
    if not np.isclose(reconstructed, margin, rtol=1e-5, atol=1e-4):
        raise ValueError("SHAP additivity failed: base_value + all contributions differs from margin")
    logistic = 1 / (1 + np.exp(-margin)) if margin >= 0 else np.exp(margin) / (1 + np.exp(margin))
    if not np.isclose(logistic, probability, rtol=1e-5, atol=1e-6):
        raise ValueError("probability is inconsistent with the logistic transform of raw margin")
    order = np.argsort(-np.abs(contributions), kind="stable")
    terms = contributions[order]
    ends = base + np.cumsum(terms, dtype=np.float64)
    starts = np.r_[base, ends[:-1]]
    fig, ax, height = _row_figure(
        len(names), f"Local SHAP waterfall — patient {patient_id or 'not supplied'}, ICU hour {hour_label}",
        f"All {len(names)} terms, including zeros | p={probability:.8g}; frozen threshold={threshold:.8g} | "
        f"base {base:+.6g} + sum {contributions.sum():+.6g} = margin {margin:+.6g}\n"
        "Read top to bottom. Labels show processed feature values; contributions are log-odds, not probability-point changes.",
        local=True)
    minimum, maximum = min(base, margin, float(ends.min())), max(base, margin, float(ends.max()))
    span = max(maximum - minimum, 0.5)
    padding = 0.014 * span
    colors = [POSITIVE if value > 0 else NEGATIVE if value < 0 else NEUTRAL for value in terms]
    for row, (start, end, value, color) in enumerate(zip(starts, ends, terms, colors)):
        ax.barh(row, abs(value), left=min(start, end), color=color, height=0.66, edgecolor="none")
        if value == 0:
            ax.plot(start, row, marker="|", color=NEUTRAL, markersize=9)
        side = 1 if value >= 0 else -1
        ax.text(end + side * padding, row, f"{value:+.5g}", va="center",
                ha="left" if side > 0 else "right", fontsize=8, color=color)
        ax.plot([end, end], [row + 0.34, row + 0.66], color="#aaaaaa", linewidth=0.65, zorder=0)
    ax.plot([base, base], [-0.66, -0.34], color="#777777", linewidth=0.8)
    ax.scatter([base, margin], [-1, len(names)], color="#333333", marker="D", s=26, zorder=5)
    ax.text(base + padding, -1, f"{base:+.7g}", va="center", fontsize=9, weight="bold")
    ax.text(margin + padding, len(names), f"{margin:+.7g}", va="center", fontsize=9, weight="bold")
    labels = ["BASE VALUE"] + [f"{names[index]} = {values[index]:.8g}" for index in order] + ["FINAL MODEL MARGIN"]
    ax.set_yticks(np.arange(-1, len(names) + 1), labels, fontsize=8.5)
    ax.set_ylim(len(names) + 0.65, -1.65)
    ax.set_xlim(minimum - 0.14 * span, maximum + 0.17 * span)
    ax.axvline(base, color="#777777", linestyle=":", linewidth=0.75, alpha=0.7)
    ax.set_xlabel("Cumulative raw model log-odds", fontsize=11)
    fig.legend(handles=[Patch(color=POSITIVE, label="Raises log-odds"),
                        Patch(color=NEGATIVE, label="Lowers log-odds"),
                        Patch(color=NEUTRAL, label="Zero contribution")],
               loc="lower right", bbox_to_anchor=(0.975, 0.10 / height), ncol=3, fontsize=8)
    return _finish(path, fig, names, order,
                   displayed_contribution_sum=float(contributions.sum(dtype=np.float64)),
                   reconstructed_margin=reconstructed, margin=margin, base_value=base,
                   probability=probability, threshold=threshold,
                   zero_contribution_count=int(np.count_nonzero(contributions == 0)),
                   attribution_units="raw log-odds", selected_cutoff=None)
