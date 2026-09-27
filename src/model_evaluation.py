"""Hourly binary evaluation; threshold selection uses validation data only.

Calibration here assesses the supplied probabilities. It does not fit a
calibrator, alter probabilities, or choose a threshold from test labels.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _validate(y, probabilities):
    """Reject invalid inputs rather than silently dropping evaluation rows."""
    try:
        labels = np.asarray(y, dtype=np.float64)
        scores = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Labels and probabilities must be numeric") from exc
    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("Labels and probabilities must be one-dimensional")
    if labels.size == 0 or labels.shape != scores.shape:
        raise ValueError("Labels and probabilities must have equal, nonzero length")
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must contain only finite binary values 0 and 1")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return labels.astype(np.uint8, copy=False), scores


def choose_threshold(y_validation, probabilities):
    """Maximize validation F1 using exact ``probability >= threshold``.

    Every distinct probability is considered. Ties choose the highest threshold
    (the smallest predicted-positive set among tied candidates). No test data
    belongs in this function.
    """
    labels, scores = _validate(y_validation, probabilities)
    total_positive = int(labels.sum())
    if total_positive in (0, labels.size):
        raise ValueError("Threshold selection requires both classes in validation")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    tp = np.cumsum(labels[order], dtype=np.int64)
    # At each last occurrence, all observations tied at that threshold are in.
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), labels.size - 1]
    f1 = 2.0 * tp[ends] / (total_positive + ends + 1)
    best = int(np.argmax(f1))  # Descending scores make the highest-threshold tie win.
    return {
        "threshold": float(sorted_scores[ends[best]]),
        "validation_f1": float(f1[best]),
        "rule": "maximize validation F1; ties choose highest threshold",
        "comparison": "probability >= threshold",
        "validation_samples": int(labels.size),
        "candidate_thresholds": int(ends.size),
    }


def _calibration_bins(labels, scores, n_bins):
    # [left, right), except the final bin includes probability exactly 1.
    indexes = np.minimum((scores * n_bins).astype(np.int64), n_bins - 1)
    counts = np.bincount(indexes, minlength=n_bins)
    predicted_sum = np.bincount(indexes, weights=scores, minlength=n_bins)
    positive_sum = np.bincount(indexes, weights=labels, minlength=n_bins)
    bins, ece = [], 0.0
    for index in range(n_bins):
        count = int(counts[index])
        mean_probability = float(predicted_sum[index] / count) if count else None
        observed = float(positive_sum[index] / count) if count else None
        if count:
            ece += count / labels.size * abs(mean_probability - observed)
        bins.append({
            "bin": index + 1,
            "lower_bound": index / n_bins,
            "upper_bound": (index + 1) / n_bins,
            "upper_bound_inclusive": index == n_bins - 1,
            "count": count,
            "positive_count": int(positive_sum[index]),
            "mean_probability": mean_probability,
            "observed_positive_fraction": observed,
        })
    return bins, float(ece)


def evaluate_binary(y, probabilities, threshold, n_bins=10):
    """Return JSON-safe hourly metrics and fixed-width calibration bins.

    Undefined precision/recall/specificity/F1 denominators return zero and are
    documented in warnings. AUROC is null when either class is absent. PR areas
    are null when there are no positive observations.
    """
    labels, scores = _validate(y, probabilities)
    if not np.isscalar(threshold) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0, 1]")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    predicted = scores >= threshold
    positive = labels == 1
    tp = int(np.count_nonzero(positive & predicted))
    tn = int(np.count_nonzero(~positive & ~predicted))
    fp = int(np.count_nonzero(~positive & predicted))
    fn = int(np.count_nonzero(positive & ~predicted))
    n_positive, n_negative = tp + fn, tn + fp
    warnings = []
    def ratio(numerator, denominator, name):
        if not denominator:
            warnings.append(f"{name} has a zero denominator; reported as 0.0")
            return 0.0
        return float(numerator / denominator)
    precision = ratio(tp, tp + fp, "Precision")
    recall = ratio(tp, n_positive, "Recall/sensitivity")
    specificity = ratio(tn, n_negative, "Specificity")
    f1 = ratio(2 * tp, 2 * tp + fp + fn, "F1")
    if n_positive and n_negative:
        auroc = float(roc_auc_score(labels, scores))
    else:
        auroc = None
        warnings.append("AUROC is undefined because only one target class is present")
    if n_positive:
        average_precision = float(average_precision_score(labels, scores))
        curve_precision, curve_recall, _ = precision_recall_curve(labels, scores)
        trapezoidal = float(auc(curve_recall, curve_precision))
    else:
        average_precision, trapezoidal = None, None
        warnings.append("PR areas are undefined because there are no positive observations")
    bins, ece = _calibration_bins(labels, scores, int(n_bins))
    prevalence = float(n_positive / labels.size)
    return {
        "threshold": float(threshold),
        "n_samples": int(labels.size),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "accuracy": float((tn + tp) / labels.size),
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "auroc": auroc,
        "auprc": average_precision,
        "pr_auc_trapezoidal": trapezoidal,
        "brier_score": float(np.mean((scores - labels) ** 2)),
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "ece": ece,
        "mean_predicted_probability": float(scores.mean()),
        "prevalence": prevalence,
        "all_negative_accuracy": float(n_negative / labels.size),
        "no_skill_auprc": prevalence,
        "prevalence_predictor_brier": float(prevalence * (1 - prevalence)),
        "calibration_bins": bins,
        "warnings": warnings,
        "definitions": {
            "unit": "hourly record; repeated records from one patient are correlated",
            "positive_prediction": "probability >= threshold",
            "confusion_matrix": "rows=true [0,1]; columns=predicted [0,1]",
            "auprc": "average precision (step-weighted PR area), not trapezoidal PR AUC",
            "pr_auc_trapezoidal": "trapezoidal integration of the precision-recall curve",
            "recall_sensitivity": "TP / (TP + FN); these are identical metrics",
            "zero_denominator": "precision, recall, specificity and F1 return 0.0 with a warning",
            "calibration": "assessment of supplied, uncalibrated probabilities; no calibrator fitted",
            "ece": f"count-weighted absolute calibration error in {n_bins} fixed uniform bins",
            "bin_boundaries": "[lower, upper); final bin includes 1; empty-bin means are null",
            "log_loss": "natural-log loss; sklearn clips endpoints to float64 machine precision",
            "baseline": "prevalence baselines describe this evaluation split, not fitted model parameters",
        },
    }


def _write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def _plot_evaluation(labels, scores, result, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Evaluation plots require matplotlib; install requirements-models.txt") from exc
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=15)
    matrix = np.asarray(result["confusion_matrix"])
    ax = axes[0, 0]
    ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center",
                    color="white" if matrix[row, column] > matrix.max() / 2 else "black", fontsize=14)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["0", "1"], yticklabels=["0", "1"],
           xlabel="Predicted label", ylabel="True label",
           title=f"Confusion matrix | threshold = {result['threshold']:.6g}")
    ax = axes[0, 1]
    if result["auroc"] is not None:
        fpr, tpr, _ = roc_curve(labels, scores)
        indexes = np.unique(np.linspace(0, fpr.size - 1, min(5000, fpr.size), dtype=int))
        ax.plot(fpr[indexes], tpr[indexes], label=f"AUROC = {result['auroc']:.4f}")
    else:
        ax.text(0.5, 0.5, "Undefined: only one target class", ha="center")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve")
    ax.legend(loc="lower right")
    ax = axes[1, 0]
    if result["auprc"] is not None:
        precision, recall, _ = precision_recall_curve(labels, scores)
        indexes = np.unique(np.linspace(0, recall.size - 1, min(5000, recall.size), dtype=int))
        ax.step(recall[indexes], precision[indexes], where="post", label=f"Average precision = {result['auprc']:.4f}")
    else:
        ax.text(0.5, 0.5, "Undefined: no positive observations", ha="center")
    ax.axhline(result["prevalence"], linestyle="--", color="gray", label=f"Prevalence = {result['prevalence']:.4f}")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    ax.legend(loc="upper right")
    ax = axes[1, 1]
    nonempty = [row for row in result["calibration_bins"] if row["count"]]
    means = [row["mean_probability"] for row in nonempty]
    observed = [row["observed_positive_fraction"] for row in nonempty]
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(means, observed, "o-", label="Observed probabilities")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed positive fraction",
           title=f"Calibration | Brier = {result['brier_score']:.4f} | ECE = {result['ece']:.4f}")
    ax.legend(loc="upper left")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.text(0.5, 0.015, "Hourly evaluation; uncalibrated probabilities. Curves may be thinned for plotting only.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_evaluation(y, probabilities, threshold, output_dir, split, model_name, n_bins=10):
    """Save metrics, exact numeric curves, calibration bins, and four-panel PNG."""
    labels, scores = _validate(y, probabilities)
    if not split or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in split):
        raise ValueError("split must be a nonempty filename-safe name")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_binary(labels, scores, threshold, n_bins=n_bins)
    result.update({"model_name": str(model_name), "split": split})
    if float(threshold) != 0.5:
        result["at_threshold_0_5"] = evaluate_binary(labels, scores, 0.5, n_bins=n_bins)
    bins = result["calibration_bins"]
    _write_csv(output_dir / f"{split}_calibration.csv", list(bins[0]), (row.values() for row in bins))
    _write_csv(output_dir / f"{split}_confusion_matrix.csv", ["true_label", "predicted_0", "predicted_1"],
               [[0, result["tn"], result["fp"]], [1, result["fn"], result["tp"]]])
    if result["auroc"] is not None:
        fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
        _write_csv(output_dir / f"{split}_roc_curve.csv", ["false_positive_rate", "true_positive_rate", "threshold"],
                   ((float(f), float(t), float(th) if np.isfinite(th) else None) for f, t, th in zip(fpr, tpr, thresholds)))
    else:
        _write_csv(output_dir / f"{split}_roc_curve.csv", ["false_positive_rate", "true_positive_rate", "threshold"], [])
    if result["auprc"] is not None:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        _write_csv(output_dir / f"{split}_pr_curve.csv", ["precision", "recall", "threshold"],
                   ((float(p), float(r), float(thresholds[i]) if i < thresholds.size else None)
                    for i, (p, r) in enumerate(zip(precision, recall))))
    else:
        _write_csv(output_dir / f"{split}_pr_curve.csv", ["precision", "recall", "threshold"], [])
    _plot_evaluation(labels, scores, result, output_dir / f"{split}_evaluation.png", f"{model_name} - {split}")
    result["artifacts"] = {name: str(output_dir / f"{split}_{suffix}") for name, suffix in {
        "metrics": "metrics.json", "calibration": "calibration.csv", "confusion_matrix": "confusion_matrix.csv",
        "roc_curve": "roc_curve.csv", "pr_curve": "pr_curve.csv", "figure": "evaluation.png",
    }.items()}
    (output_dir / f"{split}_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
