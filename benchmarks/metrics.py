"""Shared metrics for metacognition benchmarks."""

import numpy as np
from typing import Optional


def compute_ece(
    correctness: list[float],
    confidences: list[float],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Args:
        correctness: List of 0.0/1.0 indicating if answer was correct.
        confidences: List of confidence values in [0, 1].
        n_bins: Number of bins.

    Returns:
        ECE value (0 = perfect calibration).
    """
    correctness = np.array(correctness)
    confidences = np.array(confidences)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_weight = mask.sum() / len(correctness)
        ece += bin_weight * abs(bin_acc - bin_conf)
    return float(ece)


def compute_brier_score(
    correctness: list[float],
    confidences: list[float],
) -> float:
    """Compute Brier Score = mean((confidence - correctness)^2)."""
    correctness = np.array(correctness)
    confidences = np.array(confidences)
    return float(np.mean((confidences - correctness) ** 2))


def compute_auroc(
    correctness: list[float],
    confidences: list[float],
) -> float:
    """Compute AUROC treating confidence as a classifier for correctness.

    Uses the abstention interpretation: higher confidence should predict correctness.
    """
    y_true = np.array(correctness)
    y_scores = np.array(confidences)
    if len(np.unique(y_true)) < 2:
        return 0.5

    # Sort by confidence descending
    order = np.argsort(-y_scores)
    y_true_sorted = y_true[order]

    # Compute ROC AUC manually
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = 0.0
    fp = 0.0
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1.0:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr = fpr
        prev_tpr = tpr

    return float(auc)


def compute_abstention_auroc(
    correctness: list[float],
    confidences: list[float],
) -> dict:
    """Compute abstention-style AUROC (coverage vs accuracy curve).

    Returns dict with raw_auroc, normalized_auroc, coverage_accuracy_curve.
    """
    pairs = list(zip(confidences, correctness))
    pairs.sort(key=lambda x: x[0], reverse=True)  # high confidence first
    n = len(pairs)
    if n == 0:
        return {"raw_auroc": 0.5, "normalized_auroc": 0.0, "overall_accuracy": 0.0}

    coverages = []
    accuracies = []
    running_correct = 0.0
    for k in range(1, n + 1):
        running_correct += pairs[k - 1][1]
        coverages.append(k / n)
        accuracies.append(running_correct / k)

    raw_auroc = float(np.trapz(accuracies, coverages))
    overall_acc = np.mean([p[1] for p in pairs])
    random_auroc = float(overall_acc)

    if 1.0 - random_auroc > 1e-9:
        normalized = (raw_auroc - random_auroc) / (1.0 - random_auroc)
    else:
        normalized = 0.0

    return {
        "raw_auroc": raw_auroc,
        "normalized_auroc": max(min(normalized, 1.0), 0.0),
        "overall_accuracy": float(overall_acc),
    }


def spearman_rho(x: list, y: list) -> float:
    """Compute Spearman rank correlation coefficient."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if len(x) < 3:
        return 0.0

    def _rank(arr):
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    d = rx - ry
    n = len(x)
    rho = 1.0 - (6.0 * np.sum(d ** 2)) / (n * (n ** 2 - 1))
    return float(np.clip(rho, -1.0, 1.0))


def pearson_r(x: list, y: list) -> float:
    """Compute Pearson correlation coefficient."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if len(x) < 3:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def f1_score(predicted: set, actual: set) -> dict:
    """Compute precision, recall, and F1 from two sets."""
    if not predicted and not actual:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(predicted & actual)
    precision = tp / max(len(predicted), 1)
    recall = tp / max(len(actual), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}


def confidence_entropy(confidences: list[float], n_bins: int = 10) -> float:
    """Compute entropy of the confidence distribution."""
    confidences = np.array(confidences)
    hist, _ = np.histogram(confidences, bins=n_bins, range=(0, 1), density=True)
    hist = hist / (hist.sum() + 1e-12)
    entropy = -np.sum(hist * np.log(hist + 1e-12))
    return float(entropy)


def partial_correlation(x: list, y: list, z: list) -> float:
    """Compute partial correlation of x and y controlling for z."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)

    def _residuals(a, b):
        if np.std(b) < 1e-12:
            return a - a.mean()
        slope = np.cov(a, b)[0, 1] / np.var(b)
        return a - slope * b - (a.mean() - slope * b.mean())

    rx = _residuals(x, z)
    ry = _residuals(y, z)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])
