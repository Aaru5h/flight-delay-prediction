"""Calibration evaluation, fitting, and reliability diagram generation (Phase 5, Step 4)."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression


def fit_probability_calibrator(
    y_val: np.ndarray, y_val_prob: np.ndarray
) -> IsotonicRegression:
    """Fit an isotonic probability calibrator on validation data only."""
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(y_val_prob, y_val)
    return ir


def plot_reliability_diagrams(
    models_data: dict[str, tuple[np.ndarray, np.ndarray]],
    split_name: str,
    output_path: Path,
    n_bins: int = 10,
) -> None:
    """Produce publication-quality reliability diagrams with histogram bin counts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 9), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
    colors = {"Logistic Regression": "#1f77b4", "XGBoost": "#ff7f0e"}

    for name, (y_true, y_prob) in models_data.items():
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        color = colors.get(name, "#2ca02c")
        ax1.plot(
            prob_pred,
            prob_true,
            "s-",
            label=name,
            color=color,
            linewidth=2,
            markersize=6,
        )

        ax2.hist(
            y_prob,
            bins=np.linspace(0, 1, n_bins + 1),
            histtype="step",
            linewidth=1.5,
            label=name,
            color=color,
        )

    ax1.set_ylabel("Empirical Delay Fraction (True Positives)", fontsize=11)
    ax1.set_title(
        f"Reliability Diagram (Calibration Curve) — {split_name}",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_ylim(-0.02, 1.02)

    ax2.set_xlabel("Mean Predicted Probability of Delay", fontsize=11)
    ax2.set_ylabel("Flights (log)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_yscale("log")
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pr_and_roc_curves(
    models_data: dict[str, tuple[np.ndarray, np.ndarray]],
    split_name: str,
    output_path: Path,
) -> None:
    """Plot Precision-Recall and ROC curves."""
    from sklearn.metrics import precision_recall_curve, roc_curve, auc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"Logistic Regression": "#1f77b4", "XGBoost": "#ff7f0e"}

    first_y = next(iter(models_data.values()))[0]
    prevalence = float(np.mean(first_y))

    ax_pr.axhline(
        y=prevalence,
        color="gray",
        linestyle="--",
        label=f"No-skill Baseline ({prevalence:.1%})",
    )
    ax_roc.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.50)", alpha=0.7)

    for name, (y_true, y_prob) in models_data.items():
        color = colors.get(name, "#2ca02c")

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(rec, prec)
        ax_pr.plot(
            rec,
            prec,
            color=color,
            linewidth=2,
            label=f"{name} (PR AUC = {pr_auc:.4f})",
        )

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(
            fpr,
            tpr,
            color=color,
            linewidth=2,
            label=f"{name} (ROC AUC = {roc_auc:.4f})",
        )

    ax_pr.set_xlabel("Recall (Delay Coverage)", fontsize=11)
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax_pr.set_title(f"Precision-Recall Curves — {split_name}", fontsize=12, fontweight="bold")
    ax_pr.legend(loc="upper right", frameon=True)
    ax_pr.grid(True, linestyle=":", alpha=0.6)
    ax_pr.set_ylim(-0.02, 1.02)

    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax_roc.set_title(f"ROC Curves — {split_name}", fontsize=12, fontweight="bold")
    ax_roc.legend(loc="lower right", frameon=True)
    ax_roc.grid(True, linestyle=":", alpha=0.6)
    ax_roc.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
