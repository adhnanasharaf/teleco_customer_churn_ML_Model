"""
Evaluation and Visualization Module for Telco Customer Churn
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Set clean modern style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.titlesize": 14,
})


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes all standard and business evaluation metrics for a trained model pipeline.
    """
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df_vals = model.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-df_vals))
    else:
        y_proba = model.predict(X_test)

    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "threshold": float(threshold),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    return metrics, y_proba, y_pred


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1") -> float:
    """
    Finds probability threshold that maximizes F1 or Youden's J statistic.
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_score = -1.0

    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, pred, zero_division=0)
        elif metric == "youden":
            cm = confusion_matrix(y_true, pred)
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sens + spec - 1
        else:
            score = f1_score(y_true, pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = t

    return round(float(best_thresh), 3)


def plot_roc_curve(y_test: np.ndarray, y_proba: np.ndarray, auc_score: float, save_path: Optional[Path] = None):
    """Plots and saves the ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"ROC Curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1")
    plt.tight_layout()

    out = save_path or (PLOTS_DIR / "roc_curve.png")
    plt.savefig(out, dpi=300)
    plt.close()


def plot_confusion_matrix(cm_dict: Dict[str, int], save_path: Optional[Path] = None):
    """Plots and saves the Confusion Matrix heatmap."""
    cm = np.array([
        [cm_dict["tn"], cm_dict["fp"]],
        [cm_dict["fn"], cm_dict["tp"]]
    ])
    total = np.sum(cm)
    percentages = cm / total * 100

    labels = np.array([
        [f"TN\n{cm[0, 0]:,}\n({percentages[0, 0]:.1f}%)", f"FP\n{cm[0, 1]:,}\n({percentages[0, 1]:.1f}%)"],
        [f"FN\n{cm[1, 0]:,}\n({percentages[1, 0]:.1f}%)", f"TP\n{cm[1, 1]:,}\n({percentages[1, 1]:.1f}%)"]
    ])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Retained (No)", "Churned (Yes)"],
        yticklabels=["Retained (No)", "Churned (Yes)"],
        annot_kws={"size": 12, "weight": "semibold"}
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    out = save_path or (PLOTS_DIR / "confusion_matrix.png")
    plt.savefig(out, dpi=300)
    plt.close()


def plot_feature_importance(feature_names: List[str], importances: np.ndarray, top_n: int = 15, save_path: Optional[Path] = None):
    """Plots and saves the top N feature importances or coefficient magnitudes."""
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_imp,
        x="Importance",
        y="Feature",
        palette="crest"
    )
    plt.title(f"Top {top_n} Key Drivers of Customer Churn")
    plt.xlabel("Importance Score / Model Weight")
    plt.ylabel("Feature")
    plt.tight_layout()

    out = save_path or (PLOTS_DIR / "feature_importance.png")
    plt.savefig(out, dpi=300)
    plt.close()


def plot_model_comparison(benchmark_results: List[Dict[str, Any]], save_path: Optional[Path] = None):
    """Plots comparative metric bar charts across benchmarked models."""
    df_bm = pd.DataFrame(benchmark_results)
    metrics_to_plot = ["accuracy", "recall", "f1", "roc_auc"]
    df_melt = df_bm.melt(id_vars=["model_name"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")

    metric_labels = {
        "accuracy": "Accuracy",
        "recall": "Recall",
        "f1": "F1-Score",
        "roc_auc": "ROC-AUC"
    }
    df_melt["Metric"] = df_melt["Metric"].map(metric_labels)

    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=df_melt, x="model_name", y="Score", hue="Metric", palette="Set2")
    plt.title("Model Performance Benchmark Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Score (0 - 1)")
    plt.ylim([0.4, 1.0])
    plt.legend(loc="lower right", frameon=True)
    plt.xticks(rotation=15)
    plt.tight_layout()

    out = save_path or (PLOTS_DIR / "model_comparison.png")
    plt.savefig(out, dpi=300)
    plt.close()
