"""
Evaluation module for fabric defect classification models.

Computes metrics as defined in Section 4.3:
- Accuracy: (TP + TN) / (TP + TN + FP + FN)
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: 2 * (Precision * Recall) / (Precision + Recall)

Also generates:
- Confusion matrix
- Per-class metrics
- Comparison bar charts (Figure 8, 9)
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DEFECT_CLASSES, IDX_TO_CLASS

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    results_dir: str = "results",
    model_name: str = "dcnn",
) -> dict:
    """
    Comprehensive evaluation of a trained model.

    Returns dict with accuracy, precision, recall, f1, per-class metrics.
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_preds = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)

            start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

            import time
            t0 = time.perf_counter()

            if start:
                start.record()
            outputs = model(images)
            if end:
                end.record()
                torch.cuda.synchronize()
                inference_times.append(start.elapsed_time(end) / 1000)
            else:
                inference_times.append(time.perf_counter() - t0)

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    avg_time = np.mean(inference_times)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "avg_inference_time": avg_time,
    }

    logger.info("Evaluation Results for %s:", model_name)
    logger.info("  Accuracy:  %.2f%%", accuracy)
    logger.info("  Precision: %.2f%%", precision)
    logger.info("  Recall:    %.2f%%", recall)
    logger.info("  F1-Score:  %.2f%%", f1)
    logger.info("  Avg Inference Time: %.4fs", avg_time)

    # Classification report
    class_names = [IDX_TO_CLASS.get(i, f"class_{i}") for i in range(len(DEFECT_CLASSES))]
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    logger.info("\nClassification Report:\n%s", report)

    with open(results_path / f"{model_name}_report.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Precision: {precision:.2f}%\n")
        f.write(f"Recall: {recall:.2f}%\n")
        f.write(f"F1-Score: {f1:.2f}%\n")
        f.write(f"Avg Inference Time: {avg_time:.4f}s\n\n")
        f.write(report)

    # Confusion matrix
    _plot_confusion_matrix(all_labels, all_preds, class_names, results_path, model_name)

    return metrics


def _plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    results_dir: Path,
    model_name: str,
):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.savefig(results_dir / f"{model_name}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results_dir: str = "results"):
    """
    Generate comparison bar charts as in Figures 8 and 9 of the paper.
    Compares DCNN against baseline methods.
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Table 3 from the paper
    methods = ["CNN", "KNN", "RCNN", "U-Net", "SegNet", "DCNN\n(Ours)"]
    accuracy = [89.71, 90.62, 94.35, 92.78, 93.63, 96.29]
    recall = [83.6, 84.7, 89.5, 75.2, 70.6, 90.4]
    precision = [89.6, 73.8, 88.6, 89.6, 91.5, 94.5]
    f1_measure = [76.5, 83.2, 90.3, 90.2, 80.5, 92.6]

    colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFCC99", "#FF99CC", "#99CCFF"]

    # Figure 8: Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, accuracy, color=colors, edgecolor="gray", linewidth=0.5)
    for bar, val in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Comparison of Different Models")
    ax.set_ylim(85, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(results_path / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 9: Multi-metric comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods))
    width = 0.2

    bars1 = ax.barh(x + 1.5*width, precision, width, label="Precision", color="#4CAF50")
    bars2 = ax.barh(x + 0.5*width, recall, width, label="Recall", color="#2196F3")
    bars3 = ax.barh(x - 0.5*width, f1_measure, width, label="F-Measure", color="#FF9800")
    bars4 = ax.barh(x - 1.5*width, accuracy, width, label="Accuracy", color="#9C27B0")

    ax.set_yticks(x)
    ax.set_yticklabels(methods)
    ax.set_xlabel("Scores (%)")
    ax.set_title("Comparison of Different Techniques")
    ax.legend(loc="lower right")
    ax.set_xlim(60, 105)
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(results_path / "metric_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Table 4: Computation time comparison
    comp_methods = ["DCNN (ours)", "CNN", "KNN", "RCNN", "U-NET", "SEGNET"]
    comp_times = [0.16, 0.23, 0.19, 0.18, 0.17, 0.20]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(comp_methods, comp_times, color=colors, edgecolor="gray")
    for bar, val in zip(bars, comp_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.2f}s", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Computational Time Comparison")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(results_path / "computation_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Comparison plots saved to %s", results_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_model_comparison()
