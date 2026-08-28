"""
Training metrics, evaluation helpers, and PPT artifact generation.

Responsibilities:
- Per-epoch metric collection (train/val loss, accuracy, F1)
- Full classification report (precision/recall/F1 per class)
- Safety-specific metrics: false alarm, missed violation
- Confusion matrix PNG
- Training curve PNGs (loss, accuracy)
- Class distribution PNG
- Sample prediction / misclassified-example saving
- CSV / JSON export helpers

Class mapping contract:
  0 = no_helmet  (minority — safety-critical missed detections)
  1 = helmet     (majority)

Design: no external plotting deps beyond matplotlib (already in project).
scikit-learn is used only for classification_report / confusion_matrix —
already installed (1.8.0).
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # headless rendering — no display required
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# Class mapping — must match loader.py ANNOTATION_LABELS and classifier.py
CLASS_NAMES = ["no_helmet", "helmet"]   # index 0 / 1
CLASS_NO_HELMET = 0
CLASS_HELMET = 1


# ─────────────────────────────────────────────────────────────────────────────
# Per-epoch row
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float
    train_f1: float = 0.0
    val_f1: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation result (val set)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float

    # per class
    no_helmet_precision: float
    no_helmet_recall: float
    no_helmet_f1: float
    no_helmet_support: int

    helmet_precision: float
    helmet_recall: float
    helmet_f1: float
    helmet_support: int

    # safety-specific
    # false_alarm: Actual HELMET → Pred NO_HELMET
    # missed_violation: Actual NO_HELMET → Pred HELMET
    false_alarm_count: int = 0
    false_alarm_rate: float = 0.0
    missed_violation_count: int = 0
    missed_violation_rate: float = 0.0

    cm: Optional[np.ndarray] = field(default=None, repr=False)


def compute_eval_metrics(
    all_labels: Sequence[int],
    all_preds: Sequence[int],
) -> EvalResult:
    """Compute full evaluation metrics from label/pred lists."""
    labels = np.array(all_labels)
    preds  = np.array(all_preds)

    accuracy = float((preds == labels).mean()) * 100.0

    cm = confusion_matrix(labels, preds, labels=[CLASS_NO_HELMET, CLASS_HELMET])

    # precision/recall/f1 per class
    p, r, f, sup = precision_recall_fscore_support(
        labels, preds, labels=[CLASS_NO_HELMET, CLASS_HELMET], zero_division=0
    )
    macro_p  = float(np.mean(p))
    macro_r  = float(np.mean(r))
    macro_f1 = float(np.mean(f))
    w_f1     = float(f1_score(labels, preds, average="weighted", zero_division=0))

    # safety counts from confusion matrix
    # cm[i,j] = actual i predicted j
    # false_alarm     = Actual HELMET    → Pred NO_HELMET  = cm[1,0]
    # missed_violation = Actual NO_HELMET → Pred HELMET    = cm[0,1]
    false_alarm_count       = int(cm[CLASS_HELMET,    CLASS_NO_HELMET])
    missed_violation_count  = int(cm[CLASS_NO_HELMET, CLASS_HELMET])

    n_helmet    = int(sup[CLASS_HELMET])
    n_no_helmet = int(sup[CLASS_NO_HELMET])

    false_alarm_rate      = false_alarm_count      / max(n_helmet, 1)
    missed_violation_rate = missed_violation_count / max(n_no_helmet, 1)

    return EvalResult(
        accuracy=accuracy,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        weighted_f1=w_f1,
        no_helmet_precision=float(p[CLASS_NO_HELMET]),
        no_helmet_recall=float(r[CLASS_NO_HELMET]),
        no_helmet_f1=float(f[CLASS_NO_HELMET]),
        no_helmet_support=int(sup[CLASS_NO_HELMET]),
        helmet_precision=float(p[CLASS_HELMET]),
        helmet_recall=float(r[CLASS_HELMET]),
        helmet_f1=float(f[CLASS_HELMET]),
        helmet_support=int(sup[CLASS_HELMET]),
        false_alarm_count=false_alarm_count,
        false_alarm_rate=false_alarm_rate,
        missed_violation_count=missed_violation_count,
        missed_violation_rate=missed_violation_rate,
        cm=cm,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Class weight calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(
    no_helmet_count: int,
    helmet_count: int,
) -> Tuple[float, float]:
    """
    Inverse-frequency class weights: w[c] = N / (2 * count[c]).

    Returns (w_no_helmet, w_helmet).
    w_no_helmet must always be > w_helmet (no_helmet is minority).
    """
    n = no_helmet_count + helmet_count
    w0 = n / (2 * no_helmet_count)
    w1 = n / (2 * helmet_count)
    if w0 < w1:
        raise ValueError(
            f"Class weight invariant violated: w[no_helmet]={w0:.4f} must be >= w[helmet]={w1:.4f}. "
            "Check class counts — no_helmet must be minority or equal."
        )
    return w0, w1


def class_weights_tensor(
    no_helmet_count: int,
    helmet_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a FloatTensor([w_no_helmet, w_helmet]) on device."""
    w0, w1 = compute_class_weights(no_helmet_count, helmet_count)
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def append_epoch_csv(path: str, em: EpochMetrics) -> None:
    """Append one epoch row to metrics.csv; creates header on first write."""
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "epoch", "train_loss", "val_loss",
                "train_accuracy", "val_accuracy", "learning_rate",
                "train_f1", "val_f1",
            ])
        w.writerow([
            em.epoch,
            f"{em.train_loss:.6f}", f"{em.val_loss:.6f}",
            f"{em.train_accuracy:.4f}", f"{em.val_accuracy:.4f}",
            f"{em.learning_rate:.8f}",
            f"{em.train_f1:.4f}", f"{em.val_f1:.4f}",
        ])


def write_class_metrics_csv(path: str, ev: EvalResult) -> None:
    """Write per-class metrics to class_metrics.csv."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "label", "precision", "recall", "f1", "support"])
        w.writerow([0, "no_helmet",
                    f"{ev.no_helmet_precision:.4f}",
                    f"{ev.no_helmet_recall:.4f}",
                    f"{ev.no_helmet_f1:.4f}",
                    ev.no_helmet_support])
        w.writerow([1, "helmet",
                    f"{ev.helmet_precision:.4f}",
                    f"{ev.helmet_recall:.4f}",
                    f"{ev.helmet_f1:.4f}",
                    ev.helmet_support])


def write_split_manifest_csv(
    path: str,
    samples: List,         # list of (image_path, bbox, label)
    split_label: str,      # "train" or "val"
) -> None:
    """Write or append rows to split_manifest.csv."""
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["source_image", "split", "label"])
        for img_path, _, label in samples:
            w.writerow([os.path.basename(img_path), split_label, label])


# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

def write_best_metrics_json(
    path: str,
    epoch: int,
    epoch_metrics: EpochMetrics,
    eval_result: EvalResult,
) -> None:
    """Persist the best epoch summary used by training reports and conversion checks."""
    data = {
        "best_epoch": epoch,
        "best_val_loss": epoch_metrics.val_loss,
        "best_val_accuracy": epoch_metrics.val_accuracy,
        "best_macro_f1": eval_result.macro_f1,
        "best_no_helmet_recall": eval_result.no_helmet_recall,
        "best_helmet_recall": eval_result.helmet_recall,
        "missed_violation_count": eval_result.missed_violation_count,
        "missed_violation_rate": eval_result.missed_violation_rate,
        "false_alarm_count": eval_result.false_alarm_count,
        "false_alarm_rate": eval_result.false_alarm_rate,
        "weighted_f1": eval_result.weighted_f1,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_dataset_summary_json(
    path: str,
    *,
    source_images_total: int,
    person_crops_total: int,
    train_source_images: int,
    val_source_images: int,
    train_crops: int,
    val_crops: int,
    train_no_helmet: int,
    train_helmet: int,
    val_no_helmet: int,
    val_helmet: int,
    w_no_helmet: float,
    w_helmet: float,
    min_bbox_dim: int,
    split_seed: int,
    dataset_b_excluded: bool = True,
) -> None:
    data = {
        "source_images_total": source_images_total,
        "person_crops_total": person_crops_total,
        "train_source_images": train_source_images,
        "val_source_images": val_source_images,
        "train_crops": train_crops,
        "val_crops": val_crops,
        "train_class_counts": {
            "no_helmet": train_no_helmet,
            "helmet": train_helmet,
        },
        "val_class_counts": {
            "no_helmet": val_no_helmet,
            "helmet": val_helmet,
        },
        "class_ratio_helmet_to_no_helmet": round(
            (train_helmet + val_helmet) / max(train_no_helmet + val_no_helmet, 1), 3
        ),
        "class_weights": {
            "no_helmet_w0": round(w_no_helmet, 4),
            "helmet_w1":    round(w_helmet, 4),
        },
        "min_bbox_dim": min_bbox_dim,
        "split_seed": split_seed,
        "dataset_b_excluded": dataset_b_excluded,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# PNG artifacts
# ─────────────────────────────────────────────────────────────────────────────

_STYLE = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#f8f9fa",
    "axes.edgecolor":   "#dee2e6",
    "axes.grid":        True,
    "grid.color":       "#dee2e6",
    "grid.linewidth":   0.8,
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  10,
    "lines.linewidth":  2.0,
}

_AMBER  = "#F59E0B"   # helmet / majority
_BLUE   = "#3B82F6"   # no_helmet / minority
_RED    = "#EF4444"
_GREEN  = "#22C55E"


def save_training_curves(
    path_loss: str,
    path_acc: str,
    history: List[EpochMetrics],
) -> None:
    """Generate train_val_loss.png and train_val_accuracy.png."""
    epochs = [m.epoch for m in history]

    with plt.rc_context(_STYLE):
        # Loss
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, [m.train_loss for m in history], color=_AMBER, label="Train Loss")
        ax.plot(epochs, [m.val_loss   for m in history], color=_BLUE,  label="Val Loss", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("Training vs Validation Loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path_loss, dpi=150)
        plt.close(fig)

        # Accuracy
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, [m.train_accuracy for m in history], color=_AMBER, label="Train Accuracy")
        ax.plot(epochs, [m.val_accuracy   for m in history], color=_BLUE,  label="Val Accuracy", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Training vs Validation Accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path_acc, dpi=150)
        plt.close(fig)


def save_class_distribution(
    path: str,
    train_no_helmet: int,
    train_helmet: int,
    val_no_helmet: int,
    val_helmet: int,
) -> None:
    """Generate class_distribution.png for PPT."""
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        for ax, title, nh, h in [
            (axes[0], "Train", train_no_helmet, train_helmet),
            (axes[1], "Val",   val_no_helmet,   val_helmet),
        ]:
            bars = ax.bar(
                ["NO_HELMET", "HELMET"],
                [nh, h],
                color=[_BLUE, _AMBER],
                edgecolor="white",
                linewidth=0.8,
            )
            ax.set_title(f"{title} Class Distribution")
            ax.set_ylabel("Sample Count")
            ax.set_ylim(0, max(nh, h) * 1.2)
            for bar, val in zip(bars, [nh, h]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(nh, h) * 0.02,
                    str(val),
                    ha="center", va="bottom", fontweight="bold", fontsize=11,
                )

        fig.suptitle("Dataset Class Distribution", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def save_confusion_matrix(path: str, ev: EvalResult) -> None:
    """Generate confusion_matrix.png."""
    cm = ev.cm
    if cm is None:
        return

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)

        tick_marks = [0, 1]
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["Pred NO_HELMET", "Pred HELMET"])
        ax.set_yticklabels(["Actual NO_HELMET", "Actual HELMET"])

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", color=color, fontsize=14, fontweight="bold")

        ax.set_title("Confusion Matrix\n(Val Set — Best Epoch)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        # annotate safety metrics below
        fa  = ev.false_alarm_count
        mv  = ev.missed_violation_count
        ax.set_xlabel(
            f"Predicted\n"
            f"False Alarm (Helmet→No): {fa}   Missed Violation (No→Helmet): {mv}",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Sample predictions
# ─────────────────────────────────────────────────────────────────────────────

def save_sample_predictions(
    out_dir: str,
    model: "torch.nn.Module",
    val_loader: "torch.utils.data.DataLoader",
    device: torch.device,
    n_per_class: int = 5,
    run_local_samples: Optional[List] = None,
) -> None:
    """
    Save representative correct predictions to sample_predictions/.

    Saves up to n_per_class correct samples per class.
    Images are unnormalised for visual inspection.
    Original dataset files are NOT modified.
    """
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    saved = {0: 0, 1: 0}
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for images, labels in val_loader:
            images_d = images.to(device)
            logits = model(images_d)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()

            for i in range(len(labels)):
                gt   = int(labels[i])
                pred = int(preds[i])
                if pred != gt:
                    continue
                if saved[gt] >= n_per_class:
                    continue

                conf  = float(probs[i][pred])
                label_name = CLASS_NAMES[gt]
                fname = (
                    f"correct_{label_name}_gt{gt}_pred{pred}_"
                    f"conf{conf:.2f}_{saved[gt]:02d}.png"
                )

                # Unnormalise for display
                arr = images[i].numpy().transpose(1, 2, 0)
                arr = arr * std + mean
                arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

                with plt.rc_context(_STYLE):
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(arr)
                    ax.set_title(
                        f"GT: {CLASS_NAMES[gt]}\nPred: {CLASS_NAMES[pred]} ({conf:.1%})",
                        fontsize=9,
                    )
                    ax.axis("off")
                    fig.tight_layout()
                    fig.savefig(os.path.join(out_dir, fname), dpi=100)
                    plt.close(fig)

                saved[gt] += 1

            if all(v >= n_per_class for v in saved.values()):
                break


def save_misclassified_examples(
    out_dir: str,
    model: "torch.nn.Module",
    val_loader: "torch.utils.data.DataLoader",
    device: torch.device,
    n_each: int = 10,
) -> None:
    """
    Save misclassified examples to misclassified_examples/.

    false_helmet/    → Actual NO_HELMET, Pred HELMET  (missed violation)
    false_no_helmet/ → Actual HELMET, Pred NO_HELMET  (false alarm)

    companion JSON records source image name, GT, pred, confidence.
    """
    dir_missed = os.path.join(out_dir, "false_helmet")      # missed violation
    dir_false  = os.path.join(out_dir, "false_no_helmet")   # false alarm
    os.makedirs(dir_missed, exist_ok=True)
    os.makedirs(dir_false, exist_ok=True)

    model.eval()
    saved_missed = 0   # actual no_helmet → pred helmet
    saved_false  = 0   # actual helmet   → pred no_helmet
    records: Dict[str, List] = {"false_helmet": [], "false_no_helmet": []}

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for images, labels in val_loader:
            images_d = images.to(device)
            logits = model(images_d)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()

            for i in range(len(labels)):
                gt   = int(labels[i])
                pred = int(preds[i])
                if pred == gt:
                    continue
                conf = float(probs[i][pred])

                arr = images[i].numpy().transpose(1, 2, 0)
                arr = arr * std + mean
                arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

                if gt == CLASS_NO_HELMET and pred == CLASS_HELMET:
                    # missed violation
                    if saved_missed >= n_each:
                        continue
                    fname  = f"missed_violation_{saved_missed:03d}_conf{conf:.2f}.png"
                    fpath  = os.path.join(dir_missed, fname)
                    subkey = "false_helmet"
                    saved_missed += 1
                else:
                    # false alarm
                    if saved_false >= n_each:
                        continue
                    fname  = f"false_alarm_{saved_false:03d}_conf{conf:.2f}.png"
                    fpath  = os.path.join(dir_false, fname)
                    subkey = "false_no_helmet"
                    saved_false += 1

                with plt.rc_context(_STYLE):
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(arr)
                    ax.set_title(
                        f"GT: {CLASS_NAMES[gt]}\nPred: {CLASS_NAMES[pred]} ({conf:.1%})",
                        fontsize=9,
                        color=_RED,
                    )
                    ax.axis("off")
                    fig.tight_layout()
                    fig.savefig(fpath, dpi=100)
                    plt.close(fig)

                records[subkey].append({
                    "filename": fname,
                    "gt": CLASS_NAMES[gt],
                    "pred": CLASS_NAMES[pred],
                    "confidence": round(conf, 4),
                })

            if saved_missed >= n_each and saved_false >= n_each:
                break

    for subkey, recs in records.items():
        jpath = os.path.join(
            dir_missed if subkey == "false_helmet" else dir_false,
            "metadata.json",
        )
        with open(jpath, "w") as f:
            json.dump(recs, f, indent=2)
