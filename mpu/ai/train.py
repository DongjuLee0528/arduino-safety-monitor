"""
Helmet Detection Model Training

Supported architectures:
    efficientnet_b0      (default)
    mobilenet_v3_small
    mobilenet_v3_large

Usage:
    # 2-epoch smoke test (EfficientNet-B0)
    python -m mpu.ai.train \\
        --shel5k-path "/Users/dongjulee/AIdatasets/ helmet-safety-robot/raw/9rcv8mm682-4/Safety Helmet Wearing Dataset" \\
        --epochs 2 --batch-size 32 --lr 0.001 --seed 42 \\
        --output-dir mpu/ai/runs

    # 2-epoch smoke test (MobileNetV3-Small)
    python -m mpu.ai.train \\
        --architecture mobilenet_v3_small \\
        --shel5k-path "/Users/dongjulee/AIdatasets/ helmet-safety-robot/raw/9rcv8mm682-4/Safety Helmet Wearing Dataset" \\
        --epochs 2 --batch-size 32 --lr 0.001 --seed 42 \\
        --output-dir mpu/ai/runs

Class mapping:
    0 = no_helmet  (minority, safety-critical)
    1 = helmet     (majority)

Production model protection:
    Training artifacts are written ONLY to mpu/ai/runs/{run_id}/.
    mpu/ai/models/best_model.pth and .onnx are NEVER overwritten here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
)

from mpu.ai.dataset.loader import HelmetDataset, _split_source_groups
from mpu.ai.metrics import (
    EpochMetrics,
    EvalResult,
    append_epoch_csv,
    class_weights_tensor,
    compute_class_weights,
    compute_eval_metrics,
    save_class_distribution,
    save_confusion_matrix,
    save_misclassified_examples,
    save_sample_predictions,
    save_training_curves,
    write_best_metrics_json,
    write_class_metrics_csv,
    write_dataset_summary_json,
    write_split_manifest_csv,
)
from mpu.config import SHEL5K_PATH

# ── Supported architectures ──────────────────────────────────────────────────
SUPPORTED_ARCHITECTURES = ("efficientnet_b0", "mobilenet_v3_small", "mobilenet_v3_large")

# ── Production model paths — NEVER written during training ──────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PRODUCTION_MODELS_DIR = _PROJECT_ROOT / "mpu" / "ai" / "models"
_PROTECTED_FILES = [
    str(_PRODUCTION_MODELS_DIR / "best_model.pth"),
    str(_PRODUCTION_MODELS_DIR / "best_model.onnx"),
    str(_PRODUCTION_MODELS_DIR / "best_model.onnx.data"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set global seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS: torch.manual_seed covers MPS, but some ops remain non-deterministic.
    # ponytail: full determinism on MPS is not guaranteed (Apple limitation);
    #           seed is still set so most ops are reproducible.


def _seeded_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(architecture: str, num_classes: int = 2) -> nn.Module:
    """Build a pretrained model with a num_classes-output head.

    Supported architectures: efficientnet_b0, mobilenet_v3_small, mobilenet_v3_large.
    Raises ValueError for unsupported names so callers fail fast.
    """
    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
        )
    elif architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif architecture == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(
            f"Unsupported architecture: {architecture!r}. "
            f"Supported: {SUPPORTED_ARCHITECTURES}"
        )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Data loading with reproducible split
# ─────────────────────────────────────────────────────────────────────────────

def build_data_loaders(
    shel5k_path: str,
    batch_size: int,
    train_ratio: float,
    seed: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Build train/val DataLoaders from Dataset A only.
    Returns (train_loader, val_loader, split_stats).
    split_stats contains counts used for class weights and dataset_summary.json.
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load full dataset once to get split indices
    full = HelmetDataset(shel5k_path)
    train_idx, val_idx = _split_source_groups(full.samples, train_ratio)
    train_src = {full.samples[i][0] for i in train_idx}
    val_src   = {full.samples[i][0] for i in val_idx}

    # Verify no leakage
    assert train_src.isdisjoint(val_src), "Source-image leakage between train and val!"

    train_samples = [full.samples[i] for i in train_idx]
    val_samples   = [full.samples[i] for i in val_idx]

    train_no_helmet = sum(1 for _, _, l in train_samples if l == 0)
    train_helmet    = sum(1 for _, _, l in train_samples if l == 1)
    val_no_helmet   = sum(1 for _, _, l in val_samples   if l == 0)
    val_helmet      = sum(1 for _, _, l in val_samples   if l == 1)

    # Rebuild datasets with correct transforms
    train_ds = HelmetDataset(shel5k_path, transform=train_transform)
    val_ds   = HelmetDataset(shel5k_path, transform=val_transform)

    train_ds.samples = train_samples
    val_ds.samples   = val_samples

    generator = _seeded_generator(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
        drop_last=True, generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    split_stats = {
        "source_images_total": len(train_src) + len(val_src),
        "person_crops_total": len(full.samples),
        "train_source_images": len(train_src),
        "val_source_images": len(val_src),
        "train_crops": len(train_samples),
        "val_crops": len(val_samples),
        "train_no_helmet": train_no_helmet,
        "train_helmet": train_helmet,
        "val_no_helmet": val_no_helmet,
        "val_helmet": val_helmet,
        "train_samples": train_samples,
        "val_samples": val_samples,
    }
    return train_loader, val_loader, split_stats


# ─────────────────────────────────────────────────────────────────────────────
# Run directory setup
# ─────────────────────────────────────────────────────────────────────────────

def make_run_dir(output_dir: str, run_id: Optional[str], arch: str) -> Tuple[str, str]:
    """Create and return (run_dir, run_id)."""
    if run_id is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{ts}_{arch}"
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "sample_predictions"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "misclassified_examples"), exist_ok=True)
    return run_dir, run_id


def _safe_check_production_overwrite(run_dir: str) -> None:
    """Raise if run_dir would resolve to the production models directory."""
    run_abs = os.path.realpath(run_dir)
    for protected in _PROTECTED_FILES:
        prot_abs = os.path.realpath(os.path.dirname(protected))
        if run_abs == prot_abs or run_abs.startswith(prot_abs + os.sep):
            raise RuntimeError(
                f"Training run_dir '{run_dir}' points to or inside the production "
                f"models directory '{prot_abs}'. This would overwrite production "
                f"models. Specify a different --output-dir."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Git helpers
# ─────────────────────────────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return bool(out)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Config / metadata saving
# ─────────────────────────────────────────────────────────────────────────────

_PRETRAINED_WEIGHTS_NAMES = {
    "efficientnet_b0": "EfficientNet_B0_Weights.IMAGENET1K_V1",
    "mobilenet_v3_small": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
    "mobilenet_v3_large": "MobileNet_V3_Large_Weights.IMAGENET1K_V1",
}


def write_config_json(path: str, args: argparse.Namespace, run_id: str,
                      device: torch.device, w_no_helmet: float, w_helmet: float,
                      architecture: str = "efficientnet_b0") -> None:
    data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "architecture": architecture,
        "pretrained_weights": _PRETRAINED_WEIGHTS_NAMES.get(architecture, "unknown"),
        "classes": ["no_helmet", "helmet"],
        "class_mapping": {"0": "no_helmet", "1": "helmet"},
        "input_shape": [3, 224, 224],
        "seed": args.seed,
        "dataset_path": args.shel5k_path,
        "dataset_b_excluded": True,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "class_weights": {
            "no_helmet_w0": round(w_no_helmet, 4),
            "helmet_w1": round(w_helmet, 4),
            "formula": "N / (2 * class_count)",
        },
        "early_stopping_patience": args.patience,
        "best_model_criterion": "lowest_val_loss",
        "device": str(device),
        "train_augmentation": [
            "Resize(256x256)", "RandomCrop(224)", "RandomHorizontalFlip(0.5)",
            "ColorJitter(b=0.2,c=0.2,s=0.2,h=0.1)", "ToTensor", "ImageNetNorm",
        ],
        "validation_transform": ["Resize(224x224)", "ToTensor", "ImageNetNorm"],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_model_metadata_json(
    path: str,
    model: nn.Module,
    device: torch.device,
    best_epoch: int,
    best_metrics: dict,
    checkpoint_path: str,
    architecture: str = "efficientnet_b0",
) -> None:
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ckpt_mb = os.path.getsize(checkpoint_path) / 1e6 if os.path.exists(checkpoint_path) else 0.0

    data = {
        "architecture": architecture,
        "parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "input_shape": [1, 3, 224, 224],
        "class_mapping": {"0": "no_helmet", "1": "helmet"},
        "checkpoint_size_mb": round(ckpt_mb, 2),
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "device_used": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "python_version": sys.version,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Returns (avg_loss, accuracy, macro_f1)."""
    model.train()
    running_loss = 0.0
    all_labels: List[int] = []
    all_preds: List[int] = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = running_loss / len(loader)
    correct  = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = 100.0 * correct / max(len(all_labels), 1)

    from sklearn.metrics import f1_score as _f1
    f1 = float(_f1(all_labels, all_preds, average="macro", zero_division=0))
    return avg_loss, accuracy, f1


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, List[int], List[int]]:
    """Returns (avg_loss, accuracy, macro_f1, all_labels, all_preds)."""
    model.eval()
    running_loss = 0.0
    all_labels: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = running_loss / len(loader)
    correct  = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = 100.0 * correct / max(len(all_labels), 1)

    from sklearn.metrics import f1_score as _f1
    f1 = float(_f1(all_labels, all_preds, average="macro", zero_division=0))
    return avg_loss, accuracy, f1, all_labels, all_preds


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train HelmetClassifier")
    parser.add_argument("--architecture", default="efficientnet_b0",
                        choices=list(SUPPORTED_ARCHITECTURES),
                        help="Model architecture (default: efficientnet_b0)")
    parser.add_argument("--shel5k-path", default=SHEL5K_PATH,
                        help="Path to Safety Helmet Wearing Dataset (Dataset A)")
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--lr",          type=float, default=0.001)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--patience",    type=int,   default=5,
                        help="Early stopping patience (default 5)")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--output-dir",  default="mpu/ai/runs",
                        help="Parent directory for run outputs")
    parser.add_argument("--run-id",      default=None,
                        help="Explicit run ID; auto-generated if omitted")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-examples", action="store_true", default=True,
                        help="Save sample predictions and misclassified examples")
    args = parser.parse_args()

    # ── Seed ────────────────────────────────────────────────────────────────
    set_seed(args.seed)

    # ── Device ──────────────────────────────────────────────────────────────
    device = select_device()
    print(f"Device: {device}")

    # ── Run directory ────────────────────────────────────────────────────────
    run_dir, run_id = make_run_dir(args.output_dir, args.run_id, args.architecture)
    _safe_check_production_overwrite(run_dir)
    print(f"Run directory: {run_dir}")

    best_ckpt = os.path.join(run_dir, "best_model.pth")
    last_ckpt = os.path.join(run_dir, "last_model.pth")

    # ── Data ────────────────────────────────────────────────────────────────
    print("Loading dataset …")
    train_loader, val_loader, stats = build_data_loaders(
        args.shel5k_path, args.batch_size, args.train_ratio,
        args.seed, args.num_workers,
    )
    print(
        f"Train: {stats['train_crops']} crops "
        f"({stats['train_helmet']} helmet / {stats['train_no_helmet']} no_helmet) "
        f"from {stats['train_source_images']} images"
    )
    print(
        f"Val  : {stats['val_crops']} crops "
        f"({stats['val_helmet']} helmet / {stats['val_no_helmet']} no_helmet) "
        f"from {stats['val_source_images']} images"
    )

    # ── Class weights ────────────────────────────────────────────────────────
    w0, w1 = compute_class_weights(stats["train_no_helmet"], stats["train_helmet"])
    weights_tensor = class_weights_tensor(
        stats["train_no_helmet"], stats["train_helmet"], device=device
    )
    print(f"Class weights: no_helmet(0)={w0:.4f}  helmet(1)={w1:.4f}")

    # ── Config & dataset summary ──────────────────────────────────────────────
    write_config_json(
        os.path.join(run_dir, "config.json"), args, run_id, device, w0, w1,
        architecture=args.architecture,
    )
    write_dataset_summary_json(
        os.path.join(run_dir, "dataset_summary.json"),
        source_images_total=stats["source_images_total"],
        person_crops_total=stats["person_crops_total"],
        train_source_images=stats["train_source_images"],
        val_source_images=stats["val_source_images"],
        train_crops=stats["train_crops"],
        val_crops=stats["val_crops"],
        train_no_helmet=stats["train_no_helmet"],
        train_helmet=stats["train_helmet"],
        val_no_helmet=stats["val_no_helmet"],
        val_helmet=stats["val_helmet"],
        w_no_helmet=w0,
        w_helmet=w1,
        min_bbox_dim=HelmetDataset.MIN_DIM,
        split_seed=args.seed,
    )
    # Split manifest
    manifest_path = os.path.join(run_dir, "split_manifest.csv")
    write_split_manifest_csv(manifest_path, stats["train_samples"], "train")
    write_split_manifest_csv(manifest_path, stats["val_samples"], "val")

    # ── Model, loss, optimizer ────────────────────────────────────────────────
    model = build_model(args.architecture, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ── Training loop ────────────────────────────────────────────────────────
    metrics_csv  = os.path.join(run_dir, "metrics.csv")
    best_val_loss = float("inf")
    early_stop_counter = 0
    history: List[EpochMetrics] = []
    best_epoch = 1
    best_eval: Optional[EvalResult] = None
    best_em: Optional[EpochMetrics] = None

    print(f"\nStarting training: {args.epochs} epochs, patience={args.patience}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, all_labels, all_preds = validate_epoch(
            model, val_loader, criterion, device
        )

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        em = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_acc,
            val_accuracy=val_acc,
            learning_rate=current_lr,
            train_f1=train_f1,
            val_f1=val_f1,
        )
        history.append(em)
        append_epoch_csv(metrics_csv, em)

        print(
            f"[{epoch:3d}/{args.epochs}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.1f}%  val_f1={val_f1:.3f}  "
            f"({elapsed:.1f}s)"
        )

        # Always save last checkpoint
        torch.save(model.state_dict(), last_ckpt)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_epoch = epoch
            best_em = em
            best_eval = compute_eval_metrics(all_labels, all_preds)
            torch.save(model.state_dict(), best_ckpt)

            write_best_metrics_json(
                os.path.join(run_dir, "best_metrics.json"),
                best_epoch, best_em, best_eval,
            )
            print(f"  ↳ best checkpoint saved (val_loss={val_loss:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience}). Best epoch: {best_epoch}")
                break

    print(f"\nTraining complete. Best epoch: {best_epoch}")

    # ── PPT artifacts ─────────────────────────────────────────────────────────
    print("Generating PPT artifacts …")
    save_training_curves(
        os.path.join(run_dir, "train_val_loss.png"),
        os.path.join(run_dir, "train_val_accuracy.png"),
        history,
    )
    save_class_distribution(
        os.path.join(run_dir, "class_distribution.png"),
        stats["train_no_helmet"], stats["train_helmet"],
        stats["val_no_helmet"],   stats["val_helmet"],
    )
    if best_eval is not None:
        save_confusion_matrix(os.path.join(run_dir, "confusion_matrix.png"), best_eval)
        write_class_metrics_csv(os.path.join(run_dir, "class_metrics.csv"), best_eval)

    # ── Best-epoch evaluation artifacts ──────────────────────────────────────
    if os.path.exists(best_ckpt) and best_eval is not None:
        # Reload best weights for sample saving
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

        if args.save_examples:
            print("Saving sample predictions …")
            save_sample_predictions(
                os.path.join(run_dir, "sample_predictions"),
                model, val_loader, device,
            )
            print("Saving misclassified examples …")
            save_misclassified_examples(
                os.path.join(run_dir, "misclassified_examples"),
                model, val_loader, device,
            )

    # ── Model metadata ────────────────────────────────────────────────────────
    best_metrics_dict = {}
    if best_eval is not None and best_em is not None:
        best_metrics_dict = {
            "best_epoch": best_epoch,
            "val_loss": best_em.val_loss,
            "val_accuracy": best_em.val_accuracy,
            "macro_f1": best_eval.macro_f1,
            "no_helmet_recall": best_eval.no_helmet_recall,
            "missed_violation_count": best_eval.missed_violation_count,
        }
    write_model_metadata_json(
        os.path.join(run_dir, "model_metadata.json"),
        model, device, best_epoch, best_metrics_dict, best_ckpt,
        architecture=args.architecture,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Run ID      : {run_id}")
    print(f"Run dir     : {run_dir}")
    print(f"Best epoch  : {best_epoch}")
    if best_em:
        print(f"Best val_loss   : {best_em.val_loss:.4f}")
        print(f"Best val_acc    : {best_em.val_accuracy:.2f}%")
    if best_eval:
        print(f"Macro F1        : {best_eval.macro_f1:.4f}")
        print(f"Missed violations: {best_eval.missed_violation_count}")
        print(f"False alarms     : {best_eval.false_alarm_count}")
    print(f"{'='*60}")
    print(f"\nTo export ONNX from this run:")
    print(f"  python -m mpu.ai.convert \\")
    print(f"    --checkpoint {best_ckpt} \\")
    print(f"    --output {os.path.join(run_dir, 'best_model.onnx')}")


if __name__ == "__main__":
    main()
