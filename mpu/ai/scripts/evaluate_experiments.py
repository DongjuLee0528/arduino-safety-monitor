"""
Post-training evaluation for Exp-A (EfficientNet-B0 lr=0.0003) and
Exp-B (MobileNetV3-Large lr=0.001).

For each experiment:
  1. Load best checkpoint, run production-path val inference, compute metrics.
  2. Export run-local ONNX (opset 18), run ONNX checker + ORT inference.
  3. Verify PyTorch <-> ONNX parity.
  4. Threshold sweep P(helmet) in [0.50, 0.95] step 0.01.
  5. Save all artifacts to mpu/ai/comparisons/four_model_comparison/.

Threshold disclaimer:
  Mac M3 Max benchmark results only. NOT predictive of Arduino UNO Q latency.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnx.checker
import onnxruntime as ort
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mpu.ai.train import build_data_loaders, build_model
from mpu.config import SHEL5K_PATH
import torchvision.transforms as T

PROD_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

OUT_DIR = ROOT / "mpu/ai/comparisons/four_model_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXPECTED_PROD_PTH_SHA256 = "f840f364ec5b1b0e7accc40f976234ffd02dcae39a201e20823be533bf9c23f4"
EXPECTED_PROD_ONNX_SHA256 = "94771807318a0b17c3e649a21607593359924d5573bc0be508d21f7a9229c12b"

BASELINES = {
    "efficientnet_b0_lr0001": ROOT / "mpu/ai/runs/20260820_171048_efficientnet_b0",
    "mobilenet_v3_small_lr0001": ROOT / "mpu/ai/runs/20260820_193048_mobilenet_v3_small",
}


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def file_mb(path: Path) -> float:
    return round(os.path.getsize(path) / 1_048_576, 3)


def run_val_inference(sess: ort.InferenceSession, val_loader) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels_arr, probs_arr) where probs_arr has shape [N, 2]."""
    all_labels, all_probs = [], []
    for images, labels in val_loader:
        batch_np = images.numpy().astype(np.float32)
        logits = sess.run(None, {"input": batch_np})[0]
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        all_labels.extend(labels.tolist())
        all_probs.append(probs)
    return np.array(all_labels), np.vstack(all_probs)


def metrics_at_threshold(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Evaluate classifier with P(helmet) >= threshold -> predict helmet."""
    preds = (probs[:, 1] >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, missed, false_alarm, tp = cm.ravel()
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1], zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    return {
        "threshold": round(threshold, 2),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(macro_f1, 4),
        "no_helmet_recall": round(float(rec[0]), 4),
        "no_helmet_precision": round(float(prec[0]), 4),
        "no_helmet_f1": round(float(f1[0]), 4),
        "helmet_recall": round(float(rec[1]), 4),
        "helmet_precision": round(float(prec[1]), 4),
        "helmet_f1": round(float(f1[1]), 4),
        "missed_violations": int(missed),   # cm[0,1]: actual no_helmet -> pred helmet
        "false_alarms": int(false_alarm),   # cm[1,0]: actual helmet -> pred no_helmet
        "confusion_matrix": cm.tolist(),
    }


def threshold_sweep(labels: np.ndarray, probs: np.ndarray) -> list[dict]:
    results = []
    for t_int in range(50, 96):  # 0.50 to 0.95
        t = t_int / 100.0
        results.append(metrics_at_threshold(labels, probs, t))
    return results


def find_targets(sweep: list[dict]) -> dict:
    """Find Pareto-optimal thresholds and specific recall targets."""
    # Pareto-optimal: no other point has both fewer missed AND fewer false alarms
    pareto = []
    for r in sweep:
        dominated = any(
            (o["missed_violations"] <= r["missed_violations"]
             and o["false_alarms"] <= r["false_alarms"]
             and (o["missed_violations"] < r["missed_violations"]
                  or o["false_alarms"] < r["false_alarms"]))
            for o in sweep
        )
        if not dominated:
            pareto.append(r)

    def find_recall_target(target_recall: float) -> dict | None:
        """Lowest false-alarm threshold achieving no_helmet_recall >= target."""
        candidates = [r for r in sweep if r["no_helmet_recall"] >= target_recall]
        if not candidates:
            return None
        return min(candidates, key=lambda r: (r["false_alarms"], r["threshold"]))

    return {
        "target_97_recall": find_recall_target(0.97),
        "target_98_recall": find_recall_target(0.98),
        "pareto_optimal": pareto,
    }


def onnx_export_and_validate(run_dir: Path, arch: str, pth_path: Path) -> dict:
    """Export run-local ONNX and validate. Returns info dict."""
    onnx_path = run_dir / "best_model.onnx"
    print(f"  Exporting ONNX: {onnx_path}")
    model = build_model(arch, num_classes=2)
    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        export_params=True, opset_version=18,
        do_constant_folding=True,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    proto = onnx.load(str(onnx_path), load_external_data=False)
    opset = next(o.version for o in proto.opset_import if o.domain == "")
    ext_refs = sum(1 for init in proto.graph.initializer if len(init.external_data) > 0)
    proto_full = onnx.load(str(onnx_path))
    onnx.checker.check_model(proto_full, full_check=True)

    ext_path = Path(str(onnx_path) + ".data")
    onnx_mb = file_mb(onnx_path)
    ext_mb = file_mb(ext_path) if ext_path.exists() else 0.0

    # ORT smoke
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": dummy.numpy().astype(np.float32)})[0]
    assert out.shape == (1, 2) and np.all(np.isfinite(out))

    # PyTorch <-> ONNX parity
    rng = np.random.default_rng(42)
    inp_np = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
    with torch.no_grad():
        pt_logits = model(torch.from_numpy(inp_np)).numpy()[0]
    ort_logits = sess.run(None, {"input": inp_np})[0][0]
    max_diff = float(np.abs(pt_logits - ort_logits).max())
    agree = int(np.argmax(pt_logits)) == int(np.argmax(ort_logits))
    print(f"  Parity max|Δlogit|={max_diff:.2e}  agree={agree}")

    return {
        "onnx_path": str(onnx_path),
        "opset": opset,
        "external_refs": ext_refs,
        "onnx_mb": onnx_mb,
        "ext_mb": ext_mb,
        "total_onnx_mb": round(onnx_mb + ext_mb, 3),
        "checker": "PASS",
        "ort_finite": True,
        "parity_max_abs_logit_diff": max_diff,
        "parity_agree": agree,
        "onnx_sha256": sha256(onnx_path),
        "mac_benchmark_disclaimer": (
            "Latency figures (if measured) are from Mac M3 Max CPU. "
            "NOT predictive of Arduino UNO Q on-device latency."
        ),
    }


def evaluate_run(run_dir: Path, val_loader) -> dict:
    """Full evaluation of a completed training run."""
    arch = json.loads((run_dir / "config.json").read_text())["architecture"]
    lr = json.loads((run_dir / "config.json").read_text())["learning_rate"]
    best_metrics = json.loads((run_dir / "best_metrics.json").read_text())
    meta = json.loads((run_dir / "model_metadata.json").read_text())
    pth = run_dir / "best_model.pth"

    print(f"\n[evaluate] {run_dir.name}  arch={arch}  lr={lr}")
    print(f"  Best epoch={best_metrics['best_epoch']}  val_acc={best_metrics['best_val_accuracy']:.2f}%")

    onnx_info = onnx_export_and_validate(run_dir, arch, pth)
    onnx_path = Path(onnx_info["onnx_path"])

    # Production-path val inference
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    labels_arr, probs_arr = run_val_inference(sess, val_loader)
    preds_arr = np.argmax(probs_arr, axis=1)

    cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1])
    acc = accuracy_score(labels_arr, preds_arr)
    prec, rec, f1, sup = precision_recall_fscore_support(
        labels_arr, preds_arr, average=None, labels=[0, 1]
    )
    macro_f1 = float(np.mean(f1))
    missed = int(cm[0, 1])
    false_alarm = int(cm[1, 0])
    print(f"  Production-path:  acc={acc*100:.2f}%  macro_f1={macro_f1:.4f}")
    print(f"  NO_HELMET recall={rec[0]:.4f}  missed={missed}  false_alarm={false_alarm}")
    print(f"  CM: {cm.tolist()}")

    # Threshold sweep
    print("  Running threshold sweep 0.50-0.95 ...")
    sweep = threshold_sweep(labels_arr, probs_arr)
    targets = find_targets(sweep)
    if targets["target_97_recall"]:
        t97 = targets["target_97_recall"]
        print(f"  Recall>=0.97: threshold={t97['threshold']}  missed={t97['missed_violations']}  FA={t97['false_alarms']}")
    else:
        print("  Recall>=0.97: NOT achievable in [0.50, 0.95]")
    if targets["target_98_recall"]:
        t98 = targets["target_98_recall"]
        print(f"  Recall>=0.98: threshold={t98['threshold']}  missed={t98['missed_violations']}  FA={t98['false_alarms']}")
    else:
        print("  Recall>=0.98: NOT achievable in [0.50, 0.95]")

    return {
        "run_id": run_dir.name,
        "architecture": arch,
        "learning_rate": lr,
        "best_epoch": best_metrics["best_epoch"],
        "best_val_loss": best_metrics["best_val_loss"],
        "parameters": meta["parameter_count"],
        "pth_size_mb": file_mb(pth),
        "training_metrics": {
            "val_accuracy": best_metrics["best_val_accuracy"],
            "macro_f1": best_metrics["best_macro_f1"],
            "no_helmet_recall": best_metrics["best_no_helmet_recall"],
            "missed_violations": best_metrics["missed_violation_count"],
            "false_alarms": best_metrics["false_alarm_count"],
        },
        "production_val_metrics": {
            "accuracy": round(float(acc) * 100, 4),
            "macro_f1": round(macro_f1, 4),
            "no_helmet_precision": round(float(prec[0]), 4),
            "no_helmet_recall": round(float(rec[0]), 4),
            "no_helmet_f1": round(float(f1[0]), 4),
            "no_helmet_support": int(sup[0]),
            "helmet_precision": round(float(prec[1]), 4),
            "helmet_recall": round(float(rec[1]), 4),
            "helmet_f1": round(float(f1[1]), 4),
            "helmet_support": int(sup[1]),
            "missed_violations": missed,
            "false_alarms": false_alarm,
            "confusion_matrix": cm.tolist(),
        },
        "onnx": onnx_info,
        "threshold_sweep": sweep,
        "threshold_targets": {
            "target_97_recall": targets["target_97_recall"],
            "target_98_recall": targets["target_98_recall"],
            "pareto_optimal_count": len(targets["pareto_optimal"]),
            "pareto_optimal": targets["pareto_optimal"],
        },
    }


def make_threshold_plot(all_results: list[dict]) -> None:
    """Save missed-violations vs false-alarms plot for all models."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#1d4ed8", "#7c3aed", "#0d9488", "#ea580c"]
    markers = ["o", "s", "^", "D"]
    for i, r in enumerate(all_results):
        sweep = r["threshold_sweep"]
        missed = [s["missed_violations"] for s in sweep]
        fa = [s["false_alarms"] for s in sweep]
        label = f"{r['architecture']} lr={r['learning_rate']}"
        ax.plot(fa, missed, color=colors[i % len(colors)],
                marker=markers[i % len(markers)], markersize=3, label=label, alpha=0.8)
        # Mark default (0.50) point
        ax.scatter([fa[0]], [missed[0]], s=80, color=colors[i % len(colors)],
                   zorder=5, marker=markers[i % len(markers)])
    ax.set_xlabel("False Alarms (actual helmet → pred no_helmet)", fontsize=11)
    ax.set_ylabel("Missed Violations (actual no_helmet → pred helmet)", fontsize=11)
    ax.set_title("Safety Trade-off: Missed Violations vs False Alarms\nThreshold sweep P(helmet) 0.50→0.95\n(Mac M3 Max eval — NOT UNO Q)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    out_path = OUT_DIR / "threshold_tradeoff.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-a-dir", required=True, help="EfficientNet-B0 lr=0.0003 run dir")
    parser.add_argument("--exp-b-dir", required=True, help="MobileNetV3-Large lr=0.001 run dir")
    args = parser.parse_args()

    exp_a_dir = Path(args.exp_a_dir)
    exp_b_dir = Path(args.exp_b_dir)

    print("\n=== Building shared val loader (seed=42, ratio=0.8) ===")
    _, val_loader, split_stats = build_data_loaders(
        shel5k_path=SHEL5K_PATH,
        batch_size=64,
        train_ratio=0.8,
        seed=42,
        num_workers=0,
    )
    print(f"Val samples: {split_stats['val_crops']}  "
          f"no_helmet={split_stats['val_no_helmet']}  helmet={split_stats['val_helmet']}")

    all_results = []

    # Baselines
    for key, run_dir in BASELINES.items():
        if run_dir.exists():
            result = evaluate_run(run_dir, val_loader)
            all_results.append(result)
        else:
            print(f"[WARN] Baseline run dir not found: {run_dir}")

    # New experiments
    for exp_dir in [exp_a_dir, exp_b_dir]:
        if exp_dir.exists():
            result = evaluate_run(exp_dir, val_loader)
            all_results.append(result)
        else:
            print(f"[WARN] Experiment dir not found: {exp_dir}")

    # ── Produce artifacts ───────────────────────────────────────────────────
    print("\n=== Saving comparison artifacts ===")

    # Full JSON
    out_json = OUT_DIR / "four_model_comparison.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  {out_json}")

    # Summary CSV
    out_csv = OUT_DIR / "four_model_comparison.csv"
    fieldnames = [
        "run_id", "architecture", "learning_rate", "best_epoch", "best_val_loss",
        "parameters",
        "val_accuracy", "macro_f1", "no_helmet_recall",
        "missed_violations", "false_alarms",
        "no_helmet_precision", "no_helmet_f1",
        "helmet_recall", "helmet_f1",
        "pth_size_mb", "total_onnx_mb",
        "threshold_97_recall_threshold", "threshold_97_missed", "threshold_97_fa",
        "threshold_98_recall_threshold", "threshold_98_missed", "threshold_98_fa",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            pm = r["production_val_metrics"]
            t97 = r["threshold_targets"]["target_97_recall"] or {}
            t98 = r["threshold_targets"]["target_98_recall"] or {}
            w.writerow({
                "run_id": r["run_id"],
                "architecture": r["architecture"],
                "learning_rate": r["learning_rate"],
                "best_epoch": r["best_epoch"],
                "best_val_loss": round(r["best_val_loss"], 4),
                "parameters": r["parameters"],
                "val_accuracy": pm["accuracy"],
                "macro_f1": pm["macro_f1"],
                "no_helmet_recall": pm["no_helmet_recall"],
                "missed_violations": pm["missed_violations"],
                "false_alarms": pm["false_alarms"],
                "no_helmet_precision": pm["no_helmet_precision"],
                "no_helmet_f1": pm["no_helmet_f1"],
                "helmet_recall": pm["helmet_recall"],
                "helmet_f1": pm["helmet_f1"],
                "pth_size_mb": r["pth_size_mb"],
                "total_onnx_mb": r["onnx"].get("total_onnx_mb", "N/A"),
                "threshold_97_recall_threshold": t97.get("threshold", "N/A"),
                "threshold_97_missed": t97.get("missed_violations", "N/A"),
                "threshold_97_fa": t97.get("false_alarms", "N/A"),
                "threshold_98_recall_threshold": t98.get("threshold", "N/A"),
                "threshold_98_missed": t98.get("missed_violations", "N/A"),
                "threshold_98_fa": t98.get("false_alarms", "N/A"),
            })
    print(f"  {out_csv}")

    # Per-model threshold sweep CSVs
    for r in all_results:
        sweep_path = OUT_DIR / f"threshold_sweep_{r['run_id']}.csv"
        with open(sweep_path, "w", newline="") as f:
            if r["threshold_sweep"]:
                w = csv.DictWriter(f, fieldnames=r["threshold_sweep"][0].keys())
                w.writeheader()
                w.writerows(r["threshold_sweep"])
        print(f"  {sweep_path}")

    # Plot
    make_threshold_plot(all_results)

    # Production file guard
    prod_pth = ROOT / "mpu/ai/models/best_model.pth"
    prod_onnx = ROOT / "mpu/ai/models/best_model.onnx"
    actual_pth_sha = sha256(prod_pth) if prod_pth.exists() else "N/A"
    actual_onnx_sha = sha256(prod_onnx) if prod_onnx.exists() else "N/A"
    guard = {
        "expected_prod_pth_sha256": EXPECTED_PROD_PTH_SHA256,
        "expected_prod_onnx_sha256": EXPECTED_PROD_ONNX_SHA256,
        "prod_pth_sha256": actual_pth_sha,
        "prod_onnx_sha256": actual_onnx_sha,
        "prod_pth_sha256_matches_expected": actual_pth_sha == EXPECTED_PROD_PTH_SHA256,
        "prod_onnx_sha256_matches_expected": actual_onnx_sha == EXPECTED_PROD_ONNX_SHA256,
        "prod_files_untouched": (
            actual_pth_sha == EXPECTED_PROD_PTH_SHA256
            and actual_onnx_sha == EXPECTED_PROD_ONNX_SHA256
        ),
        "mac_latency_disclaimer": (
            "All latency measurements are from Mac M3 Max CPU only. "
            "NOT predictive of Arduino UNO Q on-device latency. "
            "Hardware profiling on UNO Q is required before final deployment decision."
        ),
    }
    if not guard["prod_files_untouched"]:
        raise RuntimeError("Production model SHA-256 guard failed.")
    (OUT_DIR / "production_guard.json").write_text(json.dumps(guard, indent=2))
    print(f"  {OUT_DIR}/production_guard.json")

    # Print summary table
    print("\n" + "="*80)
    print("FOUR-MODEL SUMMARY")
    print("="*80)
    print(f"{'Model':<40} {'missed':>7} {'FA':>6} {'NH_R':>7} {'MacF1':>7} {'Acc%':>7}")
    print("-"*80)
    for r in all_results:
        pm = r["production_val_metrics"]
        name = f"{r['architecture']} lr={r['learning_rate']}"
        print(f"{name:<40} {pm['missed_violations']:>7} {pm['false_alarms']:>6} "
              f"{pm['no_helmet_recall']:>7.4f} {pm['macro_f1']:>7.4f} {pm['accuracy']:>7.2f}")
    print("="*80)
    print("missed = missed violations (cm[0,1]: actual no_helmet -> pred helmet)")
    print("FA     = false alarms (cm[1,0]: actual helmet -> pred no_helmet)")
    print("NH_R   = NO_HELMET recall (safety-critical)")


if __name__ == "__main__":
    main()
