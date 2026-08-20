"""
PyTorch to ONNX Model Conversion Script

Supports efficientnet_b0 and mobilenet_v3_small.
Architecture is auto-detected from config.json in the same directory as --checkpoint.
Pass --architecture to override or when config.json is absent.

Usage:
    python -m mpu.ai.convert \\
        --checkpoint mpu/ai/runs/<run_id>/best_model.pth \\
        --output     mpu/ai/runs/<run_id>/best_model.onnx

Production model protection:
    Default paths point to run-local artifacts, not production models.
    Use --checkpoint / --output pointing inside a runs/{run_id}/ directory.
    Passing mpu/ai/models/ paths is allowed but must be an explicit choice.
"""

import json
import os
import argparse
from pathlib import Path

import torch
import torch.onnx
import onnx
import onnx.checker

from mpu.ai.train import build_model, SUPPORTED_ARCHITECTURES

# Opset 18: validated against onnxruntime 1.26.
# opset_version=11 caused a silent fallback to opset 18 via onnxscript
# version_converter (axes_input_to_attribute assertion failure on torch 2.12).
# Requesting 18 explicitly avoids the fallback and produces a clean export log.
_ONNX_OPSET = 18


def _detect_architecture(checkpoint_path: str) -> str:
    """Read architecture from config.json in the same directory as the checkpoint.

    Returns the architecture string, or raises FileNotFoundError / KeyError
    with a clear message so the caller can require --architecture.
    """
    config_path = Path(checkpoint_path).parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json not found next to checkpoint ({config_path}). "
            "Pass --architecture explicitly."
        )
    data = json.loads(config_path.read_text())
    if "architecture" not in data:
        raise KeyError(
            f"'architecture' key missing from {config_path}. "
            "Pass --architecture explicitly."
        )
    return data["architecture"]


def convert_to_onnx(model_path: str, output_path: str,
                    architecture: str = "efficientnet_b0") -> bool:
    """Convert a PyTorch checkpoint to ONNX format.

    Args:
        model_path: Path to trained .pth checkpoint.
        output_path: Destination .onnx path.
        architecture: Architecture name matching the checkpoint.

    Returns:
        True on success.

    Raises:
        FileNotFoundError: checkpoint not found.
        RuntimeError: state_dict mismatch or export error.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device("cpu")  # ONNX export runs on CPU

    model = build_model(architecture, num_classes=2)
    try:
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        print(f"Model loaded from: {model_path}")
    except RuntimeError as e:
        raise RuntimeError(
            f"state_dict mismatch loading {model_path} into {architecture}. "
            f"Verify --architecture matches the checkpoint. Detail: {e}"
        )

    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=_ONNX_OPSET,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"ONNX model exported to: {output_path}  (opset {_ONNX_OPSET})")

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation successful")

    return True


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX format")
    parser.add_argument("--checkpoint", "--model-path",
                        dest="model_path", type=str, default=None,
                        help="Path to trained .pth checkpoint")
    parser.add_argument("--output", "--output-path",
                        dest="output_path", type=str, default=None,
                        help="Path for output .onnx")
    parser.add_argument("--architecture",
                        choices=list(SUPPORTED_ARCHITECTURES), default=None,
                        help=(
                            "Model architecture. Auto-detected from config.json "
                            "when omitted; required if config.json is absent."
                        ))
    args = parser.parse_args()

    if args.model_path is None or args.output_path is None:
        parser.error(
            "Both --checkpoint and --output are required.\n"
            "Example:\n"
            "  python -m mpu.ai.convert \\\n"
            "    --checkpoint mpu/ai/runs/<run_id>/best_model.pth \\\n"
            "    --output     mpu/ai/runs/<run_id>/best_model.onnx"
        )

    # Architecture: explicit override > auto-detect from config.json
    if args.architecture is not None:
        architecture = args.architecture
    else:
        try:
            architecture = _detect_architecture(args.model_path)
        except (FileNotFoundError, KeyError) as e:
            parser.error(str(e))

    if architecture not in SUPPORTED_ARCHITECTURES:
        parser.error(
            f"Unsupported architecture {architecture!r}. "
            f"Supported: {SUPPORTED_ARCHITECTURES}"
        )

    print("=== PyTorch to ONNX Model Converter ===")
    print(f"Architecture : {architecture}")
    print(f"Input model  : {args.model_path}")
    print(f"Output path  : {args.output_path}")
    print(f"Opset        : {_ONNX_OPSET}")

    try:
        convert_to_onnx(args.model_path, args.output_path, architecture)
        print("Conversion completed successfully!")
    except Exception as e:
        print(f"Conversion failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
