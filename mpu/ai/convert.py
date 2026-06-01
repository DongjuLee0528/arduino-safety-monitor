"""
PyTorch to ONNX Model Conversion Script

This script converts trained PyTorch helmet detection models to ONNX format
for deployment in production environments. ONNX models provide better
cross-platform compatibility and optimized inference performance.

Key Features:
- Converts PyTorch .pth models to ONNX format
- Maintains exact model architecture compatibility
- Validates converted model integrity
- Optimizes for inference with constant folding
- Supports dynamic batch size for flexible deployment

Conversion Process:
1. Load trained PyTorch model weights
2. Create identical model architecture
3. Export to ONNX with optimization flags
4. Validate ONNX model structure
5. Save optimized .onnx file

Usage:
    python convert.py --model-path models/best_model.pth --output-path models/best_model.onnx
    python convert.py  # Uses default paths
"""

import os
import argparse
import torch
import torch.onnx
import onnx
import onnx.checker
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


def create_model(num_classes=2):
    """
    Create the same model architecture as used in train.py.
    This ensures compatibility when loading trained weights.

    Args:
        num_classes: Number of output classes (default: 2)

    Returns:
        Uninitialized model with correct architecture
    """
    # Create EfficientNet-B0 base model
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace classifier to match training configuration
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(in_features, num_classes)
    )

    return model


def convert_to_onnx(model_path, output_path):
    """
    Convert PyTorch model to ONNX format for optimized inference.

    Args:
        model_path: Path to trained PyTorch model (.pth file)
        output_path: Path for output ONNX model (.onnx file)

    Returns:
        True if conversion successful, False otherwise

    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    device = torch.device("cpu")  # ONNX conversion must run on CPU

    # Load trained model weights
    model = create_model(num_classes=2)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from: {model_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    model.eval()  # Set to evaluation mode for inference

    # Create dummy input tensor (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Export model to ONNX format
        torch.onnx.export(
            model,                     # Trained PyTorch model
            dummy_input,              # Example input tensor
            output_path,              # Output file path
            export_params=True,       # Export trained parameters
            opset_version=11,         # ONNX operator set version
            do_constant_folding=True, # Optimize constant operations
            input_names=['input'],    # Input tensor names
            output_names=['output'],  # Output tensor names
            dynamic_axes={            # Allow variable batch size
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"ONNX model exported to: {output_path}")

        # Validate exported ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation successful")

        return True

    except Exception as e:
        print(f"Error during ONNX conversion: {e}")
        return False


def main():
    """
    Main function for command-line model conversion.
    Handles argument parsing and orchestrates the conversion process.
    """
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX format')
    parser.add_argument('--model-path', type=str,
                       default='mpu/ai/models/best_model.pth',
                       help='Path to PyTorch model file (.pth)')
    parser.add_argument('--output-path', type=str,
                       default='mpu/ai/models/best_model.onnx',
                       help='Path for output ONNX model file (.onnx)')
    args = parser.parse_args()

    print("=== PyTorch to ONNX Model Converter ===")
    print(f"Input model: {args.model_path}")
    print(f"Output path: {args.output_path}")

    try:
        success = convert_to_onnx(args.model_path, args.output_path)
        if success:
            print("Conversion completed successfully!")
        else:
            print("Conversion failed!")
            exit(1)

    except Exception as e:
        print(f"Conversion failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()