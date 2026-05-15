import os
import argparse
import torch
import torch.onnx
import onnx
import onnx.checker
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


def create_model(num_classes=2):
    """train.py와 동일한 모델 구조 생성"""
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(in_features, num_classes)
    )

    return model


def convert_to_onnx(model_path, output_path):
    """PyTorch 모델을 ONNX 포맷으로 변환"""
    device = torch.device("cpu")  # ONNX 변환은 CPU에서 진행

    # 모델 로드
    model = create_model(num_classes=2)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from: {model_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    model.eval()

    # 더미 입력 생성 (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # ONNX로 변환
        torch.onnx.export(
            model,                     # 모델
            dummy_input,              # 더미 입력
            output_path,              # 출력 경로
            export_params=True,       # 모델 파라미터 저장
            opset_version=11,         # ONNX 연산자 셋 버전
            do_constant_folding=True, # 상수 폴딩 최적화
            input_names=['input'],    # 입력 이름
            output_names=['output'],  # 출력 이름
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"ONNX model exported to: {output_path}")

        # ONNX 모델 검증
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation successful")

        return True

    except Exception as e:
        print(f"Error during ONNX conversion: {e}")
        return False


def main():
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