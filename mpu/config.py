import os
import warnings
import requests
from typing import List

def _validate_dataset_paths():
    """Validate dataset paths and show warnings if not found"""
    paths = {
        "SHEL5K": SHEL5K_PATH,
        "SHWD": SHWD_PATH
    }

    for name, path in paths.items():
        expanded_path = os.path.expanduser(path)
        if not os.path.exists(expanded_path):
            warnings.warn(
                f"{name} dataset path not found: {expanded_path}\n"
                f"Please ensure the dataset is downloaded and path is correct in config.py",
                UserWarning,
                stacklevel=2
            )

# Dataset configuration
SHEL5K_PATH = "~/Documents/AIdatasets/ helmet-safety-robot/raw/9rcv8mm682-4/Safety Helmet Wearing Dataset"
SHWD_PATH = "~/Documents/AIdatasets/ helmet-safety-robot/raw/VOC2028"

def validate_model_files():
    """Validate ONNX model files and show warnings if not found"""
    model_files = {
        "Best Model": BEST_MODEL_PATH,
        "ONNX Model": ONNX_MODEL_PATH,
        "MobileNet SSD": MOBILENET_SSD_PATH
    }

    for name, path in model_files.items():
        if not os.path.exists(path):
            warnings.warn(
                f"{name} file not found: {path}\n"
                f"Please ensure the model file is available or retrain the model",
                UserWarning,
                stacklevel=2
            )

def validate_server_connection(url: str = DEFAULT_SERVER_URL):
    """Validate server connection and show warning if failed"""
    try:
        response = requests.get(url, timeout=5)
        return True
    except requests.exceptions.RequestException as e:
        warnings.warn(
            f"Server connection failed: {url}\n"
            f"Error: {e}\n"
            f"Alert transmission will be disabled but system will continue",
            UserWarning,
            stacklevel=2
        )
        return False

# Validate paths and connections on import
_validate_dataset_paths()
validate_model_files()
validate_server_connection()

# Server configuration
DEFAULT_SERVER_URL = "http://localhost:3000/api/alert"

# Serial communication configuration
DEFAULT_SERIAL_PORT = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 115200
DEFAULT_TIMEOUT = 1.0

# Model paths
AI_MODELS_DIR = "mpu/ai/models"
BEST_MODEL_PATH = os.path.join(AI_MODELS_DIR, "best_model.pth")
ONNX_MODEL_PATH = os.path.join(AI_MODELS_DIR, "best_model.onnx")
MOBILENET_SSD_PATH = os.path.join(AI_MODELS_DIR, "mobilenet_ssd.onnx")

# Training configuration
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_TRAIN_RATIO = 0.8

# Camera configuration
DEFAULT_CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Alert manager configuration
DEFAULT_DETECTION_THRESHOLD = 3
DEFAULT_COOLDOWN_TIME = 5.0