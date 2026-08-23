"""
Configuration settings for the helmet detection system.

This module defines paths, communication settings, camera settings, training
hyperparameters, and alert thresholds. Validation helpers are available but are
not run at import time.

Runtime validation:
    Call validate_runtime_models() at startup to fail fast when required
    model files are missing before any inference session is created.
"""

import os
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Dataset configuration
# Paths to helmet detection training datasets
SHEL5K_PATH = os.path.normpath(os.path.expanduser("~/AIdatasets/helmet-safety-robot/raw/9rcv8mm682-4/Safety Helmet Wearing Dataset"))  # Safety Helmet Wearing Dataset (Dataset A)
SHWD_PATH = os.path.normpath(os.path.expanduser("~/AIdatasets/helmet-safety-robot/raw/VOC2028"))  # Dataset B — excluded from baseline training

# Server configuration for remote alert transmission
DEFAULT_SERVER_URL = "http://localhost:3000/api/alert"  # HTTP endpoint for sending safety alerts

# UNO Q Bridge communication configuration for Arduino MCU IPC
DEFAULT_SERIAL_PORT = "unoq-bridge"  # Compatibility label; App Lab uses Bridge.call, not /dev/tty*
DEFAULT_BAUDRATE = 115200            # Legacy desktop serial compatibility value
DEFAULT_TIMEOUT = 1.0                # Legacy desktop serial compatibility value

# AI model file paths
AI_MODELS_DIR = str(_PROJECT_ROOT / "mpu" / "ai" / "models")           # Directory containing AI model files
BEST_MODEL_PATH = os.path.join(AI_MODELS_DIR, "best_model.pth")        # PyTorch helmet classifier model
ONNX_MODEL_PATH = os.path.join(AI_MODELS_DIR, "best_model.onnx")       # ONNX helmet classifier for inference
MOBILENET_SSD_PATH = os.path.join(AI_MODELS_DIR, "ssd_mobilenet_v1_12.onnx")  # Person detection model

# Training hyperparameters for AI model training
DEFAULT_BATCH_SIZE = 32        # Number of samples per training batch
DEFAULT_EPOCHS = 30            # Maximum number of training epochs
DEFAULT_LEARNING_RATE = 0.001  # Adam optimizer learning rate
DEFAULT_TRAIN_RATIO = 0.8      # Ratio of data used for training (vs validation)

# Camera capture configuration
DEFAULT_CAMERA_INDEX = 0  # Default camera device index (0 = first camera)
CAMERA_WIDTH = 640        # Camera frame width in pixels
CAMERA_HEIGHT = 480       # Camera frame height in pixels

# Alert system configuration
DEFAULT_DETECTION_THRESHOLD = 3  # Number of consecutive detections before triggering alert
DEFAULT_COOLDOWN_TIME = 5.0      # Minimum time between alerts in seconds
HELMET_ACCEPTANCE_THRESHOLD = 0.83  # Predict HELMET only when P(helmet) is at least this value
WARNING_CLEAR_THRESHOLD = 2  # Consecutive non-violation frames required to clear an active warning
TRACK_IOU_THRESHOLD = 0.3  # Minimum IoU for matching a detection to an existing temporary track
TRACK_CENTROID_DISTANCE_RATIO = 0.2  # 160px at requested 640x480; scales with actual frame diagonal
TRACK_TTL_SECONDS = 2.0  # Seconds to retain an unmatched temporary track before pruning
MAX_ACTIVE_TRACKS = 8  # Bound temporary in-memory person tracks on edge hardware

# Image processing configuration
MODEL_INPUT_SIZE = 224           # Input size for helmet classifier model (224x224)
DETECTOR_INPUT_SIZE = 300        # Input size for MobileNet SSD person detector (300x300)

# Detection configuration
PERSON_CLASS_ID = 1              # COCO dataset class ID for person (used by MobileNet SSD)
DETECTOR_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score for person detection

# Network configuration
HTTP_TIMEOUT = 10                # HTTP request timeout in seconds
RETRY_DELAY = 1                  # Delay between retry attempts in seconds

# Display configuration
# Set to True only when a physical display is available (development / demo).
# Keep False for headless deployment on the UNO Q MPU.
ENABLE_DISPLAY = False

# Hardware-free development mode.
# When True: camera and serial hardware failures are logged as warnings and
# skipped rather than raising RuntimeError, so the system can run without
# physical hardware attached (App Lab container without USB camera / Arduino).
#
# Default: False (strict / production).  Must be explicitly activated.
# Activation: set environment variable APP_LAB_DEV_MODE=true (or 1 / yes).
_dev_raw = os.environ.get("APP_LAB_DEV_MODE", "").strip().lower()
APP_LAB_DEV_MODE: bool = _dev_raw in {"true", "1", "yes"}

def validate_dataset_paths():
    """Validate dataset paths and show warnings if not found."""
    paths = {
        "SHEL5K": SHEL5K_PATH,
        "SHWD": SHWD_PATH,
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            warnings.warn(
                f"{name} dataset path not found: {path}\n"
                "Please ensure the dataset is downloaded and path is correct in config.py",
                UserWarning,
                stacklevel=2,
            )


def validate_model_files():
    """Validate model files and show warnings if not found."""
    model_files = {
        "Best Model (PyTorch)": BEST_MODEL_PATH,
        "Helmet Classifier (ONNX)": ONNX_MODEL_PATH,
        "Person Detector (ONNX)": MOBILENET_SSD_PATH,
    }

    for name, path in model_files.items():
        if not os.path.exists(path):
            warnings.warn(
                f"{name} file not found: {path}\n"
                "Please ensure the model file is available or retrain the model",
                UserWarning,
                stacklevel=2,
            )


def validate_runtime_models():
    """Raise FileNotFoundError if any model required for runtime inference is missing.

    This function is intended to be called once at system startup so that
    missing model files are detected immediately with a clear error message
    rather than causing an obscure failure inside an ONNX InferenceSession.

    Checked files:
        - ssd_mobilenet_v1_12.onnx   : person detector (PersonDetector)
        - best_model.onnx            : helmet classifier (HelmetClassifier)
        - best_model.onnx.data       : external weight data for best_model.onnx

    Not checked here:
        - best_model.pth is a training artefact and is not used at runtime.
    """
    runtime_files = {
        "Person detector model": MOBILENET_SSD_PATH,
        "Helmet classifier ONNX model": ONNX_MODEL_PATH,
        "Helmet classifier ONNX external data": ONNX_MODEL_PATH + ".data",
    }

    for description, path in runtime_files.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{description} not found: {path}\n"
                "Ensure the file is present in mpu/ai/models/ before starting the system."
            )


def validate_server_connection(url: str = DEFAULT_SERVER_URL):
    """Validate server connection and show warning if failed."""
    import requests

    try:
        requests.get(url, timeout=5)
        return True
    except requests.exceptions.RequestException as e:
        warnings.warn(
            f"Server connection failed: {url}\n"
            f"Error: {e}\n"
            "Alert transmission will be disabled but system will continue",
            UserWarning,
            stacklevel=2,
        )
        return False
