"""
Configuration Settings for Helmet Detection System

This module contains all configuration parameters for the helmet detection system including:
- Dataset paths and validation
- Model file locations
- Communication settings (serial, server)
- Camera configuration
- Training hyperparameters
- Alert system settings

The module automatically validates paths and connections on import to ensure
system components are properly configured before operation.
"""

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
        if not os.path.exists(path):
            warnings.warn(
                f"{name} dataset path not found: {path}\n"
                f"Please ensure the dataset is downloaded and path is correct in config.py",
                UserWarning,
                stacklevel=2
            )

# Dataset configuration
# Paths to helmet detection training datasets
SHEL5K_PATH = os.path.normpath(os.path.expanduser("~/Documents/AIdatasets/helmet-safety-robot/raw/9rcv8mm682-4/Safety Helmet Wearing Dataset"))  # Safety Helmet Wearing Dataset
SHWD_PATH = os.path.normpath(os.path.expanduser("~/Documents/AIdatasets/helmet-safety-robot/raw/VOC2028"))  # SHWD (Safety Helmet Workers Dataset)

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

# Server configuration for remote alert transmission
DEFAULT_SERVER_URL = "http://localhost:3000/api/alert"  # HTTP endpoint for sending safety alerts

# Serial communication configuration for Arduino bridge
DEFAULT_SERIAL_PORT = "/dev/ttyUSB0"  # USB serial port for Arduino communication
DEFAULT_BAUDRATE = 115200             # Baud rate for serial communication
DEFAULT_TIMEOUT = 1.0                 # Serial read/write timeout in seconds

# AI model file paths
AI_MODELS_DIR = "mpu/ai/models"                                        # Directory containing AI model files
BEST_MODEL_PATH = os.path.join(AI_MODELS_DIR, "best_model.pth")        # PyTorch helmet classifier model
ONNX_MODEL_PATH = os.path.join(AI_MODELS_DIR, "best_model.onnx")       # ONNX helmet classifier for inference
MOBILENET_SSD_PATH = os.path.join(AI_MODELS_DIR, "mobilenet_ssd.onnx")  # Person detection model

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

# Image processing configuration
MODEL_INPUT_SIZE = 224           # Input size for helmet classifier model (224x224)
DETECTOR_INPUT_SIZE = 300        # Input size for MobileNet SSD person detector (300x300)

# Detection configuration
PERSON_CLASS_ID = 15             # COCO dataset class ID for person (used by MobileNet SSD)
DETECTOR_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score for person detection

# Network configuration
HTTP_TIMEOUT = 10                # HTTP request timeout in seconds
RETRY_DELAY = 1                  # Delay between retry attempts in seconds

# Validate paths and connections on import
# These validation checks run automatically when the module is imported
_validate_dataset_paths()    # Check if training datasets are accessible
validate_model_files()       # Check if AI model files exist
validate_server_connection() # Test connection to alert server