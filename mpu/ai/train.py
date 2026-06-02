"""
Helmet Detection Model Training Script

This script trains an EfficientNet-B0 based helmet detection model using
transfer learning on combined SHEL5K and SHWD datasets. The model is
fine-tuned from ImageNet pre-trained weights for binary helmet classification.

Key Features:
- Transfer learning with EfficientNet-B0 backbone
- Mixed dataset training (SHEL5K + SHWD)
- Early stopping with validation monitoring
- Model checkpointing for best validation loss
- Configurable hyperparameters via command line

Training Process:
1. Load and preprocess combined datasets
2. Initialize EfficientNet-B0 with custom classifier head
3. Train with Adam optimizer and CrossEntropy loss
4. Monitor validation metrics for early stopping
5. Save best model checkpoint

Usage:
    python train.py --batch-size 32 --epochs 30 --lr 0.001
    python train.py --shel5k-path /path/to/shel5k --shwd-path /path/to/shwd
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from mpu.ai.dataset.loader import create_data_loaders

from mpu.config import SHEL5K_PATH, SHWD_PATH


def create_model(num_classes=2):
    """
    Create EfficientNet-B0 model with custom classifier for helmet detection.

    Uses transfer learning from ImageNet pre-trained weights and replaces
    the classifier head for binary helmet classification.

    Args:
        num_classes: Number of output classes (default: 2 for helmet/no_helmet)

    Returns:
        PyTorch model ready for training
    """
    # Load EfficientNet-B0 with ImageNet pre-trained weights
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace classifier head for helmet detection
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),                    # Dropout for regularization
        nn.Linear(in_features, num_classes) # Binary classification layer
    )

    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch and return average loss.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        criterion: Loss function (e.g., CrossEntropyLoss)
        optimizer: Optimizer (e.g., Adam)
        device: Device to run training on (CPU/GPU/MPS)

    Returns:
        Average training loss for the epoch
    """
    model.train()  # Set model to training mode
    running_loss = 0.0

    # Iterate through training batches
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to device
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()     # Clear gradients
        outputs = model(images)   # Model prediction
        loss = criterion(outputs, labels)  # Calculate loss

        # Backward pass
        loss.backward()           # Compute gradients
        optimizer.step()          # Update weights

        running_loss += loss.item()

    return running_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """
    Validate model and return average loss and accuracy.

    Args:
        model: PyTorch model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on

    Returns:
        Tuple of (average_loss, accuracy_percentage)
    """
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient computation for efficiency
    with torch.no_grad():
        for images, labels in val_loader:
            # Move data to device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate averages
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def main():
    """
    Main training function with early stopping and model saving.

    Handles command line argument parsing, device setup, data loading,
    model training with validation monitoring, and checkpoint saving.
    """
    parser = argparse.ArgumentParser(description='Train helmet detection model')
    parser.add_argument('--shel5k-path', type=str,
                       default=SHEL5K_PATH,
                       help='Path to SHEL5K dataset directory')
    parser.add_argument('--shwd-path', type=str,
                       default=SHWD_PATH,
                       help='Path to SHWD dataset directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = create_data_loaders(
        shel5k_path=args.shel5k_path,
        shwd_path=args.shwd_path,
        batch_size=args.batch_size,
        num_workers=0
    )

    model = create_model(num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    epochs = args.epochs
    # Early stopping configuration
    best_val_loss = float('inf')
    early_stop_patience = 5
    early_stop_counter = 0

    # Create models directory
    os.makedirs('mpu/ai/models', exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Save best model and reset early stopping counter
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            try:
                torch.save(model.state_dict(), 'mpu/ai/models/best_model.pth')
            except Exception as e:
                print(f"Error saving model: {e}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Check early stopping condition
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model for final evaluation
    try:
        model.load_state_dict(torch.load('mpu/ai/models/best_model.pth', map_location=device))
    except FileNotFoundError:
        print("Warning: best_model.pth not found")
        return
    final_val_loss, final_accuracy = validate(model, val_loader, criterion, device)
    print(f"Final validation accuracy: {final_accuracy:.2f}%")


if __name__ == "__main__":
    main()
