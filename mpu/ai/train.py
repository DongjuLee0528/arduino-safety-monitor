import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from mpu.ai.dataset.loader import create_data_loaders

SHEL5K_PATH = "~/Documents/AIdatasets/ helmet-safety-robot/raw/9rcv8mm682-4/Safety Helmet Wearing Dataset"
SHWD_PATH = "~/Documents/AIdatasets/ helmet-safety-robot/raw/VOC2028"


def create_model(num_classes=2):
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )

    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def main():
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
    best_val_loss = float('inf')
    early_stop_patience = 5
    early_stop_counter = 0

    os.makedirs('mpu/ai/models', exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            try:
                torch.save(model.state_dict(), 'mpu/ai/models/best_model.pth')
            except Exception as e:
                print(f"Error saving model: {e}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    try:
        model.load_state_dict(torch.load('mpu/ai/models/best_model.pth', map_location=device))
    except FileNotFoundError:
        print("Warning: best_model.pth not found")
        return
    final_val_loss, final_accuracy = validate(model, val_loader, criterion, device)
    print(f"Final validation accuracy: {final_accuracy:.2f}%")


if __name__ == "__main__":
    main()
