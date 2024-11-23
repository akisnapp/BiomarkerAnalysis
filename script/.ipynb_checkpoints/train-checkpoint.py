import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        device (torch.device): Device to run the training on (CPU or GPU).

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    criterion = nn.MSELoss()
    
    for images, biomarkers, labels in train_loader:
        # Move data to the correct device
        images, biomarkers, labels = images.to(device), biomarkers.to(device), labels.to(device)

        # Forward pass
        outputs = model(images, biomarkers)
        loss = criterion(outputs.squeeze(), labels)  

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Validate the model.

    Args:
        model (nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the validation on (CPU or GPU).

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, biomarkers, labels in val_loader:
            # Move data to the correct device
            images, biomarkers, labels = images.to(device), biomarkers.to(device), labels.to(device, dtype=torch.long)

            # Forward pass
            outputs = model(images, biomarkers)
            loss = criterion(outputs, labels)

            # Accumulate metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


if __name__ == "__main__":
    print("This script contains train and validation functions.")
