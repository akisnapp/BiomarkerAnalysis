import os
import numpy as np
import torch
from torch import nn, optim
from resnet50_mlp_fusion import SupCEResNet_Fusion  
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    mean_absolute_error, mean_squared_error,r2_score
)
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from torchvision.models import resnet50

# Training and Validation Functions
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, biomarkers, targets in train_loader:
        images = images.to(device)
        biomarkers = biomarkers.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images, biomarkers)

        # Compute loss
        loss = criterion(outputs.squeeze(), targets)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    avg_train_loss = running_loss / len(train_loader.dataset)
    return avg_train_loss

def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for images, biomarkers, targets in val_loader:
            images = images.to(device)
            biomarkers = biomarkers.to(device)
            targets = targets.to(device)

            outputs = model(images, biomarkers)

            # Compute loss
            loss = criterion(outputs.squeeze(), targets)

            val_loss += loss.item() * images.size(0)

            # Collect outputs and targets for metrics
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader.dataset)

    # Metrics
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs).squeeze()

    # For regression task
    mse = mean_squared_error(all_targets, all_outputs)
    mae = mean_absolute_error(all_targets, all_outputs)
    r2 = r2_score(all_targets, all_outputs)

    # If treating as classification (binary classification)
    preds = (all_outputs > 0.5).astype(int)
    targets_binary = (all_targets > 0.5).astype(int)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        accuracy = accuracy_score(targets_binary, preds)
        precision = precision_score(targets_binary, preds, average='macro', zero_division=0)
        recall = recall_score(targets_binary, preds, average='macro', zero_division=0)
        f1 = f1_score(targets_binary, preds, average='macro', zero_division=0)
        try:
            roc_auc = roc_auc_score(targets_binary, all_outputs, average='macro')
        except ValueError:
            roc_auc = float('nan')

    metrics = {
        'val_loss': avg_val_loss,
        'mse': mse,
        'mae': mae,
        'r2_score': r2,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

    return metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10, save_path='best_model.pth'):
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    train_losses = []
    val_losses = []
    all_metrics = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training phase
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validation phase
        metrics = validate_one_epoch(model, val_loader, criterion, device)
        val_loss = metrics['val_loss']
        val_losses.append(val_loss)
        all_metrics.append(metrics)

        # Learning rate adjustment
        scheduler.step(val_loss)

        # Save the model if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} - **Best Model Saved!**")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Print metrics
        print(f"MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, R2 Score: {metrics['r2_score']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1 Score: {metrics['f1_score']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

    return train_losses, val_losses, all_metrics
