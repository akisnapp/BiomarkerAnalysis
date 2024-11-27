import cv2  
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    """
    Train and validate the model for the specified number of epochs.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        device (torch.device): Device to run the training on (CPU or GPU).
        epochs (int): Number of epochs to train.

    Returns:
        None
    """
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    train_losses = []
    val_losses = []
    train_accuracies = [] #TODO
    val_accuracies = [] #TODO

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for images, biomarkers, labels in train_loader:
            # Move data to the correct device
            images = images.to(device)
            biomarkers = biomarkers.to(device)
            labels = labels.to(device).unsqueeze(1)  # Ensure labels are [batch_size, 1]

            # Forward pass
            outputs = model(images, biomarkers)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            train_loss += loss.item() * images.size(0)

        # Average loss for the epoch
        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, biomarkers, labels in val_loader:
                # Move data to the correct device
                images = images.to(device)
                biomarkers = biomarkers.to(device)
                labels = labels.to(device).unsqueeze(1)

                # Forward pass
                outputs = model(images, biomarkers)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)

        # Adjust learning rate
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses





