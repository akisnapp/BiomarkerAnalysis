import torch
import torch.nn as nn
from transformers import ViTModel
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from train import train_model
from data_preprocessing import get_data_loaders
import torch.optim as optim
from train import train_model
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultimodalTransformer(nn.Module):
    def __init__(self, num_biomarkers, num_classes):
        super(MultimodalTransformer, self).__init__()
        # Vision Transformer Backbone
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # Transformer for Biomarkers
        self.biomarker_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=num_biomarkers, nhead=2, batch_first=True
            ),
            num_layers=2
        )
        
        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size + num_biomarkers, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, biomarkers):
        # ViT processing
        outputs = self.vit(images)
        image_features = outputs.pooler_output
        if image_features is None:
            image_features = outputs.last_hidden_state[:, 0, :]

        # Biomarker processing
        # biomarkers shape: (batch_size, num_biomarkers)
        # Add a sequence dimension for the transformer
        biomarker_features = biomarkers.unsqueeze(1)  # Shape: (batch_size, seq_len=1, num_biomarkers)
        biomarker_features = self.biomarker_transformer(biomarker_features)
        biomarker_features = biomarker_features.squeeze(1)  # Shape: (batch_size, num_biomarkers)

        # Concatenate features
        combined_features = torch.cat((image_features, biomarker_features), dim=1)
        output = self.fusion(combined_features)
        return output

# Define model
model = MultimodalTransformer(
    num_biomarkers=14,  # Number of biomarkers
    num_classes=1  # Regression task (output dimension)
).to(DEVICE)

# Define the criterion and optimizer
criterion = nn.MSELoss()  # Use MSELoss for regression tasks
optimizer = optim.Adam(model.parameters(), lr=1e-4)
train_loader, val_loader = get_data_loaders()
# Train the model
train_losses, val_losses = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=DEVICE,
    epochs=25,  # Specify the number of epochs
    save_path='multimodal_transformer.pth'  # Path to save the best model
)

import matplotlib.pyplot as plt
epochs = range(1, len(train_losses) + 1)

# Training loss + validation loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs[1:], train_losses[1:], label="Training Loss", color="blue")
plt.plot(epochs[1:], val_losses[1:], label="Validation Loss", color="red")
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.savefig('multimodal_loss_plot.png')



