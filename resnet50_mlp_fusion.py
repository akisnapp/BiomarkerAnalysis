import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class SupCEResNet_Fusion(nn.Module):
    """Encoder + multimodal fusion classifier."""
    def __init__(self, name='resnet50', num_classes=2, num_features=0, in_channels=3):
        super(SupCEResNet_Fusion, self).__init__()
        # Load ResNet50 with configurable input channels
        if name == 'resnet50':
            self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            # Modify the first convolution layer to match the input channels
            self.encoder.conv1 = nn.Conv2d(
                in_channels,
                self.encoder.conv1.out_channels,
                kernel_size=self.encoder.conv1.kernel_size,
                stride=self.encoder.conv1.stride,
                padding=self.encoder.conv1.padding,
                bias=False
            )
            dim_in = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()  # Remove the original fully connected layer

        # Add a fully connected layer for multimodal fusion
        self.fc = nn.Linear(dim_in + num_features, num_classes)

    def forward(self, x, label_vector=None):
        """
        Forward pass for fusion model.
        Args:
            x (torch.Tensor): Image input.
            label_vector (torch.Tensor): Additional input features (e.g., biomarkers).
        Returns:
            torch.Tensor: Output logits.
        """
        encoded = self.encoder(x)  # Extract image features
        if label_vector is not None:
            # Normalize and concatenate label vector
            label_vector = (label_vector - label_vector.min()) / (label_vector.max() - label_vector.min() + 1e-8)
            fused = torch.cat((encoded, label_vector), dim=1)
        else:
            fused = encoded
        return self.fc(fused)
