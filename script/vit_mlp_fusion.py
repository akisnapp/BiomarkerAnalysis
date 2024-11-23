import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class ViTMLPFusion(nn.Module):
    def __init__(self, num_biomarkers, num_classes):
        super(ViTMLPFusion, self).__init__()
        # Vision Transformer Backbone
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # MLP for Biomarker Features
        self.mlp = nn.Sequential(
            nn.Linear(num_biomarkers, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(768 + 128, 512),  # 768 is the output dimension of ViT-B/16
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, biomarkers):
        image_features = self.vit(images).pooler_output
        biomarker_features = self.mlp(biomarkers)
        combined_features = torch.cat((image_features, biomarker_features), dim=1)
        output = self.fusion(combined_features)
        return output
