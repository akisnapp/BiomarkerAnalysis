import torch
import torch.nn as nn
from transformers import ViTModel

class MultimodalTransformer(nn.Module):
    def __init__(self, num_biomarkers, num_classes):
        super(MultimodalTransformer, self).__init__()
        # Vision Transformer Backbone
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # Transformer for Biomarkers
        self.biomarker_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=num_biomarkers, nhead=4), num_layers=2
        )
        
        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(768 + num_biomarkers, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, biomarkers):
        image_features = self.vit(images).pooler_output
        biomarker_features = self.biomarker_transformer(biomarkers.unsqueeze(1)).squeeze(1)
        combined_features = torch.cat((image_features, biomarker_features), dim=1)
        output = self.fusion(combined_features)
        return output
