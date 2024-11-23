import torch
import torch.nn as nn
from transformers import ViTModel

class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_biomarkers, num_classes, seq_length):
        super(TimeSeriesTransformer, self).__init__()
        self.seq_length = seq_length
        
        # Vision Transformer for Images
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # Transformer for Temporal Biomarkers
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=num_biomarkers, nhead=4), num_layers=2
        )
        
        # Positional Embeddings
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_length, num_biomarkers))
        
        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(768 + num_biomarkers, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, biomarkers_sequence):
        # Images through ViT
        image_features = self.vit(images).pooler_output
        
        # Add positional embeddings to biomarker sequence
        biomarkers_sequence += self.position_embedding[:, :self.seq_length, :]
        
        # Temporal features through Transformer
        temporal_features = self.temporal_transformer(biomarkers_sequence)
        aggregated_temporal = temporal_features.mean(dim=1)
        
        # Combine image and temporal features
        combined_features = torch.cat((image_features, aggregated_temporal), dim=1)
        output = self.fusion(combined_features)
        return output
