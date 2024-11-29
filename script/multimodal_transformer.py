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


