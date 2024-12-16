import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class FeatureFusionModel(nn.Module):
    def __init__(self, spectrogram_feature_dim: int, imagebind_feature_dim: int):
        super(FeatureFusionModel, self).__init__()

        self.spectrogram_feature_project = nn.Linear(
            spectrogram_feature_dim, imagebind_feature_dim
        )
        self.fusion_project = nn.Linear(imagebind_feature_dim, imagebind_feature_dim)

    def forward(
        self,
        spectrogram_feature: torch.Tensor,
        global_imagebind_audio_embedding: torch.Tensor = None,
    ):
        if global_imagebind_audio_embedding is not None:
            global_imagebind_audio_embedding = global_imagebind_audio_embedding.expand(
                -1, spectrogram_feature.size(1), -1
            )
        spectrogram_feature = self.spectrogram_feature_project(spectrogram_feature)

        fusion_feature = torch.sigmoid(
            spectrogram_feature + global_imagebind_audio_embedding
        )
        fusion_feature = self.fusion_project(fusion_feature)
        return fusion_feature
