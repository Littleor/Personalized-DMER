import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.layer.feature_fusion import FeatureFusionModel
from models.layer.multi_scale_attention import TransformerMultiScaleAttention
from models.layer.predicton import SequenceValuePredictionModel
from models.layer.spectrogram_feature_extractor import SpectrogramFeatureExtractor


class PDMERModel(nn.Module):
    def __init__(
        self,
        origin_feature_dim=1024,
        device="cpu",
        dropout_rate=0.2,
        num_attention_heads=4,
        num_hidden_layers=4,
        hidden_act="gelu",
        max_position_embeddings=512,
        position_embedding_type="absolute",
        local_context_length=3,
        global_context_length=30,
        using_local_attention=True,
        using_gloabl_attention=True,
        *args,
        **kwargs,
    ):
        super(PDMERModel, self).__init__()

        self.device = device
        self.dtype = torch.float32
        self.using_local_attention = using_local_attention
        self.using_gloabl_attention = using_gloabl_attention

        hidden_size = origin_feature_dim

        self.multi_scale_attention = TransformerMultiScaleAttention(
            origin_feature_dim=origin_feature_dim,
            num_attention_heads=num_attention_heads,
            hidden_act=hidden_act,
            dropout_rate=dropout_rate,
            max_position_embeddings=max_position_embeddings,
            position_embedding_type=position_embedding_type,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            local_context_length=local_context_length,
            global_context_length=global_context_length,
            device=device,
            using_local_attention=using_local_attention,
            using_gloabl_attention=using_gloabl_attention,
        )

        self.sequence_predict_model = SequenceValuePredictionModel(
            input_size=hidden_size,
            hidden_size=hidden_size // 4,
            dropout_rate=dropout_rate,
            output_size=2,
        )

        self.spectrogram_feature_extractor = SpectrogramFeatureExtractor()
        self.fusion_model: FeatureFusionModel = FeatureFusionModel(
            spectrogram_feature_dim=self.spectrogram_feature_extractor.feature_dim,
            imagebind_feature_dim=origin_feature_dim,
        )

    def forward(
        self,
        embedding: Dict[str, torch.Tensor],
        output_attentions: Optional[bool] = True,
    ) -> Tuple[torch.Tensor]:

        spectrogram_feature = self.spectrogram_feature_extractor(
            embedding["log_mel_spectrogram"].unsqueeze(2)
        )
        hidden_states = self.fusion_model(
            spectrogram_feature,
            embedding.get("gloabl_imagebind_audio_embedding"),
        )

        output_hidden_state, attention_maps = self.multi_scale_attention(
            hidden_states, output_attentions=output_attentions
        )

        value = self.sequence_predict_model(output_hidden_state)

        return_value = (value[:, :, 0], value[:, :, 1])

        return {
            "model_output": return_value,
            "attention_maps": attention_maps,
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PDMERModel(device=device).to(device)

    x = {
        "imagebind_audio_embedding": torch.randn(6, 60, 1024).to(device),
        "gloabl_imagebind_audio_embedding": torch.randn(6, 1, 1024).to(device),
        "log_mel_spectrogram": torch.randn(6, 60, 51, 128).to(device),
    }
    
    output = model(x)
    print(output["model_output"][0].shape, output["model_output"][1].shape)
