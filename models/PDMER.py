import logging
import math
import os
import sys
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.ImageBind.imagebind.data import load_and_transform_audio_data
from models.ImageBind.imagebind.models import imagebind_model
from models.ImageBind.imagebind.models.imagebind_model import ModalityType
from models.layer.feature_fusion import FeatureFusionModel
from models.layer.multi_scale_attention import TransformerMultiScaleAttention
from models.layer.predicton import SequenceValuePredictionModel
from models.layer.spectrogram_feature_extractor import SpectrogramFeatureExtractor
from utils.music.util import music_segmentation


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

        self.imagebind_model = None

    def get_embedding(
        self,
        audio_paths: Union[str, list],
        audio_segmentation_list: list = None,
        slide_length: float = 0.5,
        clip_audio: tuple = (15, 45),
    ) -> torch.Tensor:
        if self.imagebind_model is None:
            logging.info("Loading ImageBind model...")
            self.imagebind_model = imagebind_model.imagebind_huge(pretrained=True).to(
                self.device
            )
            self.imagebind_model.eval()
            for param in self.imagebind_model.parameters():
                param.requires_grad = False

        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]

        feature_num_per_audio = (
            self.feature_num_per_audio
            if audio_segmentation_list is None
            else len(audio_segmentation_list[0])
        )

        waveforms = music_segmentation(
            audio_paths=audio_paths,
            sample_rate=44100,
            clip_audio=clip_audio,
            audio_segmentation_list=(
                [
                    [
                        (
                            clip_audio[0]
                            + -1 * self.segmentation_duration / 2
                            + slide_length * i,
                            clip_audio[0]
                            + self.segmentation_duration / 2
                            + slide_length * i,
                        )
                        for i in range(self.feature_num_per_audio)
                    ]
                    for _ in range(len(audio_paths))
                ]
                if audio_segmentation_list is None
                else audio_segmentation_list
            ),
        ).to(
            self.device
        )  # [seq_len, dim]

        # Batch input to fix OOM
        outputs = []
        batch_count = math.ceil(waveforms.shape[0] / 60)
        for i in range(0, batch_count):
            current_waveforms = waveforms[
                i * 60 : ((i + 1) * 60) if i != batch_count - 1 else None
            ]

            inputs = {
                ModalityType.AUDIO: load_and_transform_audio_data(
                    audio_paths=None,
                    device=self.device,
                    sample_rate=44100,
                    clip_duration=2,
                    clips_per_video=3,
                    waveforms=current_waveforms,
                )
            }
            with torch.no_grad():
                embeds = self.imagebind_model(inputs)

            outputs.append(embeds[ModalityType.AUDIO])

        audio_embeds = torch.cat(outputs, dim=0)
        return audio_embeds.view(-1, feature_num_per_audio, audio_embeds.shape[-1])

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
            embedding.get("global_imagebind_audio_embedding"),
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
        "global_imagebind_audio_embedding": torch.randn(6, 1, 1024).to(device),
        "log_mel_spectrogram": torch.randn(6, 60, 51, 128).to(device),
    }

    output = model(x)
    print(output["model_output"][0].shape, output["model_output"][1].shape)

    # DEMO for batch input

    from utils.inference import build_batch

    audio_file_path_list = [
        "/data2/datasets/DEAM/DEAM_audio/wav_audio/2.wav",
        "/data2/datasets/DEAM/DEAM_audio/wav_audio/3.wav",
        "/data2/datasets/DEAM/DEAM_audio/wav_audio/4.wav",
    ]

    embedding, _ = build_batch(
        audio_file_path_list,
        imagebind_model=None,
        device=device,
    )
    print("\n Build batch embedding:")
    for key, value in embedding.items():
        print("\t", key, value.shape)

    print("Result:")
    output = model(embedding)
    print(
        "Arousal:",
        output["model_output"][0].shape,
        "Valence:",
        output["model_output"][1].shape,
    )
