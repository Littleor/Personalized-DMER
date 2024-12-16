import logging
import math
import os
import sys
from typing import Union

import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.ImageBind.imagebind.data import load_and_transform_audio_data
from models.ImageBind.imagebind.models import imagebind_model
from models.ImageBind.imagebind.models.imagebind_model import ModalityType
from utils.music.util import generate_split_duration_list, music_segmentation


class ImageBind(nn.Module):

    def __init__(self, device="cuda:4"):
        super(ImageBind, self).__init__()
        self.device = device

        logging.info("Loading ImageBind model...")
        self.imagebind_model = imagebind_model.imagebind_huge(pretrained=True).to(
            self.device
        )
        self.imagebind_model.eval()
        for param in self.imagebind_model.parameters():
            param.requires_grad = False

    def get_embedding(
        self,
        audio_paths: Union[str, list],
        audio_segmentation_list: list = None,
        slide_length: float = 0.5,
        clip_audio: tuple = (15, 45),
    ) -> torch.Tensor:
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

    def get_embedding_wrap(self, audio_paths: Union[str, list]) -> torch.Tensor:
        audio_paths = [audio_paths] if isinstance(audio_paths, str) else audio_paths
        image_bind_duration_list = [
            generate_split_duration_list(
                sample,
                sample_rate=44100,
                length_each_clip=5,
                begin_time=0,
                end_time=None,
                slide_start=0,
            )
            for sample in audio_paths
        ]
        audio_embeds = self.get_embedding(
            audio_paths,
            audio_segmentation_list=image_bind_duration_list,
            clip_audio=None,
        )
        return audio_embeds


if __name__ == "__main__":
    model = ImageBind()
    samples = [
        "/data/dataset/DEAM/DEAM_audio/wav_audio/2.wav",
        "/data/dataset/DEAM/DEAM_audio/wav_audio/3.wav",
        "/data/dataset/DEAM/DEAM_audio/wav_audio/4.wav",
    ]
    audio_embeds = model.get_embedding_wrap(
        samples,
    )
    print(audio_embeds.shape)
    samples = ("/data/dataset/DEAM/DEAM_audio/wav_audio/2.wav",)
    audio_embeds = model.get_embedding_wrap(
        samples,
    )
    print(audio_embeds.shape)
