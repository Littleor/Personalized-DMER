import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(MultiScaleConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size=(2, 2), padding=(1, 1)
        )  # Ensure the size is divided evenly

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class SpectrogramFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels=1,
        seq_len=60,
        *args,
        **kwargs,
    ):
        super(SpectrogramFeatureExtractor, self).__init__()
        self.seq_len = seq_len

        self.feature_extractor = nn.Sequential(
            MultiScaleConvBlock(in_channels, 8, (3, 1), padding=(2, 1)),
            MultiScaleConvBlock(8, 16, (3, 3), padding=(1, 1)),
            # MultiScaleConvBlock(16, 32, (3, 3), padding=(1, 1)),
        )

        # Dimensionality reduction using a Convolutional layer
        self.conv1x1 = nn.Conv2d(16, 1, kernel_size=(1, 1))

        self.feature_dim = 476

    def forward(self, mel_spec_seq: torch.Tensor):
        """Forward pass of the DAMFF model

        Args:
            mel_spec_seq (torch.Tensor): [N, seq_len, C=1, height=time, width=freq_bin]

        Returns:
            _type_: _description_
        """
        batch_size, seq_len, C, H, W = mel_spec_seq.shape

        mel_spec_seq = mel_spec_seq.view(
            batch_size * seq_len, C, H, W
        )  # [batch_size * seq_len, C = 1, H = 45, W = 128]

        # 1. Extract features using TemporalFrequencyMultiScaleConvolution
        features = self.feature_extractor(
            mel_spec_seq
        )  # [batch_size * seq_len, C' = 64, H' = 13, W' = 18]

        # Apply 1x1 conv to each sequence point separately
        z = self.conv1x1(features)  # shape: (batch_size * seq_len, 1, H', W')

        # Reshape to get the sequence back in the first dimension
        z = z.view(batch_size, seq_len, -1)  # shape: (N, Seq_len, hidden_size)

        assert (
            z.shape[-1] == self.feature_dim
        ), f"Expected {self.feature_dim}, got {z.shape[-1]}"

        return z
