import os
import sys
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


class SequenceValuePredictionModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 2,
        dropout_rate: float = 0.5,
    ):
        super(SequenceValuePredictionModel, self).__init__()

        self.bilstm = nn.LSTM(
            input_size, input_size // 4, batch_first=True, bidirectional=True
        )
        self.layers = nn.Sequential(
            nn.Linear(input_size // 2, hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x, _ = self.bilstm(x)  # shape: (N, Seq_len, hidden_size)
        x = self.layers(x)
        return x
