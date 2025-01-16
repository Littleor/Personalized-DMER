import glob
import logging
import os
import sys

import librosa
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils.file import check_file_exist, file_list_sort


def get_label_true_shape(x: torch.Tensor):
    true_shapes = []
    for row in x:
        true_shape = torch.sum(~torch.isnan(row)).item()
        true_shapes.append(true_shape)
    return true_shapes


class PMEmoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        duration: float = 10.0,
        sample_rate: int = 44100,
        using_cache_tensor: bool = True,
        device: str = "cpu",
        audio_embedding_dir_name: str = "audio_embedding",
        loading_data_before_use: bool = False,
        loading_cache_tensor_in_device: bool = False,
        is_train_data: bool = True,
        max_seq_length: int = 1225,
        using_personalized_data=False,  # Always False
    ):
        raise NotImplementedError("PMEmo dataset is not supported now.")
