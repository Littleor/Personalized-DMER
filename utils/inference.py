import os
import sys
from collections import defaultdict
from typing import Dict, Union

import learn2learn as l2l
import torch
import torch.nn as nn

from utils.music.util import generate_split_duration_list, get_audio_log_mel_spec

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.image_bind import ImageBind


def using_image_bind_embedding(
    model: nn.Module,
) -> bool:
    from models.PDMER import PDMERModel

    m = model.module if isinstance(model, l2l.algorithms.MAML) else model
    return isinstance(m, PDMERModel)


def check_is_model_instance(instance: nn.Module, Model):
    m = instance.module if isinstance(instance, l2l.algorithms.MAML) else instance
    return isinstance(m, Model)


def model_wrap(
    model: nn.Module,
    embedding_dict: Dict[str, torch.Tensor] = None,
    **kwargs,
):
    def model_input_wrap() -> Union[torch.Tensor, dict]:
        return embedding_dict

    return_value = model(model_input_wrap(), **kwargs)
    if (
        isinstance(return_value, tuple)
        or isinstance(return_value, list)
        or isinstance(return_value, torch.Tensor)
    ):
        return defaultdict(
            lambda: None,
            {
                "model_output": return_value,
            },
        )

    elif isinstance(return_value, dict):
        return return_value
    else:
        raise ValueError(
            f"Return value should be tuple, list, torch.Tensor or dict, but got {type(return_value)}"
        )


def slide_inference(
    model: nn.Module,
    embedding_dict: Dict[str, torch.Tensor],
):
    hop_size = (2.5, 0.5)  # A, V hop size
    hop_size = (int(hop_size[0] / 0.5), int(hop_size[1] / 0.5))
    attention_maps = None

    # Remove the value is None in embedding_dict
    embedding_dict = {k: v for k, v in embedding_dict.items() if v is not None}

    assert (
        len(embedding_dict) > 0
    ), "All the embedding is None, please check the input embedding"

    # NOTE: this only support batch_size = 1, because all the test data's length is not same
    min_seq_len = min(tensor.shape[1] for tensor in embedding_dict.values())
    embedding_dict = {
        key: tensor[:, :min_seq_len, :] for key, tensor in embedding_dict.items()
    }
    # Now all the sequence length is the min_seq_len
    seq_len = min_seq_len

    slide_count = seq_len - 60 + 1

    all_embedding_dict = defaultdict(list, [])

    for i in range(0, slide_count):
        for key, tensor in embedding_dict.items():
            all_embedding_dict[key].append(tensor[:, i : i + 60])

    for key, tensor in all_embedding_dict.items():
        all_embedding_dict[key] = torch.cat(tensor, dim=0)

    all_embedding_batch_size = next(iter(all_embedding_dict.values())).shape[0]

    # Batch input
    batch_size = 32
    a_outputs = []
    v_outputs = []

    for i in range(0, all_embedding_batch_size, batch_size):
        currrent_output = model_wrap(
            model,
            embedding_dict={
                key: tensor[i : i + batch_size]
                for key, tensor in all_embedding_dict.items()
            },
        )
        a_output, v_output = currrent_output["model_output"]
        a_outputs.append(a_output)
        v_outputs.append(v_output)

        if i == 0 and currrent_output["attention_maps"] is not None:
            # The third output is the attention map
            attention_maps = currrent_output["attention_maps"][0].unsqueeze(
                0
            )  # NOTE: this is batch_size = 1, and we just want a single attention map

    window_outputs = [
        torch.cat(a_outputs, dim=0),
        torch.cat(v_outputs, dim=0),
    ]  # ([all_batch_size, seq_len], [all_batch_size, seq_len])

    assert (
        window_outputs[0].shape[0] == window_outputs[1].shape[0]
    ), f"Output shape is not same, a_output.shape: {window_outputs[0].shape}, v_output.shape: {window_outputs[1].shape}"

    assert (
        window_outputs[0].shape[0] == all_embedding_batch_size == slide_count
    ), f"Output shape is not same, a_output.shape: {window_outputs[0].shape}, all_embedding_batch_size: {all_embedding_batch_size}, slide_count: {slide_count}"

    outputs = [None, None]
    for j, out in enumerate(window_outputs):
        if hop_size[j] == 1:
            outputs[j] = torch.cat(
                [out[0, :]] + [out[i, -1].unsqueeze(0) for i in range(1, out.shape[0])],
                dim=0,
            )
        else:
            outputs[j] = torch.cat(
                [out[0, :]]
                + [
                    out[i, -1 * hop_size[j] :]
                    for i in range(hop_size[j] + 1, out.shape[0], hop_size[j])
                ],
                dim=0,
            )
            if outputs[j].shape[0] != out.shape[0] + 60 - 1:
                last_seq = out[-1, -1 * ((out.shape[0] - 1) % hop_size[j]) :]
                if last_seq.shape[0] > hop_size[j]:
                    last_seq = last_seq[-1 * hop_size[j] :]
                outputs[j] = torch.cat(
                    [outputs[j], last_seq],
                    dim=0,
                )
        outputs[j] = outputs[j].unsqueeze(0)

    assert (
        outputs[0].shape[1] == outputs[1].shape[1] == seq_len
    ), f"Seq length is not same, a_output.shape[1]: {outputs[0].shape[1]}, v_output.shape[1]: {outputs[1].shape[1]}, seq_len: {seq_len}"

    seq_len = outputs[j].shape[1]

    return {
        "model_output": outputs,
        "attention_maps": attention_maps,
    }


def get_feature_from_file(
    audio_file_path: Union[str, list],
    imagebind_model: ImageBind = None,
    device="cuda:2",
) -> dict:
    if imagebind_model is None:
        # 1. Load the audio embedding model
        imagebind_model: ImageBind = ImageBind(device=device).to(device=device).eval()

    if not isinstance(audio_file_path, list):
        audio_file_path = [audio_file_path]

    duration_list = [
        generate_split_duration_list(
            sample,
            sample_rate=44100,
            length_each_clip=0.5,
            begin_time=0,
            end_time=None,
        )
        for sample in audio_file_path
    ]

    # 2. Get the audio embedding
    imagebind_embedding: torch.Tensor = imagebind_model.get_embedding_wrap(
        audio_file_path,
    )

    # 3. Get the audio global imagebind audio embedding
    global_imagebind_embedding: torch.Tensor = imagebind_model.get_embedding(
        audio_file_path, audio_segmentation_list=[[(0, 30)]] * len(audio_file_path)
    )

    # Get the embdding of the ImageBind
    log_mel_spec = get_audio_log_mel_spec(
        audio_file_path,
        frame_length=60,  # milliseconds
        frame_shift=10,  # milliseconds
        n_mels=128,
        duration=duration_list,
    ).to(device)

    return {
        "imagebind_audio_embedding": imagebind_embedding,
        "global_imagebind_audio_embedding": global_imagebind_embedding,
        "log_mel_spectrogram": log_mel_spec,
    }


def build_batch(
    audio_file_path: Union[str, list],
    imagebind_model: ImageBind = None,
    device="cuda:2",
    labels_list: list = None,
):
    feature = get_feature_from_file(
        audio_file_path, imagebind_model=imagebind_model, device=device
    )
    batch_size, seq_len = next(iter(feature.values())).shape[:2]

    labels = None
    if labels_list is not None:
        labels_list_tenor = torch.tensor(
            labels_list, dtype=torch.float32, device=device
        )
        labels = labels_list_tenor[:, 0, :], labels_list_tenor[:, 1, :]
        assert (
            labels[0].shape[1] == labels[1].shape[1] == seq_len,
            f"Seq length is not same, sample length: {seq_len}, label length: {labels[0].shape[1]}",
        )
    batch = (
        feature,
        (
            (
                torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device),
                torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device),
            )
            if labels is None
            else labels
        ),
    )
    return batch
