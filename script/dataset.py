import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.PDMER import PDMERModel
from utils.args import parse_args
from utils.DEAM.dataset import DEAMDataset, get_label_true_shape
from utils.logger import setup_logging
from utils.music.util import generate_split_duration_list, get_audio_log_mel_spec
from utils.PMEmo.dataset import PMEmoDataset

args = parse_args(
    description="Extract audio embedding from the DEAM dataset",
    arguments=[
        {
            "args": ["--cache_output_name"],
            "kwargs": {
                "type": str,
                "default": "audio_embedding_test",
                "help": "The root directory of the dataset",
            },
        }
    ],
)
setup_logging()

train_dataset = (PMEmoDataset if args.dataset_name == "PMEmo" else DEAMDataset)(
    root=args.dataset_root, using_cache_tensor=False, is_train_data=True
)
test_dataset = (PMEmoDataset if args.dataset_name == "PMEmo" else DEAMDataset)(
    root=args.dataset_root, using_cache_tensor=False, is_train_data=False
)
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size)
test_data_loader = DataLoader(test_dataset, batch_size=1)

model: PDMERModel = PDMERModel(
    device=args.device,
    query_embed_dim=args.model_query_embed_dim,
    num_attention_heads=args.model_num_attention_heads,
    num_hidden_layers=args.model_num_hidden_layers,
    segmentation_duration=args.model_segmentation_duration,
    feature_num_per_audio=args.model_feature_num_per_audio,
    train_audio_duration=args.model_train_audio_duration,
    dropout_rate=args.dropout_rate,
).to(args.device)

os.makedirs(
    os.path.join(args.dataset_root, "DEAM_audio", args.cache_output_name), exist_ok=True
)

duration = 30.0


def process_batch(batch, is_train_data: bool):
    samples, labels = batch
    duration_list = [
        generate_split_duration_list(
            sample,
            sample_rate=44100,
            length_each_clip=0.5,
            begin_time=15.0,
            end_time=(
                15.0 + args.model_feature_num_per_audio // 2
                if is_train_data and args.dataset_name != "PMEmo"
                else None
            ),
        )
        for sample in samples
    ]

    image_bind_duration_list = [
        generate_split_duration_list(
            sample,
            sample_rate=44100,
            length_each_clip=args.model_segmentation_duration,
            begin_time=15.0,
            end_time=(
                15.0 + args.model_feature_num_per_audio // 2
                if is_train_data and args.dataset_name != "PMEmo"
                else None
            ),
            slide_start=0,
        )
        for sample in samples
    ]

    for i in range(len(duration_list)):
        assert len(duration_list[i]) == len(
            image_bind_duration_list[i]
        ), f"{samples[i]}, len(duration_list[i]): {len(duration_list[i])}, len(image_bind_duration_list[i]): {len(image_bind_duration_list[i])}"

    a_labels, v_labels = get_label_true_shape(labels[0]), get_label_true_shape(
        labels[1]
    )
    for i in range(len(a_labels)):
        assert v_labels[i] <= len(duration_list[i]) and a_labels[i] <= len(
            duration_list[i]
        ), f"{samples[i]}, a_labels[i]: {a_labels[i]}, v_labels[i]: {v_labels[i]}, len(duration_list[i]): {len(duration_list[i])}"

        if len(duration_list[i]) > a_labels[i] or len(duration_list[i]) > v_labels[i]:
            duration_list[i] = duration_list[i][: min(a_labels[i], v_labels[i])]

        assert min(a_labels[i], v_labels[i]) == len(
            duration_list[i]
        ), f"{samples[i]}, a_labels[i]: {a_labels[i]}, v_labels[i]: {v_labels[i]}, len(duration_list[i]): {len(duration_list[i])}"

    # Get the embdding of the ImageBind, NOTE: this will remove the first 15s
    imagebind_audio_embedding = model.get_embedding(
        samples,
        audio_segmentation_list=image_bind_duration_list,
        clip_audio=(
            (15, 15 + args.model_feature_num_per_audio // 2)
            if is_train_data and args.dataset_name != "PMEmo"
            else (15, None)
        ),
    )

    gloabl_imagebind_audio_embedding = model.get_embedding(
        samples,
        audio_segmentation_list=(
            [
                [(15, 15 + args.model_feature_num_per_audio // 2)]
                for _ in range(len(samples))
            ]
            if is_train_data
            else [
                generate_split_duration_list(
                    sample,
                    sample_rate=44100,
                    length_each_clip=30,
                    begin_time=15.0,
                    end_time=None,
                    slide_start=0,
                )
                for sample in samples
            ]
        ),
        clip_audio=(
            (15, 15 + args.model_feature_num_per_audio // 2)
            if is_train_data
            else (15, None)
        ),
    )

    # Get the log mel spectrogram of the audio
    log_mel_spectrogram = get_audio_log_mel_spec(
        samples,
        frame_length=60,  # milliseconds
        frame_shift=10,  # milliseconds
        n_mels=128,
        duration=duration_list,
    )

    for i in range(len(samples)):
        torch.save(
            {
                "imagebind_audio_embedding": imagebind_audio_embedding[
                    i
                ],  # NOTE: This isn't used in the model
                # "fbank_feature": fbank_feature[i],
                "log_mel_spectrogram": log_mel_spectrogram[i],
                "gloabl_imagebind_audio_embedding": gloabl_imagebind_audio_embedding[i],
            },
            samples[i]
            .replace("wav_audio", args.cache_output_name)
            .replace("wav", "pt"),
        )


for batch in tqdm(train_data_loader, desc="Extracting train dataset's audio embedding"):
    process_batch(batch, is_train_data=True)

for batch in tqdm(test_data_loader, desc="Extracting test dataset's audio embedding"):
    process_batch(batch, is_train_data=False)
