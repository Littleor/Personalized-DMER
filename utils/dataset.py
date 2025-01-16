import os
import sys
from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.DEAM.dataset import DEAMDataset
from utils.logger import setup_logging


def reshape_audio_tensor(
    audio_tensor: torch.Tensor,
    sr: int = 44100,
    length_each_clip: float = 0.5,
):
    """
    Reshape the audio tensor to the desired shape
    Args:
        audio_tensor (torch.Tensor): The input audio tensor
        sr (int, optional): The sample rate of the audio. Defaults to 44100.
        length_each_clip (float, optional): The length of each clip. Defaults to 0.5.
    """
    block_size = int(sr * length_each_clip)

    total_length = audio_tensor.size(1)
    seq_len = (
        total_length + block_size - 1
    ) // block_size  # Calculate the number of blocks needed

    required_length = seq_len * block_size
    padding_length = required_length - total_length

    # Padding the audio tensor with zeros
    if padding_length > 0:
        padding = torch.zeros(
            (audio_tensor.size(0), padding_length), device=audio_tensor.device
        )
        audio_tensor_padded = torch.cat((audio_tensor, padding), dim=1)
    else:
        audio_tensor_padded = audio_tensor

    audio_tensor_reshaped = audio_tensor_padded.unfold(1, block_size, block_size)

    return audio_tensor_reshaped.squeeze(0)


def split_batch(batch, split_size_list: list):
    split_batches = []

    def splite_item(item):
        splited_item = []
        begin_index = 0

        if isinstance(item, torch.Tensor) or isinstance(item, list):
            for size in split_size_list:
                splited_item.append(item[begin_index : begin_index + size])
                begin_index += size
            return splited_item
        else:
            raise ValueError("The item is not a tensor or list")

    data, labels = batch

    split_data = {}
    split_label = [splite_item(label) for label in labels]
    for key in data.keys():
        split_data[key] = splite_item(data[key])

    for i in range(len(split_size_list)):
        split_batches.append(
            [
                {key: split_data[key][i] for key in split_data.keys()},
                [label[i] for label in split_label],
            ]
        )
    return split_batches


def segment_batch(batch, segment_size: int):
    # TODO support batch, Now only support batch_size = 1
    split_batches = []
    segment_count = -1

    def segment_item(item):
        splited_item = []
        if isinstance(item, torch.Tensor):
            if len(item.size()) == 1:
                item = item.unsqueeze(0)

            splited_item = item.split(segment_size, dim=1)
            if splited_item[-1].shape[1] < segment_size:
                splited_item = splited_item[:-1]
        elif isinstance(item, list):
            # Split the list
            splited_item = item
            # for i in range(0, len(item), segment_size):
            # splited_item.append(item[i : i + segment_size])
        else:
            raise ValueError("The item is not a tensor or list")
        return splited_item

    data, labels = batch
    # Remove the NaN value of tensor
    labels = [label[~torch.isnan(label)] for label in labels]

    split_data = {}
    split_label = [segment_item(label) for label in labels]
    for key in data.keys():
        split_data[key] = segment_item(data[key])
        segment_count = max(segment_count, len(split_data[key]))

    for i in range(len(split_label[0])):
        split_batches.append(
            [
                {
                    key: (
                        [split_data[key][0]]
                        if isinstance(split_data[key], list)
                        else split_data[key][i]
                    )
                    for key in split_data.keys()
                },
                [label[i] for label in split_label],
            ]
        )
    return split_batches


def merge_into_batch(batches: list):
    merged_batch = {}
    merged_labels = []
    for key in batches[0][0].keys():
        if isinstance(batches[0][0][key], list):
            merged_batch[key] = [batch[0][key] for batch in batches]
            merged_batch[key] = [
                item for sublist in merged_batch[key] for item in sublist
            ]
        else:
            merged_batch[key] = torch.cat([batch[0][key] for batch in batches], dim=0)

    for i in range(len(batches[0][1])):
        merged_labels.append(torch.cat([batch[1][i] for batch in batches], dim=0))
    return merged_batch, merged_labels


def split_batch_seq(
    batch,
    seq_len_list: list,
) -> list:
    """Split the batch into multiple batches with different sequence lengths

    Args:
        batch (_type_): The input batch
        seq_len_list (list): The list of sequence lengths

    Returns:
        list: _description_
    """
    # Extract the input batch
    input_batch, labels = batch

    # Calculate the total sequence length
    total_seq_len = min(
        [t.shape[1] for t in input_batch.values() if isinstance(t, torch.Tensor)]
    )
    # next(iter(input_batch.values())).shape[1]

    # Check if -1 is in the seq_len_list
    if -1 in seq_len_list:
        # Calculate the sum of all specified lengths
        specified_len_sum = sum([seq_len for seq_len in seq_len_list if seq_len != -1])
        # Calculate the remaining length
        remaining_length = total_seq_len - specified_len_sum

        # Replace -1 with the remaining length
        seq_len_list = [
            seq_len if seq_len != -1 else remaining_length for seq_len in seq_len_list
        ]

    # Initialize the list to hold the split batches
    split_batches = []

    # Initialize the starting index
    start_idx = 0

    # Iterate through the sequence length list
    for seq_len in seq_len_list:
        end_idx = start_idx + seq_len

        # Split the input batch
        split_input_batch = {
            key: (
                value[:, start_idx:end_idx]
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in input_batch.items()
        }

        # Split the labels
        split_labels = [label[:, start_idx:end_idx] for label in labels]

        # Append the split batch to the list
        split_batches.append((split_input_batch, split_labels))

        # Move to the next start index
        start_idx = end_idx

    return split_batches


def merge_into_seq(batches: list):
    merged_batch = {}
    merged_labels = []
    for key in batches[0][0].keys():
        if isinstance(batches[0][0][key], list):
            merged_batch[key] = [batch[0][key] for batch in batches]
            merged_batch[key] = [
                item for sublist in merged_batch[key] for item in sublist
            ]
        else:
            merged_batch[key] = torch.cat([batch[0][key] for batch in batches], dim=1)

    for i in range(len(batches[0][1])):
        merged_labels.append(torch.cat([batch[1][i] for batch in batches], dim=1))
    return merged_batch, merged_labels


def sample_from_dataset(dataset: Dataset, num: int):
    if num > len(dataset):
        raise ValueError("num must be less than or equal to the length of the dataset")

    indices = np.random.choice(len(dataset), num, replace=False)

    sampler = SubsetRandomSampler(indices)

    dataloader = DataLoader(dataset, batch_size=num, sampler=sampler)

    for batch_data, batch_labels in dataloader:
        return batch_data, batch_labels


def generate_personalized_task_from_dataset(
    dataset: Dataset, num_task: int, batch_size: int, num_shot: int = None
):
    if num_shot is None:
        num_shot = batch_size
    total_count = (num_shot + batch_size) * num_task
    if total_count > len(dataset):
        raise ValueError("num must be less than or equal to the length of the dataset")

    music_worker_id_dict: Dict[str, list] = dataset.get_music_worker_id_dict()

    worker_id_list = np.random.choice(
        list(music_worker_id_dict.keys()), num_task, replace=False
    )

    for i, worker_id in enumerate(worker_id_list):
        music_id_list = music_worker_id_dict[worker_id]
        support_music_id = np.random.choice(
            music_id_list, num_shot, replace=False
        )  # NOTE: DO NOT place the support set sample in query set

        query_music_id_list = np.random.choice(
            [id for id in music_id_list if id not in support_music_id],
            batch_size,
            replace=(num_shot + batch_size)
            > len(
                music_id_list
            ),  # If dataset length is less than num_shot + batch_size, then replace
        )
        data_indices = dataset.get_indices_from_music_id(
            [*support_music_id, *query_music_id_list], worker_id=worker_id
        )
        support_indices, query_indices = (
            data_indices[:num_shot],
            data_indices[num_shot:],
        )
        support_sampler, query_sampler = (
            SubsetRandomSampler(support_indices),
            SubsetRandomSampler(query_indices),
        )
        support_dataloader = DataLoader(
            dataset, batch_size=num_shot, sampler=support_sampler
        )
        query_dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=query_sampler
        )
        # Just return the first task
        yield next(iter(zip(support_dataloader, query_dataloader)))


def generate_task_from_dataset(
    dataset: Dataset, num_task: int, batch_size: int, num_shot: int = None
):
    if num_shot is None:
        num_shot = batch_size

    total_count = (num_shot + batch_size) * num_task
    if total_count > len(dataset):
        raise ValueError("num must be less than or equal to the length of the dataset")

    indices = np.random.choice(len(dataset), len(dataset), replace=False)

    support_indices, query_indices = (
        indices[: num_shot * num_task],
        indices[num_shot * num_task :],
    )

    support_sampler, query_sampler = (
        SubsetRandomSampler(support_indices),
        SubsetRandomSampler(query_indices),
    )

    support_dataloader = DataLoader(
        dataset, batch_size=num_shot, sampler=support_sampler
    )
    query_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=query_sampler)

    for i, (support_batch, query_batch) in enumerate(
        zip(support_dataloader, query_dataloader)
    ):
        yield support_batch, query_batch


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    setup_logging()

    dataset = DEAMDataset(
        root=os.getenv("DATASET_PATH"),
        device="cuda:4",
        audio_embedding_dir_name=os.getenv("AUDIO_EMBEDDING_DIR_NAME"),
        is_train_data=True,
        using_personalized_data=True,
    )
    tasks = generate_personalized_task_from_dataset(
        dataset, num_task=16, batch_size=5, num_shot=1
    )
    for task in tasks:
        break

    test_dataset = DEAMDataset(
        root=os.getenv("DATASET_PATH"),
        device="cuda:4",
        audio_embedding_dir_name=os.getenv("AUDIO_EMBEDDING_DIR_NAME"),
        is_train_data=False,
        using_personalized_data=True,
    )
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    wav_file_path = []
    for data, label in dataloader:
        wav_file_path.append(data["wav_file_path"][0])
    print(len(wav_file_path))