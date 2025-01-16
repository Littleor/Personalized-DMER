import glob
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List

import librosa
import pandas as pd
import soundfile as sf
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


class DEAMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        duration: float = 30.0,
        sample_rate: int = 44100,
        using_cache_tensor: bool = True,
        device: str = "cpu",
        audio_embedding_dir_name: str = "audio_embedding",
        loading_data_before_use: bool = False,
        loading_cache_tensor_in_device: bool = False,
        is_train_data: bool = True,
        using_personalized_data=False,
    ):
        """DEAM dataset

        Args:
            root (str): The root directory of the dataset
            duration (float, optional): The duration of the audio. Defaults to 30.0.
            sample_rate (int, optional): The sample rate of the audio. Defaults to 44100.
            using_cache_tensor (bool, optional): Whether to use cache tensor. Defaults to True.
            device (str, optional): The device to load the data. Defaults to "cpu".
            audio_embedding_dir_name (str, optional): The directory name of the audio embedding. Defaults to "audio_embedding".
            loading_data_before_use (bool, optional): Whether to load the cache tensor when load data, this will speed up the training process. Defaults to True.
            loading_cache_tensor_in_device (bool, optional): Whether to load the cache tensor in device, if False, the cache tensor will be loaded in cpu. Defaults to False.
            using_personalized_data (bool, optional): Whether to use the personalized data, if `True` will only use the data with workId. Defaults to False.
        """
        self.root = root
        if is_train_data:
            self.duration = duration
        else:
            self.duration = (
                626.5 - 15.0
            )  # Because the DEAM 2015 has the max time is 626500ms
        self.sample_rate = sample_rate
        self.data = []  # Wav file path
        self.labels = []  # ([arousal], [valence])
        self.ids = []  # [2, 3, ...]
        self.using_cache_tensor = using_cache_tensor
        self.device = device
        self.audio_embedding_dir_name = audio_embedding_dir_name
        self.loading_data_before_use = loading_data_before_use
        self.loading_cache_tensor_in_device = loading_cache_tensor_in_device
        self.is_train_data = is_train_data
        self.racer_id_dict = None
        self.using_personalized_data = using_personalized_data

        self.cache_music_data = {}

        logging.info(
            f"Loading DEAM {'train' if is_train_data else 'test'} dataset ({os.path.join(root,'DEAM_audio', audio_embedding_dir_name)}), we will cache the tensor to device: {self.device if loading_cache_tensor_in_device else 'cpu'}, use with {self.device} device ..."
        )
        self.load_data()
        logging.info(
            f"DEAM dataset loaded, there are {len(self)} samples (We {'[bold red]DO[/bold red]' if self.using_personalized_data else 'do not'} use the Personalized data).",
            extra={"markup": True},
        )

    def load_data(self):
        # Load labels and ids
        arousal_csv_file_path = os.path.join(
            self.root,
            "DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv",
        )
        valence_csv_file_path = os.path.join(
            self.root,
            "DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/valence.csv",
        )

        # Read from csv file
        def read_from_csv_file(file_path: str):
            df = pd.read_csv(file_path, delimiter=",")
            keys = df.iloc[:, 0].tolist()
            # Crop the data to the duration
            data = df.iloc[:, 1 : int(2 * self.duration) + 1].values
            data = torch.tensor(data, dtype=torch.float, device=self.device)

            data_dict = dict(zip(keys, data))
            return data_dict

        arousal_data = read_from_csv_file(arousal_csv_file_path)
        valence_data = read_from_csv_file(valence_csv_file_path)

        if self.using_personalized_data:
            arousal_data_dict = self.loading_personalized_data(type="arousal")
            valence_data_dict = self.loading_personalized_data(type="valence")
            for id, data in arousal_data.items():
                arousal_data_dict[id]["all"] = data  # Add the data with all worker id
            for id, data in valence_data.items():
                valence_data_dict[id]["all"] = data
            arousal_data = arousal_data_dict
            valence_data = valence_data_dict

        assert len(arousal_data) == len(valence_data)

        for key in arousal_data.keys():
            music_id = int(key)
            if music_id <= 2000 and not self.is_train_data:
                # remove train data
                continue

            if (5000 > music_id > 2000) and self.is_train_data:
                # remove test data
                continue

            if self.is_train_data and self.using_personalized_data:
                if music_id <= 1000:
                    # remove the non-personalized data
                    continue

            if key in valence_data.keys():
                if isinstance(arousal_data[key], torch.Tensor):
                    self.labels.append((arousal_data[key], valence_data[key]))
                    self.ids.append(music_id)
                else:
                    for worker_id in arousal_data[key].keys():
                        if worker_id in valence_data[key].keys():
                            self.labels.append(
                                (
                                    arousal_data[key][worker_id],
                                    valence_data[key][worker_id],
                                )
                            )
                            self.ids.append(f"{music_id}-{worker_id}")

        if self.using_personalized_data:
            racer_id_dict: Dict[str, List[int]] = self.get_music_worker_id_dict()
            filter_racer_id_dict = {
                k: v for k, v in racer_id_dict.items() if len(v) <= 15
            }
            for worker_id, music_id_list in filter_racer_id_dict.items():
                for music_id in music_id_list:
                    idx = self.ids.index(f"{music_id}-{worker_id}")
                    self.labels.pop(idx)
                    self.ids.pop(idx)
                self.racer_id_dict.pop(worker_id)

        # Convert mp3 to wav
        if not os.path.exists(os.path.join(self.root, "DEAM_audio/wav_audio")):

            logging.info("Converting mp3 to wav...")

            mp3_files = file_list_sort(
                glob.glob(os.path.join(self.root, "DEAM_audio/MEMD_audio", f"*.mp3"))
            )

            os.makedirs(os.path.join(self.root, "DEAM_audio/wav_audio"), exist_ok=True)

            # Using the librosa or soundfile to convert mp3 to wav
            for mp3_file_path in tqdm(mp3_files):
                wav_file_path = mp3_file_path.replace(".mp3", ".wav").replace(
                    "MEMD_audio", "wav_audio"
                )
                audio_data, sr = librosa.load(mp3_file_path, sr=self.sample_rate)
                sf.write(wav_file_path, audio_data, sr)

        for i, music_id_str in enumerate(
            tqdm(self.ids, leave=False, desc="Loading data")
        ):
            if self.using_personalized_data:
                id = int(music_id_str.split("-")[0])
                worker_id = music_id_str.split("-")[1]
            else:
                id = int(music_id_str)

            wav_file_path = check_file_exist(
                os.path.join(
                    self.root,
                    "DEAM_audio/wav_audio/{}.wav".format(id),
                )
            )

            if self.using_cache_tensor:
                # Use the cache tensor from `.pt` file
                pt_file_path = check_file_exist(
                    os.path.join(
                        self.root,
                        "DEAM_audio/{}/{}.pt".format(self.audio_embedding_dir_name, id),
                    )
                )
                if self.loading_data_before_use:
                    # Load all tensor device to speed up
                    if id in self.cache_music_data:
                        embedding = self.cache_music_data[id]
                    else:
                        embedding = torch.load(
                            # Too large to load to GPU
                            pt_file_path,
                            map_location=torch.device(
                                self.device
                                if self.loading_cache_tensor_in_device
                                else "cpu"
                            ),
                        )
                        self.cache_music_data[id] = embedding
                    self.data.append(embedding)
                else:
                    self.data.append(pt_file_path)
            else:
                self.data.append(wav_file_path)
        assert len(self.labels) == len(self.data) == len(self.ids)

    def __len__(self):
        return len(self.data)

    def get_indices_from_music_id(self, music_id_list: list, worker_id: str):
        return [self.ids.index(f"{music_id}-{worker_id}") for music_id in music_id_list]

    def loading_personalized_data(self, type="arousal"):
        music_labels = defaultdict(dict)

        racer_data_dir_path = os.path.join(
            self.root,
            f"DEAM_Annotations/annotations/annotations per each rater/dynamic (per second annotations)/{type}",
        )
        for file_name in tqdm(
            sorted(os.listdir(racer_data_dir_path)), desc="Loading personalized data"
        ):
            if file_name.endswith(".csv"):
                file_path = os.path.join(
                    racer_data_dir_path,
                    file_name,
                )
                id = int(file_name.split(".")[0])

                if (not self.is_train_data and id <= 2000) or (
                    self.is_train_data and id > 2000
                ):
                    continue

                df = pd.read_csv(file_path, delimiter=",")
                if "WorkerId" not in df.columns:
                    continue

                keys = df.iloc[:, 0].tolist()
                # Crop the data to the duration
                data = df.iloc[:, 1 : int(2 * self.duration) + 1].values
                data = torch.tensor(data, dtype=torch.float, device=self.device)

                data_dict = dict(zip(keys, data))

                for worker_id, value in data_dict.items():
                    diff = torch.abs(value[:60] - torch.mean(value[:60])).mean().item()
                    if diff < 0.01:
                        # logging.info(
                        #     f"Skip the data with the mean value is too small({diff}, {worker_id} - {id})."
                        # )
                        continue
                    music_labels[id][worker_id] = value

        return music_labels

    def get_music_worker_id_dict(self):
        if self.racer_id_dict is not None:
            return self.racer_id_dict

        racer_id_dict = defaultdict(list)
        for id_str in self.ids:
            music_id, worker_id = id_str.split("-")
            music_id = int(music_id)
            racer_id_dict[worker_id].append(music_id)

        cache_json_file_path = os.path.join(
            self.root,
            f"DEAM_Annotations/{'train' if self.is_train_data else 'test'}_racer_id_dict.json",
        )

        self.racer_id_dict = racer_id_dict
        with open(
            cache_json_file_path,
            "w",
        ) as f:
            json.dump(racer_id_dict, f, indent=4, ensure_ascii=False)
        return racer_id_dict

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if not self.loading_data_before_use and isinstance(sample, str):
            if sample.endswith(".pt"):
                pt_file_path = sample
                sample = torch.load(
                    pt_file_path, map_location=torch.device(self.device)
                )
        if isinstance(sample, dict):
            for keys in sample.keys():
                if isinstance(sample[keys], torch.Tensor):
                    sample[keys] = sample[keys].to(self.device)
            if self.using_personalized_data:
                music_id = self.ids[idx].split("-")[0]
                worker_id = self.ids[idx].split("-")[1]
            else:
                music_id = self.ids[idx]
                worker_id = "all"
            sample["wav_file_path"] = os.path.join(
                self.root, "DEAM_audio/wav_audio", f"{music_id}.wav"
            )
            sample["worker_id"] = worker_id

        return sample, label
