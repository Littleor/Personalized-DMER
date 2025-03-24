import io
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import librosa
import soundfile as sf
import numpy as np
import logging
import torch

import torchaudio
import torchaudio.transforms as transforms


def read_wav_as_tensor(
    filepath: str, sample_rate: int = 44100, duration: Tuple[float, float] = None
) -> Tuple[torch.Tensor, int]:
    """Read the wav file as tensor

    Args:
        filepath (str): The path of the wav file
        sample_rate (int, optional): The sample rate of the audio. Defaults to 44100.
        duration (Tuple[float, float], optional): The duration of the audio, will fisrt clip this audio to the duration. Defaults to None, means no clip.

    Returns:
        Tuple[torch.Tensor, int]: The waveform tensor and the sample rate
    """
    waveform, sr = torchaudio.load(filepath)
    if sample_rate != sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=sample_rate
        )
    if duration is not None:
        begin, end = duration
        waveform = waveform[
            :,
            int(begin * sample_rate) : (
                int(end * sample_rate) if end is not None else None
            ),
        ]
        if end is not None and waveform.shape[1] < sample_rate * (end - begin):
            if (sample_rate * (end - begin) - waveform.shape[1]) / (
                sample_rate * (end - begin)
            ) > 0.1:
                logging.warning(
                    f"Audio {filepath} duration ({round(waveform.shape[1] / sample_rate, 2)}s) is much less than {end - begin} seconds"
                )

            # padding
            waveform = torch.cat(
                [
                    waveform,
                    torch.zeros(
                        waveform.shape[0],
                        int(sample_rate * (end - begin) - waveform.shape[1]),
                    ).to(waveform.device),
                ],
                dim=1,
            )

    if waveform.shape[0] != 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


def generate_split_duration_list(
    filepath: str,
    sample_rate: int = 44100,
    length_each_clip: float = 0.5,
    begin_time: float = 15.0,
    end_time: float = None,
    slide_length: float = 0.5,
    slide_start: float = 0.25,
    min_duration: float = None,
):
    """Generate the split duration list

    Args:
        filepath (str): The path of the wav file
        sample_rate (int, optional): The sample rate of the audio. Defaults to 44100.
        length_each_clip (float, optional): The length of each clip. Defaults to 0.5.
        begin_time (float, optional): The begin time of the audio. Defaults to 15..
        end_time (float, optional): The end time of the audio. Defaults to None.
        slide_start (float, optional): The start of the slide. Defaults to 0.25.

    Returns:
        list: The split duration list
    """
    if end_time is None:
        waveform, sr = read_wav_as_tensor(
            filepath, sample_rate=sample_rate, duration=(begin_time, end_time)
        )
        duration = waveform.shape[1] / sample_rate
    else:
        duration = end_time - begin_time

    if min_duration is not None:
        duration = max(duration, min_duration)

    return [
        (
            begin_time + slide_start + -1 * length_each_clip / 2 + i * slide_length,
            begin_time + slide_start + i * slide_length + length_each_clip / 2,
        )
        for i in range(math.floor(duration / slide_length))
    ]


def read_wav(
    filepath: str, target_sr: int = 44100, duration: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """Read a wav file, either local on on GCS."""

    with open(filepath, "rb") as f:
        bytes_as_string = f.read()

    # Samplerate does not allow to specify sr when reading; if desired, the audio will need to be resampled in a postprocessing step.
    # For some reason, librosa fails to read due to an issue with lazy-loading of modules when executed within a beam pipeline.
    samples, audio_sr = sf.read(
        io.BytesIO(bytes_as_string),
        frames=math.floor(target_sr * duration) if duration is not None else -1,
    )
    logging.debug(
        f"Reading audio from {filepath} with sr {audio_sr} "
        f"with duration {round(len(samples)/audio_sr,2)} secs"
    )

    if audio_sr != target_sr:
        logging.debug(
            f"resampling audio input {filepath} from {audio_sr} to {target_sr}"
        )
        samples = librosa.resample(samples, orig_sr=audio_sr, target_sr=target_sr)

    assert np.issubdtype(
        samples.dtype, float
    ), f"exected floating-point audio; got type {samples.dtype}"

    return samples, target_sr


def read_element_wav(
    elem: Dict[str, Any],
    filepath: str,
    target_sr=44100,
    duration: Optional[float] = None,
) -> Dict[str, Any]:
    samples, sr = read_wav(filepath=filepath, target_sr=target_sr, duration=duration)
    elem["audio"] = samples
    elem["audio_sample_rate"] = sr
    return elem


def music_segmentation(
    audio_paths: List[str],
    sample_rate: int = 44100,
    clip_audio: tuple = None,
    audio_segmentation_list: List[List[tuple]] = None,
) -> np.ndarray:
    """Segment a music audio into feature_num_per_audio segments."""

    waveform_segmentaion_list = []

    # We should use the true time of this audio as the clip duration, so we now need just adjust the audio_segmentation_list to the new audio time
    if clip_audio is not None:
        begin, end = clip_audio
        for i in range(len(audio_segmentation_list)):
            for j in range(len(audio_segmentation_list[i])):
                audio_segmentation_list[i][j] = (
                    audio_segmentation_list[i][j][0] - begin,
                    audio_segmentation_list[i][j][1] - begin,
                )

    for audio_index, audio_path in enumerate(audio_paths):
        waveform, sr = read_wav_as_tensor(
            filepath=audio_path,
            sample_rate=sample_rate,
            duration=clip_audio,
        )

        for i in range(len(audio_segmentation_list[audio_index])):
            # 2. Segment the audio
            audio_segmentation = audio_segmentation_list[audio_index][i]
            current_waveform: torch.Tensor = None
            if audio_segmentation[0] < 0:
                current_waveform = waveform[
                    :, : int(audio_segmentation[1] * sample_rate)
                ]
                current_waveform = torch.cat(
                    [
                        current_waveform[
                            :, : -1 * int(audio_segmentation[0] * sample_rate)
                        ],
                        current_waveform,
                    ],
                    dim=1,
                )
            elif audio_segmentation[1] * sample_rate > waveform.size(1):
                overlap = int(audio_segmentation[1] * sample_rate - waveform.size(1))
                current_waveform = waveform[
                    :, int(audio_segmentation[0] * sample_rate) :
                ]
                current_waveform = torch.cat(
                    [
                        current_waveform,
                        current_waveform[:, current_waveform.size(1) - overlap :],
                    ],
                    dim=1,
                )
            else:
                current_waveform = waveform[
                    :,
                    int(audio_segmentation[0] * sample_rate) : int(
                        audio_segmentation[1] * sample_rate
                    ),
                ]

            waveform_segmentaion_list.append(current_waveform.clone())

    return torch.cat(waveform_segmentaion_list, dim=0)


# Get the audio fbank feature with audio file path
def get_audio_fbank_feature(
    audio_file_path_list: list,
    sample_rate: int = 44100,
    n_mels: int = 128,
    frame_length: int = 25,  # milliseconds
    frame_shift: int = 10,  # milliseconds
    duration: tuple = (15, 45),
) -> torch.Tensor:
    """Get the audio fbank feature with audio file path

    Args:
        audio_file_path_list (list): The path of the audio file
        sample_rate (int, optional): The sample rate of the audio. Defaults to 44100.
        n_mels (int, optional): The number of mel filters. Defaults to 128.
        frame_length (int, optional): The frame length of the audio. Defaults to 25.
        frame_shift (int, optional): The frame shift of the audio. Defaults to 10.
        duration (tuple, optional): The duration of the audio. Defaults to (15, 45).

    Returns:
        torch.Tensor: The fbank feature of the audio
    """
    if isinstance(audio_file_path_list, str):
        audio_file_path_list = [audio_file_path_list]

    output = []
    for audio_file_path in audio_file_path_list:
        if isinstance(duration[0], int):
            duration = [duration]

        waveform_start_time, waveform_end_time = duration[0][0], duration[-1][-1]
        full_waveform, sr = read_wav_as_tensor(
            filepath=audio_file_path,
            sample_rate=sample_rate,
            duration=(waveform_start_time, waveform_end_time),
        )
        current_output = []
        for current_duration in duration:

            begin, end = current_duration
            waveform = full_waveform[
                :,
                int((begin - waveform_start_time) * sr) : int(
                    (end - waveform_start_time) * sr
                ),
            ]

            assert sr == sample_rate, "The sample rate should be the same."
            assert end > begin, "The end time should be larger than the begin time."

            n_frames = (1000 * (end - begin) - frame_length) // frame_shift + 1

            fbank = torchaudio.compliance.kaldi.fbank(
                waveform,
                htk_compat=True,
                window_type="hanning",
                num_mel_bins=n_mels,
                frame_length=frame_length,
                frame_shift=frame_shift,
                sample_frequency=sample_rate,
            )  # shape (n_frames, num_mel_bins)

            assert (
                n_frames == fbank.shape[0]
            ), f"The number of frames should be the same: n_frames {n_frames} vs fbank {fbank.shape}"
            current_output.append(fbank)
        current_output = torch.stack(current_output, dim=0)
        output.append(current_output)

    output = torch.stack(output, dim=0)
    return output


def get_audio_log_mel_spec(
    audio_file_path_list: list,
    sample_rate: int = 44100,
    n_mels: int = 128,
    frame_length: int = 25,  # milliseconds
    frame_shift: int = 10,  # milliseconds
    duration: list = [[(15, 45)]],  # [batch_size, seq_len, 2]
) -> torch.Tensor:
    """Get the log mel spectrogram with audio file path

    Args:
        audio_file_path_list (list): The path of the audio file
        sample_rate (int, optional): The sample rate of the audio. Defaults to 44100.
        n_mels (int, optional): The number of mel filters. Defaults to 128.
        frame_length (int, optional): The frame length of the audio. Defaults to 25.
        frame_shift (int, optional): The frame shift of the audio. Defaults to 10.
        duration (tuple, optional): The duration of the audio. Defaults to (15, 45).

    Returns:
        torch.Tensor: The log mel spectrogram of the audio
    """
    if isinstance(audio_file_path_list, str):
        audio_file_path_list = [audio_file_path_list]
    # if isinstance(duration[0], int) or isinstance(duration[0], float):
    #     duration = [[duration]]
    # elif not isinstance(duration[0][0], list):
    #     duration = [duration]
    assert len(duration) == len(
        audio_file_path_list
    ), "The audio count and the duration length should be the same."

    output = []
    for i, audio_file_path in enumerate(audio_file_path_list):
        waveform_start_time, waveform_end_time = duration[i][0][0], duration[i][-1][-1]
        full_waveform, sr = read_wav_as_tensor(
            filepath=audio_file_path,
            sample_rate=sample_rate,
            duration=(waveform_start_time, waveform_end_time),
        )
        current_output = []
        waveform_list = []
        for current_duration in duration[i]:

            begin, end = current_duration
            waveform = full_waveform[
                :,
                int((begin - waveform_start_time) * sr) : int(
                    (end - waveform_start_time) * sr
                ),
            ]
            waveform_list.append(waveform)

            assert sr == sample_rate, "The sample rate should be the same."
            assert end > begin, "The end time should be larger than the begin time."

        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=int(sample_rate * frame_length / 1000),
            win_length=int(sample_rate * frame_length / 1000),
            hop_length=int(sample_rate * frame_shift / 1000),
            n_mels=n_mels,
        )(
            torch.cat(waveform_list, dim=0)
        )  # shape (n_frames, num_mel_bins)
        log_mel_spec = torch.log(
            mel_spectrogram + 1e-6
        )  # Adding a small value to avoid log(0)

        # Swap the time and frequency axis
        log_mel_spec = log_mel_spec.permute(0, 2, 1)
        output.append(log_mel_spec)

    output = torch.stack(output, dim=0)
    return output


def get_log_mel_band_energy(
    audio_file_path_list: list,
    sample_rate: int = 44100,
    n_mels: int = 64,
    duration: tuple = [[(15, 45)]],  # [batch_size, seq_len, 2]
) -> torch.Tensor:
    """Get the log mel spectrogram with audio file path

    Args:
        audio_file_path_list (list): The path of the audio file
        sample_rate (int, optional): The sample rate of the audio. Defaults to 44100.
        n_mels (int, optional): The number of mel filters. Defaults to 128.
        frame_length (int, optional): The frame length of the audio. Defaults to 25.
        frame_shift (int, optional): The frame shift of the audio. Defaults to 10.
        duration (tuple, optional): The duration of the audio. Defaults to (15, 45).

    Returns:
        torch.Tensor: The log mel spectrogram of the audio
    """
    if isinstance(audio_file_path_list, str):
        audio_file_path_list = [audio_file_path_list]
    # if isinstance(duration[0], int) or isinstance(duration[0], float):
    #     duration = [[duration]]
    # elif not isinstance(duration[0][0], list):
    #     duration = [duration]
    assert len(duration) == len(
        audio_file_path_list
    ), "The audio count and the duration length should be the same."

    output = []
    for i, audio_file_path in enumerate(audio_file_path_list):
        waveform_start_time, waveform_end_time = duration[i][0][0], duration[i][-1][-1]
        full_waveform, sr = read_wav_as_tensor(
            filepath=audio_file_path,
            sample_rate=sample_rate,
            duration=(waveform_start_time, waveform_end_time),
        )
        current_output = []
        waveform_list = []
        for current_duration in duration[i]:

            begin, end = current_duration
            waveform = full_waveform[
                :,
                int((begin - waveform_start_time) * sr) : int(
                    (end - waveform_start_time) * sr
                ),
            ]
            waveform_list.append(waveform)

            assert sr == sample_rate, "The sample rate should be the same."
            assert end > begin, "The end time should be larger than the begin time."

        # mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        #     sample_rate=sample_rate,
        #     n_fft=int(sample_rate * frame_length / 1000),
        #     win_length=int(sample_rate * frame_length / 1000),
        #     hop_length=int(sample_rate * frame_shift / 1000),
        #     n_mels=n_mels,
        # )(
        #     torch.cat(waveform_list, dim=0)
        # )  # shape (n_frames, num_mel_bins)

        mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=sr,
            n_mels=n_mels,
            n_fft=(sr - 1) // 2,
            hop_length=(sr - 1) // 2 + 1,
        )(torch.cat(waveform_list, dim=0))

        log_mel_spec = transforms.AmplitudeToDB()(mel_spectrogram)

        # Swap the time and frequency axis
        log_mel_spec = log_mel_spec.permute(0, 2, 1)
        output.append(log_mel_spec)

    output = torch.stack(output, dim=0)

    return output.squeeze(0)


# def get_log_mel_band_energy(file_path, sr=44100, n_mels=64, hop_length=512):
#     waveform, sample_rate = torchaudio.load(file_path)
#     if sample_rate != sr:
#         resample_transform = transforms.Resample(orig_freq=sample_rate, new_freq=sr)
#         waveform = resample_transform(waveform)

#     mel_spectrogram = transforms.MelSpectrogram(
#         sample_rate=sr, n_mels=n_mels, hop_length=hop_length
#     )(waveform)

#     log_mel_spectrogram = transforms.AmplitudeToDB()(mel_spectrogram)
#     log_mel_spectrogram = log_mel_spectrogram.squeeze(
#         0
#     )  # Remove channel dimension if exists
#     return log_mel_spectrogram


if __name__ == "__main__":
    x = get_audio_fbank_feature(
        "/data/dataset/DEAM/DEAM_audio/wav_audio/2.wav",
        frame_length=60,  # milliseconds
        frame_shift=10,  # milliseconds
        duration=(15, 45),
    )
    print("30s in one", x.shape)
    x = get_audio_fbank_feature(
        "/data/dataset/DEAM/DEAM_audio/wav_audio/2.wav",
        frame_length=60,  # milliseconds
        frame_shift=10,  # milliseconds
        duration=[(15 + i * 0.5, 15 + (i + 1) * 0.5) for i in range(60)],
    )
    print("30s in 60", x.shape)

    print("--- Log Mel Spec ---")
    x = get_audio_log_mel_spec(
        "/data/dataset/DEAM/DEAM_audio/wav_audio/2.wav",
        frame_length=60,  # milliseconds
        frame_shift=10,  # milliseconds
        duration=[[(15, 45)]],
    )
    print("30s in one", x.shape)
    x = get_audio_log_mel_spec(
        "/data/dataset/DEAM/DEAM_audio/wav_audio/2.wav",
        frame_length=60,  # milliseconds
        frame_shift=10,  # milliseconds
        duration=[[(15 + i * 0.5, 15 + (i + 1) * 0.5) for i in range(60)]],
    )
    print("30s in 60", x.shape)

    print("--- Log Mel Band Energy ---")
    x = get_log_mel_band_energy(
        "/data/dataset/DEAM/DEAM_audio/wav_audio/2.wav",
        duration=[[(15 + i * 0.5, 15 + (i + 1) * 0.5) for i in range(60)]],
    )
    print(
        "Log Mel Band Energy",
        x.shape,
    )
