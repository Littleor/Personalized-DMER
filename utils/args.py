import argparse
import datetime
import os
from dotenv import load_dotenv


def parse_args(description=None, arguments: list = []):
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(
        description="Train the model" if description is None else description
    )

    for argument in arguments:
        parser.add_argument(*argument["args"], **argument["kwargs"])

    parser.add_argument(
        "--log_dir",
        type=str,
        default=os.environ.get("LOG_DIR"),
        help="The directory to save the logs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=15,
        help="The batch size to use for training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2000,
        help="The number of epochs to train for",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="The learning rate to use for training",
    )
    parser.add_argument(
        "--meta_lr",
        type=float,
        default=5e-3,
        help="The learning rate to use for meta-training",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=4e-6,
        help="The learning rate to use for training",
    )
    parser.add_argument(
        "--lr_down_factor",
        type=float,
        default=4,
        help="The learning rate to use for training",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="The learning rate to use for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:4",
        help="The device to use for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="The random seed to use for training",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.9,
        help="The proportion of the dataset to use for training",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.05,
        help="The proportion of the dataset to use for validation",
    )
    parser.add_argument(
        "--train_name",
        type=str,
        default=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        help="The name of the train",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="DEAM",
        help="The name of the dataset",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.getenv("DATASET_PATH", "/data/dataset/DEAM"),
        help="The root directory of the DEAM dataset",
    )
    parser.add_argument(
        "--audio_embedding_dir_name",
        type=str,
        default=os.getenv("AUDIO_EMBEDDING_DIR_NAME", "audio_embedding"),
        help="The name of the audio embedding directory",
    )
    parser.add_argument(
        "--save_every_epoch",
        type=int,
        default=50,
        help="Save model every n epoch",
    )
    # Model params
    parser.add_argument(
        "--model_query_embed_dim",
        type=int,
        default=128,
        help="The query embed dim of the model",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=-1,
        help="The embed dim before enter the QFormer, if -1, it will not use the adapter, else will using a adpter to change the embed dim",
    )
    parser.add_argument(
        "--model_num_attention_heads",
        type=int,
        default=16,
        help="The number of attention heads",
    )
    parser.add_argument(
        "--model_num_hidden_layers",
        type=int,
        default=3,
        help="The number of hidden layers",
    )
    parser.add_argument(
        "--model_segmentation_duration",
        type=int,
        default=5,
        help="The duration of the segmentation",
    )
    parser.add_argument(
        "--model_feature_num_per_audio",
        type=int,
        default=60,
        help="The number of feature per audio",
    )
    parser.add_argument(
        "--model_train_audio_duration",
        type=int,
        default=30,
        help="The duration of the train audio",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.1,
        help="The dropout rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="The weight decay",
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        default=3072,
        help='Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.',
    )
    parser.add_argument(
        "--hidden_act",
        type=str,
        default="gelu",
        help='The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,`"relu"`, `"silu"` and `"gelu_new"` are supported.',
    )
    parser.add_argument(
        "--max_position_embeddings",
        type=int,
        default=512,
        help="The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).",
    )
    parser.add_argument(
        "--loading_cache_tensor_in_device",
        action="store_true",
        help="Whether to load the cache tensor in device, if False, the cache tensor will be loaded in cpu.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="The optimizer to use for training",
    )
    parser.add_argument(
        "--audio_input_key",
        type=str,
        default=os.environ.get("AUDIO_INPUT_KEY"),
        help='The key of the audio input, should be "log_mel_spectrogram" or "fbank_feature"',
    )
    parser.add_argument(
        "--attention_map_num",
        type=int,
        default=2,
        help="The number of attention map to visualize",
    )
    parser.add_argument(
        "--local_context_length",
        type=int,
        default=5,
        help="The length of the local context",
    )
    parser.add_argument(
        "--global_context_length",
        type=int,
        default=30,
        help="The length of the global context",
    )
    parser.add_argument(
        "--position_embedding_type",
        type=str,
        default="relative_key",
        help="The position embedding type",
    )
    parser.add_argument(
        "--adapt_steps",
        type=int,
        default=16,
        help="The MAML adaptation steps",
    )
    parser.add_argument(
        "--loading_data_before_use",
        action="store_true",
        help="Whether to cache all the data to `cpu` or `cuda` before train",
    )
    parser.add_argument(
        "--not_using_cache_tensor",
        action="store_true",
        help="Whether not to use the `pt` cache tensor",
    )
    parser.add_argument(
        "--num_shot",
        type=int,
        default=1,
        help="The support set size of the few-shot learning",
    )
    parser.add_argument(
        "--meta_batch_count",
        type=int,
        default=16,
        help="The meta batch count of the few-shot learning",
    )
    parser.add_argument(
        "--not_using_maml",
        action="store_true",
        help="Whether not to use MAML for training",
    )
    parser.add_argument(
        "--using_personalized_data_train",
        action="store_true",
        help="Whether to use the personalized data",
    )
    parser.add_argument(
        "--using_personalized_data_validate",
        action="store_true",
        help="Whether to use the personalized data for validate",
    )

    args = parser.parse_args()

    if args.seed == -1:
        args.seed = int(datetime.datetime.now().timestamp()) % 1000000

    if args.dataset_name == "PMEmo":
        args.dataset_root = os.getenv("PMEMO_DATASET_PATH", "/data/dataset/PMEmo")
        args.audio_embedding_dir_name = os.getenv(
            "PMEMO_AUDIO_EMBEDDING_DIR_NAME", "audio_embedding"
        )
        args.model_feature_num_per_audio = 20

    return args
