import logging
import os
import sys
from collections import defaultdict
from typing import List

import learn2learn as l2l
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch.nn.functional as F

from models.PDMER import PDMERModel
from utils.dataset import split_batch_seq
from utils.inference import model_wrap, slide_inference
from utils.score import ccc, get_score, pcc, rmse

name_list = ["arousal", "valence"]


def get_attention_loss(
    attention_maps: torch.Tensor,  # [batch_size, layer, head, seq_length, seq_length]
    target_attention_maps: List[
        torch.Tensor
    ],  # [ [batch_size, seq_length, seq_length] * layer]
):
    attention_map_list = [
        x.squeeze() for x in torch.split(attention_maps, 1, dim=1)
    ]  # [ [batch_size, head, seq_length, seq_length] * layer]
    loss = 0.0
    for i in range(len(attention_map_list)):
        attention_map = attention_map_list[
            i
        ]  # [batch_size, head, seq_length, seq_length]
        if len(attention_map.shape) == 4:
            attention_map = attention_map.mean(
                dim=1
            )  # [batch_size, seq_length, seq_length]
        loss += F.mse_loss(attention_map, target_attention_maps[i])
    return loss


def get_loss_with_batch(
    model: torch.nn.Module,
    batch: list,
    criterion,
    device: str,
    audio_input_key: str,
    is_validation: bool = False,
    image_bind_input_key: str = "imagebind_audio_embedding",
    args: dict = None,
):
    embedding_dict, labels = batch
    attention_maps = None  # Only When `is_validation` = True, this will return the first attention map

    # Remove all keys that is not torch.Tensor
    embedding_dict = {
        key: value
        for key, value in embedding_dict.items()
        if isinstance(value, torch.Tensor)
    }

    batch_size, first_seq_len = next(iter(embedding_dict.values())).shape[:2]

    if is_validation and first_seq_len >= 60:
        for key, value in embedding_dict.items():
            assert not torch.isnan(
                value
            ).any(), f"Tensor ({key}, {value.shape}) contains NaNs!"

        # NOTE: this only support batch_size = 1, because all the test data's length is not same
        model_output_dict = slide_inference(model, embedding_dict=embedding_dict)
        seq_len = model_output_dict["model_output"][0].shape[1]
        # Remove the NaN value
        labels = [label[:, :seq_len] for label in labels]
        # Assert there are NO NaN
        for label in labels:
            assert (
                torch.isnan(label).sum() == 0
            ), f"NaN in label: {label}, seq_len: {seq_len}"
    else:
        model_output_dict = model_wrap(
            model=model,
            embedding_dict=embedding_dict,
            output_attentions=True,
        )  # [batch_size, seq_len], [batch_size, seq_len]

    outputs = model_output_dict["model_output"]
    attention_maps = model_output_dict["attention_maps"]

    loss = 0.0
    scores = defaultdict(float)

    batch_size = outputs[0].shape[0]

    assert (
        len(outputs) == len(labels) == len(name_list)
    ), f"The number of outputs and labels should be the same, len(outputs): {len(outputs)}, len(labels): {len(labels)}, len(name_list): {len(name_list)}"
    assert (
        outputs[0].shape == labels[0].shape
    ), f"The shape of outputs and labels should be the same, outputs[0].shape: {outputs[0].shape}, labels[0].shape: {labels[0].shape}"

    criterion_loss = 0
    attention_map_loss = 0

    for i in range(len(labels)):
        for j in range(batch_size):
            score = get_score(outputs[i][j], labels[i][j], return_tensor=True)
            for key, value in score.items():
                scores[f"{name_list[i]}_{key}"] += value

    # # TODO support validation
    if attention_maps is not None:

        # [32, 6, 16, 60, 60]
        average_maps = attention_maps.mean(dim=2)  # [32, 6, 60, 60]
        local_attention_map = average_maps[
            :, args.model_num_hidden_layers - 1, :, :
        ]  # [32, 60, 60]
        global_attention_map = average_maps[:, -1, :, :]  # [32,60, 60]
        local_attention_map = local_attention_map.mean(dim=0)  # [60, 60]
        global_attention_map = global_attention_map.mean(dim=0)

        attention_map_loss += F.mse_loss(
            torch.diag(local_attention_map),
            torch.full_like(torch.diag(local_attention_map), 0.9),
        )
        attention_map_loss += F.mse_loss(
            torch.diag(global_attention_map),
            torch.full_like(torch.diag(global_attention_map), 0.1),
        )

    score_dict = {key: value / batch_size for key, value in scores.items()}

    loss = 1 - get_average_score(score_dict)

    loss += attention_map_loss

    return (
        loss,
        {key: value.item() for key, value in score_dict.items()},
        attention_maps if is_validation else None,
        (outputs, labels),
    )


def get_average_score(val_score: dict) -> float:
    # Log average score
    average_score = 0.0
    for key, value in val_score.items():
        if "RMSE" in key:
            average_score += 1 - value
        else:
            average_score += value
    average_score /= len(val_score.items())
    return average_score


def fast_adapter(
    maml: l2l.algorithms.MAML,
    criterion,
    args,
    is_validation=False,
    support_set=None,
    query_set=None,
    writer=None,
    return_dict: bool = False,
):
    learner = maml.clone()
    meta_loss, meta_score = None, None

    if support_set is not None:
        # 1. Use support set to adapt the model
        for i in range(args.adapt_steps):
            meta_loss, meta_score = get_loss_with_batch(
                model=learner,
                batch=support_set,
                criterion=criterion,
                device=args.device,
                audio_input_key=args.audio_input_key,
                args=args,
                is_validation=False,
            )[:2]
            learner.adapt(meta_loss)
            writer and writer.add_scalar(
                f"Loss/{'Validation' if is_validation else 'Training'} /Adapt",
                meta_loss,
                i,
            )

    with torch.no_grad() if is_validation else torch.enable_grad():
        # 2. Use the query set to update the model
        meta_validation_result = get_loss_with_batch(
            model=learner,
            batch=query_set,
            criterion=criterion,
            device=args.device,
            audio_input_key=args.audio_input_key,
            args=args,
            is_validation=is_validation,
        )

    if return_dict:
        current_val_loss, score, current_attention_maps, (outputs, labels) = (
            meta_validation_result
        )
        return {
            "adapt_loss": meta_loss if meta_loss is not None else torch.tensor(0.0),
            "adapt_score": meta_score if meta_score is not None else None,
            "val_loss": current_val_loss,
            "val_score": score,
            "attention_maps": current_attention_maps,
            "outputs": outputs,
            "labels": labels,
        }
    if not is_validation:
        meta_validation_result = meta_validation_result[
            :2
        ]  # Only return the loss and score when train
    return meta_validation_result


def validate_model(
    model: PDMERModel,
    dataset,
    criterion,
    writer: SummaryWriter,
    val_mode: str = "Validation",
    epoch: int = None,
    args=None,
    min_loss=float("inf"),
    is_maml=True,
):
    # Validation the model (Val + Test)
    # model.eval()
    val_loss = 0.0
    val_score = defaultdict(float)

    attention_maps = []

    indices = np.random.choice(len(dataset), len(dataset), replace=False)
    dataloader = DataLoader(dataset, batch_size=1, sampler=SubsetRandomSampler(indices))

    for i, batch in enumerate(
        tqdm(
            dataloader,
            desc=val_mode,
            leave=False,
        )
    ):
        if is_maml:
            support_set, query_set = split_batch_seq(
                batch, [args.model_feature_num_per_audio, -1]
            )
            current_val_loss, score, current_attention_maps, (outputs, labels) = (
                fast_adapter(
                    maml=model,
                    support_set=support_set,
                    query_set=query_set,
                    criterion=criterion,
                    args=args,
                    is_validation=True,
                    writer=writer,
                )
            )
        else:
            with torch.no_grad():
                current_val_loss, score, current_attention_maps, (outputs, labels) = (
                    get_loss_with_batch(
                        model=model,
                        batch=batch,
                        criterion=criterion,
                        device=args.device,
                        audio_input_key=args.audio_input_key,
                        args=args,
                        is_validation=True,
                    )
                )

        if i < args.attention_map_num and current_attention_maps is not None:
            # Save the attention map, [batch_size=1, layer, head, seq_length, seq_length]
            attention_maps.append(current_attention_maps)

        val_loss += current_val_loss.item()
        for key, value in score.items():
            val_score[key] += value

    # Average the loss and score every batch
    val_loss /= len(dataloader)

    for key, value in val_score.items():
        val_score[key] /= len(dataloader)

    # Log validation loss to TensorBoard
    writer.add_scalar(f"Loss/{val_mode}", val_loss, epoch)

    # Log validation score to TensorBoard
    for key, value in val_score.items():
        writer.add_scalar(f"{key}/{val_mode}", value, epoch)

    # Log average score
    average_score = get_average_score(val_score)

    writer.add_scalar(f"AverageScore/{val_mode}", average_score, epoch)

    return val_loss, val_score


def get_dataloader_with_split_dataset(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,  # NOTE In few-shot learning we need one sample as support set, one sample as query set
    )  # batch_size=1 for validation, because we need to calculate the score for uncertainty length
    return train_loader, test_loader


def print_params_info(model: PDMERModel):

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(f"Total params: {total_params/1e6}M")
    logging.info(f"Trainable params: {trainable_params/1e6}M")

    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        total_params += module_params
        logging.info(f"\t {name}: {module_params/1e6}M")


def save_config(args, writer: SummaryWriter = None):
    # Using JSON to save the configuration
    import json

    with open(os.path.join(args.log_dir, f"{args.train_name}/config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    if writer is not None:
        writer.add_text("Config", json.dumps(vars(args), indent=4))


# Save Model
def save_model(model: PDMERModel, epoch: int, train_name: str, log_dir: str):
    save_dir = os.path.join(log_dir, f"{train_name}/models")
    os.makedirs(save_dir, exist_ok=True)

    save_dict = {}
    for name, param in model.state_dict().items():
        if name.split(".")[0] not in ["imagebind_model"]:
            save_dict[name] = param

    torch.save(save_dict, os.path.join(save_dir, f"epoch_{epoch}.pt"))


# Load the model
def load_model(model: torch.nn.Module, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint, strict=True)
    return model


def create_lr_scheduler(
    optimizer,
    num_step: int,  # every epoch has how much step
    epochs: int,  # total epochs
    warmup=True,  # whether to use warmup
    warmup_epochs=1,  # warmup for how many epochs
    warmup_factor=1e-3,  # warmup factor
    min_factor=0.01,
    down_factor=2,
):
    assert num_step > 0 and epochs > 0

    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return max(
                min_factor,
                (
                    1
                    - (x - warmup_epochs * num_step)
                    / ((epochs - warmup_epochs) * num_step)
                )
                ** (down_factor),
            )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_optimizer(
    optimizer_name: str,
    *args,
    **kwargs,
):
    if not hasattr(torch.optim, optimizer_name):
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")
    return getattr(torch.optim, optimizer_name)(*args, **kwargs)


def get_dataset(DatasetClass, args):
    train_dataset = DatasetClass(
        root=args.dataset_root,
        device=args.device,
        audio_embedding_dir_name=args.audio_embedding_dir_name,
        loading_cache_tensor_in_device=args.loading_cache_tensor_in_device,
        is_train_data=True,
        loading_data_before_use=args.loading_data_before_use,
    )
    test_dataset = DatasetClass(
        root=args.dataset_root,
        device=args.device,
        audio_embedding_dir_name=args.audio_embedding_dir_name,
        loading_cache_tensor_in_device=args.loading_cache_tensor_in_device,
        is_train_data=False,
    )
