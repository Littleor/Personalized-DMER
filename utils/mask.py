import math
import torch
from PIL import Image


def generate_short_context_mask(
    length: int, context_size: int, sparse=False, ignore_future=False
):
    mask = torch.zeros(length, length)
    for i in range(length):

        start = i - context_size
        end = i + context_size + 1
        if sparse:
            pos_list = [0, 0]
            while True:
                next_pos = pos_list[-1] + (pos_list[-1] - pos_list[-2]) + 1
                if next_pos >= context_size:
                    pos_list.pop(0)
                    break
                pos_list.append(next_pos)

            for j in range(start, end):
                if abs(i - j) in pos_list and 0 <= j < length:
                    mask[i, j] = 1
        else:
            start = max(0, start)
            end = min(end, length)
            mask[i, start:end] = 1

        if ignore_future:
            mask[i, i + 1 :] = 0

    return mask


def generate_only_past_mask(length: int):
    return torch.tril(
        torch.ones(
            length,
            length,
        )
    )
