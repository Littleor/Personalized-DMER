import os
import sys
from typing import Optional

import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from utils.mask import generate_short_context_mask
from models.layer.transformer_encoder import TransformerEncoder


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        local_attention_model: nn.Module,
        global_attention_model: nn.Module,
        device="cpu",
        local_context_length=3,
        global_context_length=30,
        using_local_attention=True,
        using_gloabl_attention=True,
    ):
        super(MultiScaleAttention, self).__init__()

        self.device = device
        self.dtype = torch.float32

        self.local_context_length = local_context_length
        self.global_context_length = global_context_length
        self.using_local_attention = using_local_attention
        self.using_gloabl_attention = using_gloabl_attention

        self.global_attention = global_attention_model
        self.local_attention = local_attention_model

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = True,
    ):
        attention_maps = None

        local_attention_mask = (
            generate_short_context_mask(
                hidden_states.shape[1],
                context_size=self.local_context_length,
                sparse=False,
            )
            .to(self.device)
            .unsqueeze(0)
            .expand(hidden_states.size()[0], -1, -1)
        )

        global_attention_mask = (
            generate_short_context_mask(
                hidden_states.shape[1],
                context_size=self.global_context_length,
                sparse=False,
            )
            .to(self.device)
            .unsqueeze(0)
            .expand(hidden_states.size()[0], -1, -1)
        )

        output_attention_map = []
        if self.using_local_attention:
            local_outputs = self.local_attention(
                hidden_states,
                attention_mask=local_attention_mask,
                output_attentions=output_attentions,
            )
            output_attention_map += local_outputs["attentions"]

        if self.using_gloabl_attention:
            global_outputs = self.global_attention(
                hidden_states,
                attention_mask=global_attention_mask,
                output_attentions=output_attentions,
            )
            output_attention_map += global_outputs["attentions"]

        if self.using_local_attention and self.using_gloabl_attention:
            output_hidden_state = torch.sigmoid(
                local_outputs["last_hidden_state"] + global_outputs["last_hidden_state"]
            )
        elif self.using_local_attention:
            output_hidden_state = local_outputs["last_hidden_state"]
        elif self.using_gloabl_attention:
            output_hidden_state = global_outputs["last_hidden_state"]
        else:
            outputs = self.global_attention(
                hidden_states,
                attention_mask=None,
                output_attentions=output_attentions,
            )
            output_hidden_state = outputs["last_hidden_state"]
            output_attention_map += outputs["attentions"]

        if output_attentions and len(output_attention_map) > 0:
            attention_maps = torch.stack(output_attention_map, dim=0).permute(
                1, 0, 2, 3, 4
            )  # [batch_size, layer, head, seq_length, seq_length]

        return output_hidden_state, attention_maps


class TransformerMultiScaleAttention(nn.Module):
    def __init__(
        self,
        origin_feature_dim=1024,
        device="cpu",
        dropout_rate=0.2,
        num_attention_heads=4,
        num_hidden_layers=4,
        context_size=15,
        hidden_act="gelu",
        max_position_embeddings=512,
        position_embedding_type="absolute",
        local_context_length=3,
        global_context_length=30,
        hidden_size=512,
        using_local_attention=True,
        using_gloabl_attention=True,
    ):
        super(TransformerMultiScaleAttention, self).__init__()

        self.device = device
        self.dtype = torch.float32

        self.encoder = TransformerEncoder(
            hidden_size=origin_feature_dim,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=2048,
            hidden_act=hidden_act,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
            max_position_embeddings=max_position_embeddings,
            output_hidden_size=hidden_size,
            position_embedding_type=position_embedding_type,
        )

        self.multi_scale_attention = MultiScaleAttention(
            local_attention_model=self.encoder,
            global_attention_model=self.encoder,
            device=device,
            local_context_length=local_context_length,
            global_context_length=global_context_length,
            using_local_attention=using_local_attention,
            using_gloabl_attention=using_gloabl_attention,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = True,
    ):
        return self.multi_scale_attention(hidden_states, output_attentions)
