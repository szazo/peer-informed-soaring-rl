from typing import Optional, Literal
import logging
import torch
import torch.nn as nn
from .encoder import Encoder


class Transformer(nn.Module):

    def __init__(self, encoder_embed_dim: int, output_dim: int,
                 attention_internal_dim: int, attention_head_num: int,
                 ffnn_hidden_dim: int, ffnn_dropout_rate: float,
                 softmax_output: bool, encoder_layer_count: int,
                 enable_layer_normalization: bool,
                 encoder_activation: Literal['relu'] | Literal['swish']):

        super().__init__()

        self._log = logging.getLogger(__class__.__name__)

        self._log.debug('encoder embed dimension: %s', encoder_embed_dim)
        self._log.debug('output dimension: %s', output_dim)

        # create the encoder
        encoder_layers = [
            Encoder(input_dim=encoder_embed_dim,
                    attention_internal_dim=attention_internal_dim,
                    attention_head_num=attention_head_num,
                    ffnn_hidden_dim=ffnn_hidden_dim,
                    ffnn_dropout_rate=ffnn_dropout_rate,
                    layer_count=encoder_layer_count,
                    enable_layer_normalization=enable_layer_normalization,
                    activation=encoder_activation)
            for _ in range(encoder_layer_count)
        ]

        self._encoders = nn.ModuleList(encoder_layers)

        self._dense = None
        if encoder_embed_dim != output_dim:
            self._dense = nn.Linear(in_features=encoder_embed_dim,
                                    out_features=output_dim)

        if softmax_output:
            self._softmax = nn.Softmax(dim=-1)
        else:
            self._softmax = None

    def forward(self,
                input_sequence: torch.Tensor,
                decoded_sequence: torch.Tensor,
                input_attention_mask: Optional[torch.Tensor] = None):

        log = self._log

        log.debug('calling encoder, input shape: %s, decoded shape: %s',
                  input_sequence.shape, decoded_sequence.shape)

        encoder_output = input_sequence
        for encoder_layer in self._encoders:
            encoder_output = encoder_layer(encoder_output,
                                           attention_mask=input_attention_mask)

        log.debug('encoder output shape: %s', encoder_output.shape)

        if self._dense is not None:
            dense_output = self._dense(encoder_output)
            log.debug('output shape after dense: %s %s', dense_output.shape,
                      dense_output)
        else:
            dense_output = encoder_output

        log.debug('output shape after optional dense: %s %s',
                  dense_output.shape, dense_output)

        if self._softmax is not None:
            softmax_output = self._softmax(dense_output)
            result = softmax_output
        else:
            result = dense_output

        log.debug('transformer output shape: %s %s', result.shape, result)

        return result
