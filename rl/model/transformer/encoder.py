import logging
from typing import Optional, Literal
from functools import partial
import torch
import torch.nn as nn
from .multihead_attention2 import MultiheadAttention2
from .initialization import initialize_t_fixup_weights


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


# The encoder contains the self attention and feed forward network
class Encoder(nn.Module):

    _dense1_activation: nn.Module

    def __init__(self, input_dim: int, attention_internal_dim: int,
                 ffnn_hidden_dim: int, ffnn_dropout_rate: float,
                 attention_head_num: int, layer_count: int,
                 enable_layer_normalization: bool,
                 activation: Literal['relu'] | Literal['swish']):
        super().__init__()

        self._log = logging.getLogger(__class__.__name__)

        self._enable_layer_normalization = enable_layer_normalization

        param_init_ = partial(self._attention_custom_param_init_,
                              layer_count=layer_count)
        self._self_attention = MultiheadAttention2(
            head_num=attention_head_num,
            internal_dim=attention_internal_dim,
            source_dim=input_dim,
            target_dim=input_dim,  # self attention
            custom_parameter_init_=param_init_)

        if self._enable_layer_normalization:
            # create layer normalization layers, they will normalize along
            # the feature vectors with dimension equal to input_dim
            self._layer_norm1 = nn.LayerNorm(input_dim)
            self._layer_norm2 = nn.LayerNorm(input_dim)

        # create the feed forward neural network with one hidden layer
        self._dense1 = nn.Linear(in_features=input_dim,
                                 out_features=ffnn_hidden_dim)

        if activation == 'swish':
            self._dense1_activation = Swish()
        elif activation == 'relu':
            self._dense1_activation = nn.ReLU(inplace=True)
        else:
            raise ValueError('not supported activation: ' + activation)

        self._dense2 = nn.Linear(in_features=ffnn_hidden_dim,
                                 out_features=input_dim)
        self._dense2_dropout = nn.Dropout(ffnn_dropout_rate)

        initialize_t_fixup_weights(self._dense1.weight,
                                   layer_count=layer_count)
        initialize_t_fixup_weights(self._dense2.weight,
                                   layer_count=layer_count)

    def _attention_custom_param_init_(self, query: torch.nn.ParameterList,
                                      key: torch.nn.ParameterList,
                                      value: torch.nn.ParameterList,
                                      output: torch.Tensor, layer_count: int):

        for query_head_params in query:
            initialize_t_fixup_weights(query_head_params,
                                       layer_count=layer_count)

        for key_head_params in key:
            initialize_t_fixup_weights(key_head_params,
                                       layer_count=layer_count)

        for value_head_params in value:
            initialize_t_fixup_weights(value_head_params,
                                       layer_count=layer_count)

        initialize_t_fixup_weights(output, layer_count=layer_count)

    def forward(self,
                input: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None):

        # the input is a matrix, the rows contains the elements, the column dimension
        # is equal to input_dim
        z = self._self_attention(
            input, input,
            attention_mask=attention_mask)  # self attention, input twice

        if self._enable_layer_normalization:
            # with residual
            add_normalize1_result = self._layer_norm1(input + z)
        else:
            add_normalize1_result = input + z

        log = self._log

        # feed into the feed forward neural network
        dense1_result = self._dense1_activation(
            self._dense1(add_normalize1_result))
        log.debug("shape after first dense layer %s", dense1_result.shape)
        dense2_result = self._dense2_dropout(self._dense2(dense1_result))
        log.debug("shape after second dense layer %s", dense2_result.shape)

        # add the result from the dense layer and state after first normalization
        if self._enable_layer_normalization:
            add_normalize2_result = self._layer_norm2(add_normalize1_result +
                                                      dense2_result)
        else:
            add_normalize2_result = add_normalize1_result + dense2_result

        result = add_normalize2_result
        log.debug("result shape %s", result.shape)
        return result
