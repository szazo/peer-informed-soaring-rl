from typing import Callable
import logging
from .index_and_gather_nonzero_items_along_axis import create_zero_rows_mask
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class MultiheadAttention2(torch.nn.Module):

    # number of heads
    _head_num: int

    # this will be the dimension used for key and query and value,
    # these are trainable low dimensional representations
    _internal_dim: int

    # the dimension of the source sequence items
    _source_dim: int

    # the dimension of the target sequence items
    _target_dim: int

    # used for the normalization
    _normalizer: float

    # (source_dim, internal_dim) shape
    query_head_weights: nn.ParameterList

    # (target_dim, internal_dim) shape
    key_head_weights: nn.ParameterList

    # (target_dim, internal_dim) shape
    value_head_weights: nn.ParameterList

    # (internal_dim * head_num, source_dim) shape
    output_head_weights: torch.Tensor

    # custom initializer variable for (query,key,value,output) params
    _custom_parameter_init_: None | Callable[
        [nn.ParameterList, nn.ParameterList, nn.ParameterList, torch.Tensor],
        None]

    def __init__(
        self,
        head_num: int,
        internal_dim: int,
        source_dim: int,
        target_dim: int | None,
        custom_parameter_init_: None | Callable[[
            nn.ParameterList, nn.ParameterList, nn.ParameterList, torch.Tensor
        ], None] = None):
        super().__init__()

        self._log = logging.getLogger(__class__.__name__)

        self._head_num = head_num

        # this will be the dimension used for key and query and value
        self._internal_dim = internal_dim
        self._source_dim = source_dim
        self._target_dim = target_dim if target_dim is not None else source_dim

        self._custom_parameter_init_ = custom_parameter_init_

        # used for normalizations
        self._normalizer = 1 / np.sqrt(self._internal_dim)

        # create the parameters for the mapping that produces queries based on input
        self.query_head_weights = self._create_empty_parameter_list(
            (self._source_dim, self._internal_dim))

        # create the parameters for the mapping that produces keys and values based on the target
        self.key_head_weights = self._create_empty_parameter_list(
            (self._target_dim, self._internal_dim))
        self.value_head_weights = self._create_empty_parameter_list(
            (self._target_dim, self._internal_dim))

        # create the parameters which will map the head_num number of z with internal dim to the same as the input
        self.output_head_weights = nn.Parameter(
            torch.empty(
                ((self._internal_dim * self._head_num, self._source_dim))))

        self._reset_parameters()

    def _create_empty_parameter_list(self, shape: tuple[int, int]):
        return nn.ParameterList(
            [nn.Parameter(torch.empty(shape)) for _ in range(self._head_num)])

    def _reset_parameters(self):

        if self._custom_parameter_init_ is not None:
            self._custom_parameter_init_(self.query_head_weights,
                                         self.key_head_weights,
                                         self.value_head_weights,
                                         self.output_head_weights)
        else:
            for i in range(self._head_num):
                nn.init.xavier_uniform_(self.query_head_weights[i])
                nn.init.xavier_uniform_(self.key_head_weights[i])
                nn.init.xavier_uniform_(self.value_head_weights[i])
            nn.init.xavier_uniform_(self.output_head_weights)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        assert source.shape[
            -1] == self._source_dim, 'last dimension should match source_dim'

        log = self._log

        log.debug("source shape %s", source.shape)

        # the target is optional, it means the content which we want to attend on
        # it it is None, then it will be the same as source, it will be the self-attention
        if target is None:
            target = source

        head_output_list = []

        for i in range(self._head_num):

            query = source @ self.query_head_weights[i]
            key = target @ self.key_head_weights[i]
            value = target @ self.value_head_weights[i]

            dot = query @ key.transpose(-2, -1)

            normalized = dot * self._normalizer

            if attention_mask is not None:
                normalized += attention_mask

                # create a mask which only contains the nonzero sequence rows
                softmax_mask = ~create_zero_rows_mask(
                    attention_mask, zero_value=-torch.inf, dim=-1)

                # calculate softmax for only these rows
                masked_scores = F.softmax(normalized[softmax_mask], dim=-1)

                # create the scores with zeros and fill only values from mask
                scores = torch.zeros_like(normalized)
                scores[softmax_mask] = masked_scores
            else:
                # calculate softmax for each
                scores = F.softmax(normalized, dim=-1)

            weighted_sum = scores @ value

            head_output_list.append(weighted_sum)

        head_outputs_concated = torch.concat(head_output_list, dim=-1)
        output_new = head_outputs_concated @ self.output_head_weights

        return output_new
