import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
from .convert_input_to_torch import convert_input_to_torch
import numpy as np
import torch
from .transformer import Transformer
from .create_causal_attention_mask import create_causal_attention_mask
from .positional_encoding import PositionalEncoding
from .embedding import Embedding
from .index_and_gather_nonzero_items_along_axis import (
    collect_nonzero_indices_along_axis, create_zero_rows_mask,
    gather_based_on_indices_along_axis)


@dataclass
class TransformerNetParameters:
    input_dim: int
    output_dim: int
    attention_internal_dim: int
    attention_head_num: int
    ffnn_hidden_dim: int
    ffnn_dropout_rate: float
    max_sequence_length: int
    embedding_dim: int | None
    encoder_layer_count: int
    enable_layer_normalization: bool
    enable_causal_attention_mask: bool
    # the sequence is reversed when last time step is the first item in the sequence
    is_reversed_sequence: bool = False
    # should be false, because Actor class will add the softmax at the end
    softmax_output: bool = False
    pad_value: float = 0.
    encoder_activation: str = 'swish'  # relu | swish


class TransformerNet(torch.nn.Module):

    _pad_value: float
    """The value which was used for padding zero sequence_items"""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 attention_internal_dim: int,
                 attention_head_num: int,
                 ffnn_hidden_dim: int,
                 ffnn_dropout_rate: float,
                 max_sequence_length: int,
                 softmax_output: bool,
                 embedding_dim: Union[int, None],
                 encoder_layer_count: int,
                 enable_layer_normalization: bool,
                 enable_causal_attention_mask: bool,
                 is_reversed_sequence: bool,
                 device: torch.device | None = None,
                 pad_value: float = 0.,
                 encoder_activation: str = 'swish'):
        super().__init__()

        self._device = device
        self._pad_value = pad_value

        assert pad_value == 0., 'currently if the pad is nonzero e.g. nan, the output will contain nans.'

        self._log = logging.getLogger(__class__.__name__)
        log = self._log

        if embedding_dim is not None:
            self._embedding = Embedding(in_features=input_dim,
                                        out_features=embedding_dim,
                                        layer_count=encoder_layer_count,
                                        device=device)
            self._embed_dim = embedding_dim
        else:
            self._embedding = None
            self._embed_dim = input_dim

        log.debug("input_dim %s", input_dim)
        log.debug("output_dim %s", output_dim)
        log.debug("attention_internal_dim %s", attention_internal_dim)

        self._output_dim = output_dim
        self._enable_causal_attention_mask = enable_causal_attention_mask
        self._is_reversed_sequence = is_reversed_sequence

        self._positional_encoding = PositionalEncoding(
            dimension=self._embed_dim,
            max_sequence_length=max_sequence_length,
            device=device)

        assert encoder_activation == 'relu' or encoder_activation == 'swish'

        self._transformer = Transformer(
            encoder_embed_dim=self._embed_dim,
            output_dim=output_dim,
            attention_internal_dim=attention_internal_dim,
            attention_head_num=attention_head_num,
            ffnn_hidden_dim=ffnn_hidden_dim,
            ffnn_dropout_rate=ffnn_dropout_rate,
            softmax_output=softmax_output,
            encoder_layer_count=encoder_layer_count,
            enable_layer_normalization=enable_layer_normalization,
            encoder_activation=encoder_activation).to(device)

    def _create_nonpad_attention_mask(self, input: torch.Tensor):
        """Creates an attention mask which skip items which are fully pad_value on the last dimension"""

        # create mask for all pad items
        pad_seq_items_mask = create_zero_rows_mask(input,
                                                   zero_value=self._pad_value,
                                                   dim=-1)

        # expand to the last dimension
        pad_seq_items_mask = pad_seq_items_mask.unsqueeze(-1)
        pad_seq_items_mask = pad_seq_items_mask.expand(
            *pad_seq_items_mask.shape[:-1], pad_seq_items_mask.size(-2))
        pad_seq_items_mask_transpose = pad_seq_items_mask.transpose(
            -2, -1)  # transpose the last two dimensions

        nonpad_attention_mask = torch.zeros(pad_seq_items_mask.shape,
                                            device=self._device)
        nonpad_attention_mask[pad_seq_items_mask] = -torch.inf
        nonpad_attention_mask[pad_seq_items_mask_transpose] = -torch.inf

        return nonpad_attention_mask

    def forward(self,
                input: torch.Tensor | np.ndarray,
                state: Optional[torch.Tensor] = None,
                info: Dict[str, Any] = {}):
        assert isinstance(input, torch.Tensor) or isinstance(input, np.ndarray)

        log = self._log
        input_sequence_length = input.shape[-2]
        log.debug("input %s %s %s; length: %s", input, input.shape,
                  input.dtype, input_sequence_length)

        input = convert_input_to_torch(input, device=self._device)

        log.debug("input as torch tensor %s %s", input.shape, input.dtype)

        if torch.any(torch.isnan(input)):
            # temporary sanity checking
            raise ValueError('NaN found in transformer input.')

        nonpad_attention_mask = self._create_nonpad_attention_mask(input)

        # find first and last nonzero index for each sequence in the batch
        first_nonzero_indices, last_nonzero_indices = collect_nonzero_indices_along_axis(
            input, zero_value=self._pad_value, dim=-1)

        if self._embedding is not None:
            input = self._embedding(input)

        input = self._positional_encoding(input)

        log.debug("input after positional encoding %s %s %s", input,
                  input.shape, input.dtype)
        log.debug("encoding...")

        current_decoded_sequence = torch.zeros((1, self._embed_dim),
                                               device=self._device)

        mask = nonpad_attention_mask
        if self._enable_causal_attention_mask:
            mask += create_causal_attention_mask(
                sequence_length=input_sequence_length,
                reverse=self._is_reversed_sequence,
                device=self._device)

        transformer_output = self._transformer(
            input_sequence=input,
            decoded_sequence=current_decoded_sequence,
            input_attention_mask=mask)

        log.debug("transformer_output %s %s", transformer_output,
                  transformer_output.shape)

        if self._is_reversed_sequence:
            # if it is a reversed sequence, use the first nonpad item as output
            indices = first_nonzero_indices
        else:
            # use the last item of each sequence of the batch
            indices = last_nonzero_indices

        gathered = gather_based_on_indices_along_axis(transformer_output,
                                                      indices)

        result = gathered.squeeze(dim=-2)
        log.debug("result %s %s", result, result.shape)

        return result, state
