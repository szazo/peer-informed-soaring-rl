from typing import Any, Literal
import numpy as np
import torch
from .embedding import Embedding
from .transformer_net import TransformerNet
from .convert_input_to_torch import convert_input_to_torch
from .index_and_gather_nonzero_items_along_axis import create_zero_rows_mask

# REVIEW: remove these "compiler time" parameters, not used
EMBEDDING = False
SEPARATE_EMBEDDING_COMMON_ENCODER = False


class TwoLevelTransformerNet(torch.nn.Module):

    _device: torch.device | None

    _embedding: Embedding

    _ego_embedding: Embedding
    _peer_embedding: Embedding

    _ego_sequence_transformer: TransformerNet
    _peer_sequence_transformer: TransformerNet
    _item_transformer: TransformerNet

    _pad_value: float

    _mode: Literal['level1_same_params'] | Literal['level1_different_params']
    _clear_ego_sequence_max_length: None | int

    def __init__(
            self,
            ego_sequence_transformer: TransformerNet,
            peer_sequence_transformer: TransformerNet,
            item_transformer: TransformerNet,
            device: torch.device | None = None,
            pad_value: float = 0.,
            mode: Literal['level1_same_params']
        | Literal['level1_different_params'] = 'level1_different_params',
            clear_ego_sequence_max_length: None | int = None):

        super().__init__()

        self._pad_value = pad_value
        self._mode = mode
        self._clear_ego_sequence_max_length = clear_ego_sequence_max_length

        if EMBEDDING:
            self._embedding = Embedding(in_features=7,
                                        out_features=64,
                                        layer_count=3,
                                        device=device)

        if SEPARATE_EMBEDDING_COMMON_ENCODER:
            self._ego_embedding = Embedding(in_features=7,
                                            out_features=64,
                                            layer_count=3,
                                            device=device)
            self._peer_embedding = Embedding(in_features=7,
                                             out_features=64,
                                             layer_count=3,
                                             device=device)

        self._ego_sequence_transformer = ego_sequence_transformer
        self._peer_sequence_transformer = peer_sequence_transformer
        self._item_transformer = item_transformer

        self._device = device

    def forward(self,
                input: torch.Tensor | np.ndarray,
                state: torch.Tensor | None = None,
                info: dict[str, Any] = {}):

        input = convert_input_to_torch(input, device=self._device)

        if EMBEDDING:
            input = self._embedding(input)
            assert isinstance(input, torch.Tensor)

        if self._clear_ego_sequence_max_length is not None:
            input = torch.tensor(input)
            input[..., 0, self._clear_ego_sequence_max_length:, :] = 0.

        # separate the sequence to ego and non-ego sequences
        # eqo sequences should not be full empty

        # input: batch x item x sequence x dim
        ego_input = input[..., 0:1, :, :]
        peer_input = input[..., 1:, :, :]

        if SEPARATE_EMBEDDING_COMMON_ENCODER:
            ego_input = self._ego_embedding(ego_input)
            peer_input = self._peer_embedding(peer_input)
            ego_peer_input = torch.cat((ego_input, peer_input), dim=-3)

            zero_item_mask = self._create_zero_item_mask(
                ego_peer_input, zero_value=self._pad_value)
            ego_peer_sequence_output, _ = self._ego_sequence_transformer(
                ego_peer_input)

            ego_peer_output = torch.zeros_like(ego_peer_sequence_output)
            ego_peer_output[~zero_item_mask] = ego_peer_sequence_output[
                ~zero_item_mask]

        else:

            if self._mode == 'level1_different_params':

                # call the ego transformer
                ego_sequence_result, _ = self._ego_sequence_transformer(
                    ego_input)

                # call the peer transformer
                peer_zero_item_mask = self._create_zero_item_mask(
                    peer_input, zero_value=self._pad_value)

                peer_sequence_output, _ = self._peer_sequence_transformer(
                    peer_input)

                # use only the non masked values from the peer
                masked_peer_output = torch.zeros_like(peer_sequence_output)
                masked_peer_output[
                    ~peer_zero_item_mask] = peer_sequence_output[
                        ~peer_zero_item_mask]

                # stack with the ego_output over the item axis
                ego_peer_output = torch.cat(
                    (ego_sequence_result, masked_peer_output), dim=-2)

            elif self._mode == 'level1_same_params':

                zero_item_mask = self._create_zero_item_mask(
                    input, zero_value=self._pad_value)

                sequence_output, _ = self._ego_sequence_transformer(input)

                # use only the non masked values
                masked_output = torch.zeros_like(sequence_output)
                masked_output[~zero_item_mask] = sequence_output[
                    ~zero_item_mask]

                # stack with the ego_output over the item axis
                ego_peer_output = masked_output

            else:
                raise ValueError(self._mode)

        level2_output, _ = self._item_transformer(ego_peer_output)

        return level2_output, None

    def _create_zero_item_mask(self, input: torch.Tensor, zero_value: float):
        # filter fully zero sequence items
        zero_row_mask = create_zero_rows_mask(input,
                                              zero_value=zero_value,
                                              dim=-1)

        # create mask for the whole items
        zero_item_mask = torch.all(zero_row_mask, dim=-1)

        return zero_item_mask
