from typing import Any, Literal
import numpy as np
import torch
from tianshou.data import to_torch
from .embedding import Embedding
from utils.gradient_visualizer import render_gradient
from .transformer_net import TransformerNet
from .convert_input_to_torch import convert_input_to_torch
from .index_and_gather_nonzero_items_along_axis import create_zero_rows_mask

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

        # print('TWO LEVEL INPUT', input.shape)

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
            # print('ego_peer_input', ego_peer_input.shape)

            zero_item_mask = self._create_zero_item_mask(
                ego_peer_input, zero_value=self._pad_value)
            ego_peer_sequence_output, _ = self._ego_sequence_transformer(
                ego_peer_input)

            ego_peer_output = torch.zeros_like(ego_peer_sequence_output)
            ego_peer_output[~zero_item_mask] = ego_peer_sequence_output[
                ~zero_item_mask]

        else:

            if self._mode == 'level1_different_params':

                # print('ego_input', ego_input, ego_input.shape)
                # print('peer_input', peer_input, peer_input.shape)

                # call the ego transformer
                ego_sequence_result, _ = self._ego_sequence_transformer(
                    ego_input)
                #print('EGO RESULT', ego_sequence_result, ego_sequence_result.shape)

                # call the peer transformer
                peer_zero_item_mask = self._create_zero_item_mask(
                    peer_input, zero_value=self._pad_value)

                # print('peer_zero_item_mask', peer_zero_item_mask, peer_zero_item_mask.shape)

                # print('CALL PEER', peer_input)
                peer_sequence_output, _ = self._peer_sequence_transformer(
                    peer_input)
                #print('PEER RESULT', peer_sequence_output, peer_sequence_output.shape)

                # use only the non masked values from the peer
                masked_peer_output = torch.zeros_like(peer_sequence_output)
                masked_peer_output[
                    ~peer_zero_item_mask] = peer_sequence_output[
                        ~peer_zero_item_mask]

                # print('zero', (~peer_zero_item_mask)[0])

                # print('PEER SEQ OUT', peer_sequence_output[~peer_zero_item_mask], peer_sequence_output[~peer_zero_item_mask].shape)

                # print('peer_sequence_output', peer_sequence_output)
                # print('masked_peer_output', masked_peer_output)

                # stack with the ego_output over the item axis
                ego_peer_output = torch.cat(
                    (ego_sequence_result, masked_peer_output), dim=-2)

            elif self._mode == 'level1_same_params':

                zero_item_mask = self._create_zero_item_mask(
                    input, zero_value=self._pad_value)

                # print('peer_zero_item_mask', peer_zero_item_mask, peer_zero_item_mask.shape)

                # print('CALL PEER', peer_input)
                sequence_output, _ = self._ego_sequence_transformer(input)
                #print('PEER RESULT', peer_sequence_output, peer_sequence_output.shape)

                # use only the non masked values
                masked_output = torch.zeros_like(sequence_output)
                masked_output[~zero_item_mask] = sequence_output[
                    ~zero_item_mask]

                # print('PEER SEQ OUT', peer_sequence_output[~peer_zero_item_mask], peer_sequence_output[~peer_zero_item_mask].shape)

                # print('peer_sequence_output', peer_sequence_output)
                # print('masked_peer_output', masked_peer_output)

                # stack with the ego_output over the item axis
                ego_peer_output = masked_output

            else:
                raise ValueError(self._mode)

        # print('input', input.shape, ego_peer_output.shape, ego_peer_output)
        # tmp_mask = self._create_zero_item_mask(ego_peer_output, zero_value=self._pad_value)
        # print('TMP MASK', tmp_mask.shape, tmp_mask)

        #print('ego_peer_output', ego_peer_output, ego_peer_output.shape)

        # feed the it to the level 2 transformer
        # level2_item_mask = self._create_zero_item_mask(
        #     ego_peer_output, zero_value=zero_value)

        # print('EGO_PEER_OUTPUT', ego_peer_output)

        # print('calling level2', ego_peer_output.shape)
        level2_output, _ = self._item_transformer(ego_peer_output)

        # # HACK: REMOVE THIS
        # level2_output = ego_peer_output[..., 0, :]
        # print('level2', level2_output.shape)
        # print('level2_output', level2_output)

        # print('level2_output', level2_output, level2_output.shape)

        # # call the peer transformer
        # self._peer_sequence_transformer(peer_input)

        # print('peer_mask', peer_zero_item_mask, peer_zero_item_mask.shape)

        # if torch.any(torch.isnan(level2_output)):
        #     raise ValueError(
        #         'NaN found in level 2 transformer output.'
        #     )

        # assert isinstance(level2_output, torch.Tensor)
        # if level2_output.requires_grad:
        #     render_gradient(level2_output, dict(self.named_parameters()),
        #                     filename='graph.png')
        #     raise Exception('stopit')

        return level2_output, None

    def _create_zero_item_mask(self, input: torch.Tensor, zero_value: float):
        # filter fully zero sequence items
        zero_row_mask = create_zero_rows_mask(input,
                                              zero_value=zero_value,
                                              dim=-1)

        # print('zero_row_mask', zero_row_mask, zero_row_mask.shape)

        # create mask for the whole items
        zero_item_mask = torch.all(zero_row_mask, dim=-1)

        return zero_item_mask


class MultiLevelTransformerNet(torch.nn.Module):

    _device: torch.device | None
    _transformers: torch.nn.ModuleList

    def __init__(self,
                 transformers: list[TransformerNet],
                 device: torch.device | None = None):

        super().__init__()

        self._transformers = torch.nn.ModuleList(transformers)
        self._device = device

    def forward(self,
                input: torch.Tensor | np.ndarray,
                state: torch.Tensor | None = None,
                info: dict[str, Any] = {}):

        current_input = convert_input_to_torch(input, device=self._device)

        if torch.any(torch.isnan(current_input)):
            raise ValueError('NaN found in multi-level transformer input.')

        for transformer in self._transformers:

            zero_row_mask = create_zero_rows_mask(current_input,
                                                  zero_value=0.,
                                                  dim=-1)
            zero_item_mask = torch.all(zero_row_mask, dim=-1)

            output, _ = transformer(current_input)

            # use only the items which are nonzero
            masked_output = torch.zeros_like(output)
            masked_output[~zero_item_mask] = output[~zero_item_mask]

            current_input = masked_output

            if torch.any(torch.isnan(current_input)):
                raise ValueError(
                    'NaN found in multi-level transformer intermediate output.'
                )

        return current_input, None
