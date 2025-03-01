from dataclasses import asdict
from model.transformer.multi_level_transformer_net import MultiLevelTransformerNet
import numpy as np
from thermal.zero import zero_air_velocity_field
import torch

from trainer.multi_agent.tests.create_sample_items import create_and_stack_sample_items, create_sample_items
from trainer.multi_agent.tests.test_multi_agent_policy import create_agent_trajectory

from model.transformer.transformer_net import TransformerNet, TransformerNetParameters, create_zero_rows_mask

# def test_multi_level_transformer


def test_sandbox():

    batch_size = 2
    sequence_length = 2
    dim = 2

    obs = create_and_stack_sample_items(shape=(batch_size, sequence_length,
                                               dim),
                                        item_axis=1,
                                        count=4)

    # batch x item x sequence x dim
    obs[0, 0, 1] = np.nan
    obs[0, 1, 0] = np.nan
    obs[0, 2, :] = np.nan  # clear the whole item

    obs[1, 0, 1] = np.nan
    obs[1, 2, 0] = np.nan
    obs[1, 1, :] = np.nan  # clear the whole item

    obs = np.nan_to_num(obs, nan=0.)

    obs = torch.Tensor(obs)

    # create mask for all zero inputs
    zero_row_mask = create_zero_rows_mask(obs, zero_value=0., dim=-1)
    zero_item_mask = torch.all(zero_row_mask, dim=-1)
    print('obs', obs, obs.shape)
    print('zero_row_mask', zero_row_mask, zero_row_mask.shape)
    print('zero_item_mask', zero_item_mask, zero_item_mask.shape)

    level1_params = _create_transformer_net_params(input_dim=2,
                                                   encoder_layer_count=1,
                                                   attention_head_num=1,
                                                   attention_internal_dim=2,
                                                   output_dim=3)
    transformer_net = _create_transformer_net(level1_params)

    output, _ = transformer_net(obs)
    print('output', output, output.shape)

    # clear the zero items output
    # print('')

    masked_output = torch.zeros_like(output)
    masked_output[~zero_item_mask] = output[~zero_item_mask]

    output[zero_item_mask] = 0.
    print('cleared output', output)
    print('masked_output', masked_output)

    # obs = obs[:, :2, ...]
    # print('obs_cleared', obs, obs.shape)
    # output, _ = transformer_net(obs)
    # print('output', output, output.shape)

    level2_params = _create_transformer_net_params(input_dim=3,
                                                   encoder_layer_count=1,
                                                   attention_head_num=1,
                                                   attention_internal_dim=2,
                                                   output_dim=2)
    item_transformer_net = _create_transformer_net(level2_params)

    # execute again the transformer on the resulting items
    item_output, _ = item_transformer_net(output)
    print('item_output', item_output, item_output.shape)

    multilevel_transformer = MultiLevelTransformerNet(
        [transformer_net, item_transformer_net])

    multilevel_output, _ = multilevel_transformer(obs)
    print('multi out', multilevel_output, multilevel_output.shape)

    return

    print(obs)
    print(obs.shape)

    tensor = torch.tensor(obs)

    # first step, remove fully empty items
    mask = torch.isnan(tensor)

    print('mask', mask, mask.shape)

    mask = mask.all(dim=(1, 3), keepdim=True)
    print('dim mask', mask, mask.shape)

    # create positional encoding
    position = torch.arange(sequence_length).unsqueeze(1)

    pe = torch.zeros(sequence_length, dim)
    pe[:, 0::2] = 1000 + position * 100
    pe[:, 1::2] = 2000 + position * 100
    print('pe', pe, pe.shape)

    # unsqueeze pe tensor along the item axis
    pe = pe.unsqueeze(-2)
    print('pe unsqueezed', pe, pe.shape)

    print('tensor', tensor, tensor.shape)

    # add positional encoding
    added = tensor + pe

    print('added', added)


def _create_transformer_net_params(input_dim: int,
                                   encoder_layer_count=1,
                                   attention_head_num=1,
                                   attention_internal_dim=2,
                                   output_dim=3,
                                   pad_value=0.):
    params = TransformerNetParameters(
        input_dim=input_dim,
        output_dim=output_dim,
        attention_internal_dim=attention_internal_dim,
        attention_head_num=attention_head_num,
        ffnn_hidden_dim=4,
        ffnn_dropout_rate=0.0,
        max_sequence_length=100,
        embedding_dim=4,
        encoder_layer_count=encoder_layer_count,
        enable_layer_normalization=False,
        enable_causal_attention_mask=True,
        is_reversed_sequence=True,
        softmax_output=False,
        pad_value=pad_value)
    return params


def _create_transformer_net(params: TransformerNetParameters):

    transformer_net = TransformerNet(**asdict(params))

    return transformer_net
