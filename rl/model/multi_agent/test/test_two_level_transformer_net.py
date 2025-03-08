from dataclasses import asdict
import numpy as np
import torch
from model.transformer.multi_level_transformer_net import TwoLevelTransformerNet
from pytest_mock import MockerFixture

from trainer.multi_agent.tests.create_sample_items import create_and_stack_sample_items
from model.transformer.transformer_net import (TransformerNet,
                                               TransformerNetParameters)


def _create_obs():
    batch_size = 2
    item_count = 3
    sequence_length = 2
    dim = 4

    obs = create_and_stack_sample_items(shape=(batch_size, sequence_length,
                                               dim),
                                        item_axis=1,
                                        count=item_count)

    # clear [batch=0,item=0,sequence=1]
    obs[0, 0, 1, :] = np.nan

    # clear [batch=0,item=1,sequence=0]
    obs[0, 1, 0, :] = np.nan

    # clear [batch=0,item=2,sequence=all]
    obs[0, 2, :, :] = np.nan

    # clear [batch=1,item=0,sequence=1]
    obs[1, 0, 1, :] = np.nan

    # clear [batch=1,item=2,sequence=1]
    obs[1, 2, 1, :] = np.nan

    # clear [batch=1,item=1,sequence=all]
    obs[1, 1, :, :] = np.nan

    # convert nan to zeros
    obs = np.nan_to_num(obs, nan=0.)

    return obs


def test_should_call_different_transformer_for_the_first_item(
        mocker: MockerFixture):

    obs = _create_obs()

    ego_sequence_transformer = mocker.create_autospec(TransformerNet,
                                                      instance=True)
    peer_sequence_transformer = mocker.create_autospec(TransformerNet,
                                                       instance=True)
    item_transformer = mocker.create_autospec(TransformerNet, instance=True)

    model = TwoLevelTransformerNet(
        ego_sequence_transformer=ego_sequence_transformer,
        peer_sequence_transformer=peer_sequence_transformer,
        item_transformer=item_transformer)

    # ego result for the two batch
    ego_transformer_result = torch.tensor(
        np.array([[[1., 2., 3.]], [[4., 5., 6.]]]))
    ego_sequence_transformer.return_value = (ego_transformer_result, None)

    # peer result for the two batch
    peer_transformer_result = torch.tensor(
        np.array([[[7., 8., 9.], [10., 11., 12.]],
                  [[13., 14., 15.], [16., 17., 18.]]]))
    peer_sequence_transformer.return_value = (peer_transformer_result, None)

    # level 2 transformer output (two vectors for the two batches)
    level2_transformer_output = torch.tensor(np.array([[21., 22.], [23.,
                                                                    24.]]))
    item_transformer.return_value = (level2_transformer_output, None)

    # when
    output = model.forward(obs)

    # then

    # assert ego transformer called with the ego trajectories
    ego_sequence_transformer.assert_called_once()

    expected = torch.tensor(np.stack((obs[0, 0:1, :, :], obs[1, 0:1, :, :]),
                                     axis=0),
                            dtype=torch.get_default_dtype())
    actual = ego_sequence_transformer.call_args[0][0]

    assert isinstance(actual, torch.Tensor)
    assert expected.shape == actual.shape
    assert torch.allclose(expected, actual)

    # assert peer transformer called with the remaining trajectories
    peer_sequence_transformer.assert_called_once()

    expected = torch.tensor(np.stack((obs[0, 1:, :, :], obs[1, 1:, :, :]),
                                     axis=0),
                            dtype=torch.get_default_dtype())
    actual = peer_sequence_transformer.call_args[0][0]

    assert isinstance(actual, torch.Tensor)
    assert expected.shape == actual.shape
    assert torch.allclose(expected, actual)

    # assert level 2 transformer called with zero masked empty items
    expected = torch.cat((ego_transformer_result, peer_transformer_result),
                         dim=-2)

    # clear [batch=0,item=2,sequence=all]
    expected[0, 2, :] = 0.
    expected[1, 1, :] = 0.
    actual = item_transformer.call_args[0][0]
    assert isinstance(actual, torch.Tensor)
    assert expected.shape == actual.shape
    assert torch.allclose(expected, actual)


def test_should_integrate(mocker: MockerFixture):

    torch.manual_seed(42)

    # given
    obs = _create_obs()

    level1_params = _create_transformer_net_params(input_dim=4,
                                                   encoder_layer_count=1,
                                                   attention_head_num=1,
                                                   attention_internal_dim=2,
                                                   output_dim=3)
    ego_sequence_transformer = _create_transformer_net(level1_params)
    peer_sequence_transformer = _create_transformer_net(level1_params)

    level2_params = _create_transformer_net_params(input_dim=3,
                                                   encoder_layer_count=1,
                                                   attention_head_num=1,
                                                   attention_internal_dim=2,
                                                   output_dim=5)
    item_transformer = _create_transformer_net(level2_params)

    model = TwoLevelTransformerNet(
        ego_sequence_transformer=ego_sequence_transformer,
        peer_sequence_transformer=peer_sequence_transformer,
        item_transformer=item_transformer)

    output, _ = model(obs)

    # then
    assert output.shape == (2, 5)


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
