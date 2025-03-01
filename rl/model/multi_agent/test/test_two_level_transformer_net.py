import numpy as np
import torch
from model.transformer.multi_level_transformer_net import TwoLevelTransformerNet
from pytest_mock import MockerFixture

from .test_multi_agent_transformer_net import _create_transformer_net, _create_transformer_net_params
from model.multi_agent.test.test_multi_agent_transformer_net import TransformerNet, create_and_stack_sample_items


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
    # print('obs', obs, obs.shape)

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
    print('expected_level2', expected)
    print('actual', actual)
    assert isinstance(actual, torch.Tensor)
    assert expected.shape == actual.shape
    assert torch.allclose(expected, actual)

    print('output', output)

    # print('actual', actual)
    # print('expected', expected)

    # print(peer_sequence_transformer.call_args)

    # print(result)


def test_should_integrated(mocker: MockerFixture):

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

    expected = torch.randn(2, 5)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output, expected)
    print('loss', loss)

    loss.backward()

    grads1 = [param.grad.mean() for param in model.parameters()]
    print('grads1', grads1)

    print('output', output)

    # then
    assert output.shape == (2, 5)

    print(output, output.shape)

    print(obs, obs.shape)

    zero_rows_mask = np.logical_not(np.all(obs == 0., axis=-1))

    print('mask', zero_rows_mask)

    # print('tmp', np.stack((obs[1,0], obs[1,2]), axis=0))

    obs2 = np.stack((obs[0, 0:2], np.stack((obs[1, 0], obs[1, 2]), axis=0)),
                    axis=0)
    # obs2 = obs

    # obs2 = obs[0]
    print('obs2', obs2, obs2.shape)

    output2, _ = model(obs2)

    print('output2', output2)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output2, expected)
    print('loss', loss)

    model.zero_grad()
    loss.backward()

    grads2 = [param.grad.mean() for param in model.parameters()]

    print('grads1', torch.tensor(grads1).mean(), torch.tensor(grads1).std())
    print('grads1', torch.tensor(grads2).mean(), torch.tensor(grads2).std())


def test_sandbox():

    w_orig = torch.randn(5, 3)
    b_orig = torch.randn(3)

    y_orig = torch.randn(1, 3)

    x = torch.ones(1, 5)  # input tensor
    y = y_orig  # expected output

    print('input', x)
    print('y', y)

    w = torch.tensor(w_orig, requires_grad=True)
    b = torch.tensor(b_orig, requires_grad=True)
    z = torch.matmul(x, w) + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

    loss.backward()

    print(w.grad)
    print(b.grad)

    # model 2
    # x = torch.ones(2, 5) # input tensor
    x = torch.cat((torch.zeros((1, 5)), torch.ones(1, 5)))
    y = torch.cat((torch.zeros((1, 3)), y_orig))  # expected output

    print('input', x)
    print('y', y)

    w = torch.tensor(w_orig, requires_grad=True)
    b = torch.tensor(b_orig, requires_grad=True)
    z = (torch.matmul(x, w) + b)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z[1], y[1])
    loss.backward()

    print('---model2')
    print(w.grad)
    print(b.grad)
