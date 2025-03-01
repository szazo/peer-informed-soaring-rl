from model.transformer.multihead_attention2 import MultiheadAttention2
from model.transformer.create_causal_attention_mask import create_causal_attention_mask
import torch
import numpy as np
import scipy
from functools import partial


def test_init_should_initialize_parameters():

    # given
    head_num = 2
    source_dim = 3
    target_dim = 4
    internal_dim = 2

    # when
    attention = MultiheadAttention2(head_num=head_num,
                                    internal_dim=internal_dim,
                                    source_dim=source_dim,
                                    target_dim=target_dim)

    # then
    default_init_min_max = (-10., 10.)

    for i in range(head_num):

        # query
        assert attention.query_head_weights[i].shape == (source_dim,
                                                         internal_dim)
        assert torch.all(
            (attention.query_head_weights[i] >= default_init_min_max[0])
            & (attention.query_head_weights[i] <= default_init_min_max[1]))

        # key
        assert attention.key_head_weights[i].shape == (target_dim,
                                                       internal_dim)
        assert torch.all(
            (attention.key_head_weights[i] >= default_init_min_max[0])
            & (attention.key_head_weights[i] <= default_init_min_max[1]))

        # value
        assert attention.value_head_weights[i].shape == (target_dim,
                                                         internal_dim)
        assert torch.all(
            (attention.value_head_weights[i] >= default_init_min_max[0])
            & (attention.value_head_weights[i] <= default_init_min_max[1]))

    # output
    assert attention.output_head_weights.shape == (internal_dim * head_num,
                                                   source_dim)
    assert torch.all(
        (attention.output_head_weights >= default_init_min_max[0])
        & (attention.output_head_weights <= default_init_min_max[1]))


def test_init_should_initialize_using_custom_initializer_when_set():

    # given
    head_num = 2
    source_dim = 3
    target_dim = 4
    internal_dim = 2

    def custom_param_init_(query: torch.nn.ParameterList,
                           key: torch.nn.ParameterList,
                           value: torch.nn.ParameterList,
                           output: torch.Tensor):

        for i in range(head_num):
            assert query[i].shape == (source_dim, internal_dim)
            assert key[i].shape == (target_dim, internal_dim)
            assert value[i].shape == (target_dim, internal_dim)

            torch.nn.init.eye_(query[i])
            torch.nn.init.eye_(key[i])
            torch.nn.init.eye_(value[i])

        assert output.shape == (internal_dim * head_num, source_dim)
        torch.nn.init.eye_(output)

    # when
    attention = MultiheadAttention2(head_num=head_num,
                                    internal_dim=internal_dim,
                                    source_dim=source_dim,
                                    target_dim=target_dim,
                                    custom_parameter_init_=custom_param_init_)

    # then
    for i in range(head_num):
        assert torch.allclose(attention.query_head_weights[i],
                              torch.eye(source_dim, internal_dim))
        assert torch.allclose(attention.key_head_weights[i],
                              torch.eye(target_dim, internal_dim))
        assert torch.allclose(attention.value_head_weights[i],
                              torch.eye(target_dim, internal_dim))
    assert torch.allclose(attention.output_head_weights,
                          torch.eye(internal_dim * head_num, source_dim))


def test_should_calculate_when_single_head_attention():

    # from utils.logging import configure_logger

    # configure_logger(debug_loggers=['MultiheadAttention2'])

    # given
    head_num = 1
    source_dim = 2
    target_dim = 3
    internal_dim = 4

    head_param_init = partial(_custom_range_param_init3_,
                              head_num=head_num,
                              source_dim=source_dim,
                              target_dim=target_dim,
                              internal_dim=internal_dim)
    attention = MultiheadAttention2(head_num=head_num,
                                    internal_dim=internal_dim,
                                    source_dim=source_dim,
                                    target_dim=target_dim,
                                    custom_parameter_init_=head_param_init)

    source = torch.tensor([[1., 2], [3, 4], [5, 6], [7, 8]])
    source_batch = torch.unsqueeze(source, dim=0)

    target = torch.tensor([[1., 2, 3], [4, 5, 6]])
    target_batch = torch.unsqueeze(target, 0)

    expected_output = _reference_calculation(
        source,
        target,
        source_dim=source_dim,
        target_dim=target_dim,
        internal_dim=internal_dim,
    )

    # when
    output = attention.forward(source_batch, target=target_batch)

    # then
    assert np.allclose(output.detach().numpy(), expected_output)


def test_should_sum_when_multi_head_with_same_params():

    # given
    head_num = 2
    source_dim = 2
    target_dim = 3
    internal_dim = 4

    head_param_init = partial(_custom_range_param_init3_,
                              head_num=head_num,
                              source_dim=source_dim,
                              target_dim=target_dim,
                              internal_dim=internal_dim)
    attention = MultiheadAttention2(head_num=head_num,
                                    internal_dim=internal_dim,
                                    source_dim=source_dim,
                                    target_dim=target_dim,
                                    custom_parameter_init_=head_param_init)

    source = torch.tensor([[1., 2], [3, 4], [5, 6], [7, 8]])
    source_batch = torch.unsqueeze(source, dim=0)

    target = torch.tensor([[1., 2, 3], [4, 5, 6]])
    target_batch = torch.unsqueeze(target, 0)

    head_expected_output = _reference_calculation(source,
                                                  target,
                                                  source_dim=source_dim,
                                                  target_dim=target_dim,
                                                  internal_dim=internal_dim)

    # when
    output = attention.forward(source_batch, target=target_batch)

    # then
    assert np.allclose(output.detach().numpy(),
                       head_expected_output + head_expected_output)

    #assert np.allclose(output.detach().numpy(), expected_output)
    # https://stackoverflow.com/questions/76648620/how-do-i-implement-this-attention-layer-in-pytorch
    #source_batched = torch.unsqueeze(source, dim=0)


def test_should_sum_when_multi_head_with_different_params():

    # given
    head_num = 2
    source_dim = 2
    target_dim = 3
    internal_dim = 4

    head_param_init = partial(_custom_range_param_init3_,
                              head_num=head_num,
                              source_dim=source_dim,
                              target_dim=target_dim,
                              internal_dim=internal_dim,
                              head_range_param_starts=[1, 2])
    attention = MultiheadAttention2(head_num=head_num,
                                    internal_dim=internal_dim,
                                    source_dim=source_dim,
                                    target_dim=target_dim,
                                    custom_parameter_init_=head_param_init)

    source = torch.tensor([[1., 2], [3, 4], [5, 6], [7, 8]])
    source_batch = torch.unsqueeze(source, dim=0)

    target = torch.tensor([[1., 2, 3], [4, 5, 6]])
    target_batch = torch.unsqueeze(target, 0)

    head1_expected_output = _reference_calculation(source,
                                                   target,
                                                   source_dim=source_dim,
                                                   target_dim=target_dim,
                                                   internal_dim=internal_dim,
                                                   range_param_start=1)
    head2_expected_output = _reference_calculation(source,
                                                   target,
                                                   source_dim=source_dim,
                                                   target_dim=target_dim,
                                                   internal_dim=internal_dim,
                                                   range_param_start=2)

    # when
    output = attention.forward(source_batch, target=target_batch)

    # then
    assert np.allclose(output.detach().numpy(),
                       head1_expected_output + head2_expected_output)


def test_should_use_attention_for_each_head_when_mask_set():

    # given
    head_num = 2
    source_dim = 2
    target_dim = 3
    internal_dim = 4

    head_param_init = partial(_custom_range_param_init3_,
                              head_num=head_num,
                              source_dim=source_dim,
                              target_dim=target_dim,
                              internal_dim=internal_dim,
                              head_range_param_starts=[1, 2])
    attention = MultiheadAttention2(head_num=head_num,
                                    internal_dim=internal_dim,
                                    source_dim=source_dim,
                                    target_dim=target_dim,
                                    custom_parameter_init_=head_param_init)

    source = torch.tensor([[1., 2], [3, 4], [5, 6], [7, 8]])
    source_batch = torch.unsqueeze(source, dim=0)

    target = torch.tensor([[1., 2, 3], [4, 5, 6]])
    target_batch = torch.unsqueeze(target, 0)

    mask = torch.zeros((source.shape[0], target.shape[0]))
    mask.fill_diagonal_(-torch.inf)
    mask_batch = torch.unsqueeze(mask, 0)

    head1_expected_output = _reference_calculation(source,
                                                   target,
                                                   source_dim=source_dim,
                                                   target_dim=target_dim,
                                                   internal_dim=internal_dim,
                                                   range_param_start=1,
                                                   mask=mask)
    head2_expected_output = _reference_calculation(source,
                                                   target,
                                                   source_dim=source_dim,
                                                   target_dim=target_dim,
                                                   internal_dim=internal_dim,
                                                   range_param_start=2,
                                                   mask=mask)

    # when
    output = attention.forward(source_batch,
                               target=target_batch,
                               attention_mask=mask_batch)

    # then
    assert np.allclose(output.detach().numpy(),
                       head1_expected_output + head2_expected_output)


def test_should_calculate_batched_when_multiple_batches():

    # given
    head_num = 2
    source_dim = 2
    target_dim = 3
    internal_dim = 4

    param_init = partial(_custom_range_param_init2_,
                         head_num=head_num,
                         source_dim=source_dim,
                         target_dim=target_dim,
                         internal_dim=internal_dim,
                         head_range_param_starts=[1, 2])
    head_param_init = partial(_custom_range_param_init3_,
                              head_num=head_num,
                              source_dim=source_dim,
                              target_dim=target_dim,
                              internal_dim=internal_dim,
                              head_range_param_starts=[1, 2])
    attention = MultiheadAttention2(head_num=head_num,
                                    internal_dim=internal_dim,
                                    source_dim=source_dim,
                                    target_dim=target_dim,
                                    custom_parameter_init_=head_param_init)

    source1 = torch.tensor([[1., 2], [3, 4], [5, 6], [7, 8]])
    source2 = torch.tensor([[2., 3], [4, 5], [6, 7], [8, 9]])
    source_batch = torch.stack((source1, source2), dim=0)

    target1 = torch.tensor([[1., 2, 3], [4, 5, 6]])
    target2 = torch.tensor([[2., 3, 4], [5, 6, 7]])
    target_batch = torch.stack((target1, target2), dim=0)

    mask = torch.zeros((source1.shape[0], target1.shape[0]))
    mask.fill_diagonal_(-torch.inf)
    mask_batch = torch.unsqueeze(mask, 0).expand(2, -1, -1)

    # expected output for two items with two heads
    item1_expected_output = [
        _reference_calculation(source1,
                               target1,
                               source_dim=source_dim,
                               target_dim=target_dim,
                               internal_dim=internal_dim,
                               range_param_start=i + 1,
                               mask=mask) for i in range(head_num)
    ]
    item2_expected_output = [
        _reference_calculation(source2,
                               target2,
                               source_dim=source_dim,
                               target_dim=target_dim,
                               internal_dim=internal_dim,
                               range_param_start=i + 1,
                               mask=mask) for i in range(head_num)
    ]

    # when
    output = attention.forward(source_batch,
                               target=target_batch,
                               attention_mask=mask_batch)

    # then
    assert np.allclose(output.detach().numpy()[0],
                       item1_expected_output[0] + item1_expected_output[1])
    assert np.allclose(output.detach().numpy()[1],
                       item2_expected_output[0] + item2_expected_output[1])


def _reference_calculation(source_tensor: torch.Tensor,
                           target_tensor: torch.Tensor,
                           source_dim: int,
                           target_dim: int,
                           internal_dim: int,
                           range_param_start: int = 1,
                           mask: torch.Tensor | None = None):

    source = source_tensor.detach().numpy()
    target = target_tensor.detach().numpy()

    query_weights = _create_range_param_numpy((source_dim, internal_dim),
                                              start_at=range_param_start)
    key_weights = _create_range_param_numpy((target_dim, internal_dim),
                                            start_at=range_param_start)
    value_weights = _create_range_param_numpy((target_dim, internal_dim),
                                              start_at=range_param_start)
    output_weights = _create_range_param_numpy((internal_dim, source_dim),
                                               start_at=range_param_start)

    query = source @ query_weights
    key = target @ key_weights
    value = target @ value_weights

    dot = query @ key.transpose()

    # mask
    if mask is not None:
        dot += mask.detach().numpy()

    normalized = dot / np.sqrt(internal_dim)
    sm = scipy.special.softmax(normalized, axis=1)

    output = sm @ value @ output_weights

    return output


def _custom_range_param_init2_(query: torch.Tensor,
                               key: torch.Tensor,
                               value: torch.Tensor,
                               output: torch.Tensor,
                               head_num: int,
                               source_dim: int,
                               target_dim: int,
                               internal_dim: int,
                               head_range_param_starts: list[int]
                               | None = None):

    query_params = _create_range_param(
        head_num=head_num,
        shape=(source_dim, internal_dim),
        stack_axis=1,
        head_range_param_starts=head_range_param_starts)
    key_params = _create_range_param(
        head_num=head_num,
        shape=(target_dim, internal_dim),
        stack_axis=1,
        head_range_param_starts=head_range_param_starts)
    value_params = _create_range_param(
        head_num=head_num,
        shape=(target_dim, internal_dim),
        stack_axis=1,
        head_range_param_starts=head_range_param_starts)
    output_params = _create_range_param(
        head_num=head_num,
        shape=(internal_dim, source_dim),
        stack_axis=0,
        head_range_param_starts=head_range_param_starts)

    with torch.no_grad():
        assert query.shape == query_params.shape
        assert key.shape == key_params.shape
        assert value.shape == value_params.shape
        assert output.shape == output_params.shape

        query.copy_(torch.Tensor(query_params))
        key.copy_(torch.Tensor(key_params))
        value.copy_(torch.Tensor(value_params))
        output.copy_(torch.Tensor(output_params))


def _custom_range_param_init3_(query: torch.nn.ParameterList,
                               key: torch.nn.ParameterList,
                               value: torch.nn.ParameterList,
                               output: torch.Tensor,
                               head_num: int,
                               source_dim: int,
                               target_dim: int,
                               internal_dim: int,
                               head_range_param_starts: list[int]
                               | None = None):

    query_params = _create_range_param2(
        head_num=head_num,
        shape=(source_dim, internal_dim),
        head_range_param_starts=head_range_param_starts)
    key_params = _create_range_param2(
        head_num=head_num,
        shape=(target_dim, internal_dim),
        head_range_param_starts=head_range_param_starts)
    value_params = _create_range_param2(
        head_num=head_num,
        shape=(target_dim, internal_dim),
        head_range_param_starts=head_range_param_starts)
    output_params = _create_range_param2(
        head_num=head_num,
        shape=(internal_dim, source_dim),
        head_range_param_starts=head_range_param_starts)
    output_params_merged = torch.cat(
        [torch.Tensor(item) for item in output_params], dim=0)

    with torch.no_grad():
        for i in range(head_num):

            assert query[i].shape == query_params[i].shape
            assert key[i].shape == key_params[i].shape
            assert value[i].shape == value_params[i].shape
            # assert output[i].shape == output_params[i].shape
            assert output.shape == output_params_merged.shape

            query[i].copy_(torch.Tensor(query_params[i]))
            key[i].copy_(torch.Tensor(key_params[i]))
            value[i].copy_(torch.Tensor(value_params[i]))
            output.copy_(output_params_merged)
            # output[i].copy_(torch.Tensor(output_params[i]))


def _create_range_param2(head_num: int,
                         shape: tuple[int, int],
                         head_range_param_starts: list[int] | None = None):

    if head_range_param_starts is None:
        # same for each head
        head_param = _create_range_param_numpy(shape)
        assert head_param is not None
        result = [np.copy(head_param) for _ in range(head_num)]

        return result
    else:
        result: list[np.ndarray] = []
        for i in range(head_num):
            head_param = _create_range_param_numpy(
                shape, start_at=head_range_param_starts[i])
            result.append(head_param)

        return result


def _create_range_param(head_num: int,
                        shape: tuple[int, int],
                        stack_axis: int,
                        head_range_param_starts: list[int] | None = None):

    if head_range_param_starts is None:
        # same for each head
        head_param = _create_range_param_numpy(shape)
        reps = (head_num, 1) if stack_axis == 0 else (1, head_num)
        result = np.tile(head_param, reps)

        return result
    else:
        result: np.ndarray | None = None
        for i in range(head_num):
            head_param = _create_range_param_numpy(
                shape, start_at=head_range_param_starts[i])

            if result is None:
                result = head_param
            else:
                result = np.concatenate((result, head_param), axis=stack_axis)

        assert result is not None
        return result


# def _custom_range_param_init_(query: torch.Tensor,
#                            key: torch.Tensor,
#                            value: torch.Tensor,
#                            output: torch.Tensor):

#     with torch.no_grad():
#         query.copy_(create_range_param(query))
#         key.copy_(create_range_param(key))
#         value.copy_(create_range_param(value))
#         output.copy_(create_range_param(output))

# def create_range_param(for_tensor: torch.Tensor):
#     return (torch.arange(1, for_tensor.numel() + 1) / 100.).reshape(for_tensor.shape)


def _create_range_param_numpy(shape: tuple[int, int], start_at: int = 1):
    return (np.arange(start_at, start_at + shape[0] * shape[1]) /
            100).reshape(shape)
