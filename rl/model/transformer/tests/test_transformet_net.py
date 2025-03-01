from dataclasses import asdict
import numpy as np
import torch
from model.transformer.transformer_net import TransformerNet, TransformerNetParameters
from trainer.multi_agent.tests.create_sample_items import create_sample_items_with_shapes


def test_transformer_net_should_return_same_result_with_padded_sequence():
    """TransformerNet should not use the zero padded part from any item of the batch for calculating the result"""

    # create the items
    dim = 2
    seq_lens = [4, 2]

    items = create_sample_items_with_shapes((seq_lens[0], dim),
                                            (seq_lens[1], dim))

    # pad
    pads = [(0, 2), (0, 4)]
    padded = [np.pad(item, (pads[i], (0, 0))) for i, item in enumerate(items)]

    # stack
    input = np.stack(padded, axis=0)

    transformer_net = _create_transformer_net(input_dim=dim,
                                              encoder_layer_count=1,
                                              attention_head_num=1)

    # when execute separately
    seq0_result, _ = transformer_net(np.expand_dims(items[0], axis=0))
    seq1_result, _ = transformer_net(np.expand_dims(items[1], axis=0))

    # when execute once withing a single batch
    batch_result, _ = transformer_net(input)

    assert batch_result.shape == (2, 3)
    assert torch.allclose(seq0_result, batch_result[0])
    assert torch.allclose(seq1_result, batch_result[1])


def test_transformer_net_should_work_if_multiple_dimensions_before_sequence():
    """TransformerNet should allow multiple dimensions before the sequence dimension."""

    # create the items
    dim = 2
    seq_lens = [4, 2, 3, 6]

    items = create_sample_items_with_shapes(
        (seq_lens[0], dim),
        (seq_lens[1], dim),
        (seq_lens[2], dim),
        (seq_lens[3], dim),
    )

    # pad (because of the positional encoding, padding only after)
    pads = [(0, 2), (0, 4), (0, 3), (0, 0)]
    padded = [np.pad(item, (pads[i], (0, 0))) for i, item in enumerate(items)]

    # stack
    input0 = np.stack(padded[:2], axis=0)
    input1 = np.stack(padded[2:], axis=0)

    input = np.stack((input0, input1), axis=0)
    transformer_net = _create_transformer_net(input_dim=dim,
                                              encoder_layer_count=3,
                                              attention_head_num=4)

    # # when execute separately
    seq0_result, _ = transformer_net(np.expand_dims(items[0], axis=0))
    seq1_result, _ = transformer_net(np.expand_dims(items[1], axis=0))
    seq2_result, _ = transformer_net(np.expand_dims(items[2], axis=0))
    seq3_result, _ = transformer_net(np.expand_dims(items[3], axis=0))

    # # when execute once withing a single batch
    batch_result, _ = transformer_net(input)

    assert batch_result.shape == (2, 2, 3)

    assert torch.allclose(seq0_result, batch_result[0][0])
    assert torch.allclose(seq1_result, batch_result[0][1])
    assert torch.allclose(seq2_result, batch_result[1][0])
    assert torch.allclose(seq3_result, batch_result[1][1])


def _create_transformer_net(input_dim: int,
                            encoder_layer_count=1,
                            attention_head_num=1,
                            output_dim=3):
    params = TransformerNetParameters(input_dim=input_dim,
                                      output_dim=output_dim,
                                      attention_internal_dim=7,
                                      attention_head_num=attention_head_num,
                                      ffnn_hidden_dim=4,
                                      ffnn_dropout_rate=0.0,
                                      max_sequence_length=100,
                                      embedding_dim=4,
                                      encoder_layer_count=encoder_layer_count,
                                      enable_layer_normalization=True,
                                      enable_causal_attention_mask=True,
                                      is_reversed_sequence=True,
                                      softmax_output=False)

    transformer_net = TransformerNet(**asdict(params))

    return transformer_net


def test_transformer_net_should_not_smoke_when_everything_enabled():

    # given
    params = TransformerNetParameters(input_dim=5,
                                      output_dim=6,
                                      attention_internal_dim=7,
                                      attention_head_num=8,
                                      ffnn_hidden_dim=4,
                                      ffnn_dropout_rate=0.1,
                                      max_sequence_length=100,
                                      embedding_dim=64,
                                      encoder_layer_count=4,
                                      enable_layer_normalization=True,
                                      enable_causal_attention_mask=True,
                                      is_reversed_sequence=True,
                                      softmax_output=True)

    # when
    transformer_net = TransformerNet(**asdict(params))
    batch_size = 4
    sequence_length = 6
    dim = 5
    transformer_net(torch.rand((batch_size, sequence_length, dim)))

    # then
    assert transformer_net is not None
