import numpy as np
import torch
from model.transformer.create_causal_attention_mask import create_causal_attention_mask


def test_create_should_create():

    # given
    length = 3

    # when
    mask = create_causal_attention_mask(sequence_length=length)

    # then
    assert mask.shape == (3, 3)
    assert np.allclose(
        mask,
        np.array([[0., -torch.inf, -torch.inf], \
                  [0., 0., -torch.inf], \
                  [0., 0., 0.]]))


def test_create_should_create_when_reverse():

    # given
    length = 3

    # when
    mask = create_causal_attention_mask(sequence_length=length, reverse=True)

    # then
    assert mask.shape == (3, 3)
    assert np.allclose(
        mask,
        np.array([[0., 0, 0], \
                  [-torch.inf, 0., 0], \
                  [-torch.inf, -torch.inf, 0.]]))


def test_create_should_return_zero_length_when_zero_sequence_length():

    # given
    length = 0

    # when
    mask = create_causal_attention_mask(sequence_length=length)

    # then
    assert mask.shape == (0, 0)
