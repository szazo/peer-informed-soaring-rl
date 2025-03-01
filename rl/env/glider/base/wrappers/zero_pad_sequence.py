from typing import Literal, Any

import numpy as np
from utils.vector import VectorNxN


def zero_pad_sequence(input: VectorNxN,
                      max_sequence_length: int,
                      pad_at: Literal['start', 'end'],
                      constant_value: Any = 0.,
                      target_axis: int = 0):
    """Pads a multi dimensional array along the first axis"""

    assert pad_at == 'start' or pad_at == 'end'

    length = input.shape[target_axis]
    if length < max_sequence_length:

        # config for the first axis
        pad_length = max_sequence_length - length
        pad_start = pad_length if pad_at == 'start' else 0
        pad_end = pad_length if pad_at == 'end' else 0
        target_axis_config = (pad_start, pad_end)

        # create config for each axis
        pad_width = list([(0, 0) for _ in range(input.ndim)])
        pad_width[target_axis] = target_axis_config

        # pad the rows dimension only
        padded = np.pad(input,
                        pad_width=pad_width,
                        constant_values=constant_value)
        return padded

    return input
