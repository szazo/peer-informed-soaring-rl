from typing import Literal
from dataclasses import dataclass
import numpy as np
from utils.vector import VectorN, VectorNxN

from .trajectory_transformer import TrajectoryTransformer, TrajectoryTransformerParams
from .sequence_window import SequenceWindow
from .zero_pad_sequence import zero_pad_sequence


@dataclass
class TrajectoryEgocentricSequenceObservationTransformerParams:
    max_sequence_length: int
    trajectory_transform_params: TrajectoryTransformerParams
    position_3d_start_column_index: int
    velocity_3d_start_column_index: int
    reverse: bool
    zero_pad_at: None | Literal['start', 'end']


class TrajectoryEgocentricSequenceObservationTransformer:

    _params: TrajectoryEgocentricSequenceObservationTransformerParams

    _sequence_window: SequenceWindow
    _trajectory_transformer: TrajectoryTransformer

    def __init__(
            self,
            params: TrajectoryEgocentricSequenceObservationTransformerParams):

        self._params = params
        self._sequence_window = SequenceWindow(
            max_sequence_length=params.max_sequence_length)
        self._trajectory_transformer = TrajectoryTransformer(
            params=params.trajectory_transform_params)

    def reset(self):
        self._sequence_window.reset()

    def modify_obs(self, obs: VectorN) -> VectorNxN:

        assert obs.ndim == 1, 'the single observation window should be one dimensional'

        # add it at the end of the current sequence with windowing
        window = self._sequence_window.add(obs)

        # get the position and velocity parts
        position = window[:, self._params.position_3d_start_column_index:self.
                          _params.position_3d_start_column_index + 3]
        velocity = window[:, self._params.velocity_3d_start_column_index:self.
                          _params.velocity_3d_start_column_index + 3]

        # rotate and translate the trajectory to be egocentric
        position_transformed, velocity_transformed = self._trajectory_transformer.transform(
            position, velocity)

        # write back to the window
        window[:, self._params.position_3d_start_column_index:self._params.
               position_3d_start_column_index + 3] = position_transformed
        window[:, self._params.velocity_3d_start_column_index:self._params.
               velocity_3d_start_column_index + 3] = velocity_transformed

        # reverse
        if self._params.reverse:
            window = np.flip(window, axis=0)

        # zero pad
        if self._params.zero_pad_at is not None:
            window = zero_pad_sequence(
                window,
                max_sequence_length=self._params.max_sequence_length,
                pad_at=self._params.zero_pad_at)

        return window
