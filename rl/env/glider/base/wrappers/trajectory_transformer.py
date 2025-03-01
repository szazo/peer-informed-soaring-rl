from typing import Literal
from dataclasses import dataclass
import numpy as np
from utils.vector import VectorNx3

from .trajectory_rotator import TrajectoryRotator, translate_trajectory_to_origo


@dataclass
class TrajectoryRotatorParams:

    rotate_around: Literal['last', 'first'] | None
    rotate_to: list[float] | None
    project_to: Literal['xy_plane'] | None = 'xy_plane'


@dataclass(kw_only=True)
class TrajectoryTransformerParams(TrajectoryRotatorParams):

    translate_relative_to: Literal['last', 'first']


def create_trajectory_rotator(params: TrajectoryRotatorParams):

    if params.rotate_around is None:
        return None

    # we project the anchor vector to the xy plane, it is used to define the rotation axis,
    # so the z coordinate won't change during rotation
    assert params.project_to is None or params.project_to == 'xy_plane'
    anchor_section_transform = None
    if params.project_to == 'xy_plane':
        anchor_section_transform = lambda v: np.array([v[0], v[1], 0.])

    assert params.rotate_to is not None, 'rotate_to should be set'
    rotator = TrajectoryRotator(
        rotate_around=params.rotate_around,
        target_axis=np.array(params.rotate_to),
        anchor_vector='velocity',
        anchor_section_transform=anchor_section_transform)

    return rotator


class TrajectoryTransformer:

    _params: TrajectoryTransformerParams

    def __init__(self, params: TrajectoryTransformerParams):
        self._params = params

    def transform(self, position: VectorNx3, velocity: VectorNx3):

        assert position.ndim == 2, 'only matrix position input is allowed'
        assert velocity.ndim == 2, 'only matrix velocity input is allowed'

        # rotate
        rotator = create_trajectory_rotator(params=self._params)
        if rotator is not None:
            position, velocity = rotator.rotate(position, velocity)

        # translate the positions, so the first or the last position will be at the origo
        position_rotated_translated, _ = translate_trajectory_to_origo(
            position, relative_to=self._params.translate_relative_to)
        return position_rotated_translated, velocity
