from typing import Literal, Callable
from dataclasses import dataclass, field
import numpy as np
import quaternion
from utils import Vector3D, Vector3, VectorNx3


@dataclass
class Transform:
    # This translation need to be done before the rotation (and after the rotation it should be restored).
    translation_before_rotation: Vector3 = field(
        default_factory=lambda: np.array([0., 0., 0.]))

    # Quaternion for the actual rotation
    rotation: quaternion.quaternion = field(
        default_factory=lambda: quaternion.quaternion(1))


class TrajectoryRotator:

    _rotate_around: Literal['last', 'first']
    _target_axis: Vector3D
    _anchor_vector: Literal['position_diff', 'velocity']
    _anchor_section_transform: Callable[[Vector3D], Vector3D] | None

    def __init__(self,
                 rotate_around: Literal['last', 'first'],
                 target_axis: Vector3D,
                 anchor_vector: Literal['position_diff', 'velocity'],
                 anchor_section_transform: Callable[[Vector3D], Vector3D]
                 | None = None):

        self._rotate_around = rotate_around
        self._target_axis = target_axis
        self._anchor_vector = anchor_vector
        self._anchor_section_transform = anchor_section_transform

    def rotate(self, position: VectorNx3, velocity: VectorNx3):

        # calculate the required translation and rotation
        transform = self.create_transform(position, velocity)

        if transform is None:
            return position, velocity

        # translate and rotate
        position_new, velocity_new = self.transform(transform, position,
                                                    velocity)
        return position_new, velocity_new

    def transform(self, transform: Transform, position: VectorNx3,
                  velocity: VectorNx3):

        # translate the position vector by the translate vector
        position_translated = position + transform.translation_before_rotation

        # rotate the trajectory (position and velocity)
        position_translated_rotated = rotate_by_quaternion(
            position_translated, transform.rotation)
        velocity_rotated = rotate_by_quaternion(velocity, transform.rotation)

        # translate back
        position_rotated = position_translated_rotated - transform.translation_before_rotation

        return position_rotated, velocity_rotated

    def create_transform(self, position: VectorNx3,
                         velocity: VectorNx3) -> Transform | None:
        assert position.ndim == 2, 'position list should be a matrix'
        assert velocity.ndim == 2, 'velocity list should be a matrix'
        assert position.shape[0] == velocity.shape[
            0], 'position count should match velocity count'

        if self._anchor_vector == 'velocity' and position.shape[0] == 0:
            return None
        elif self._anchor_vector == 'position_diff' and position.shape[0] < 2:
            # for position, we need minimum two points
            return None

        assert position.shape[1] == 3, 'position vectors should be 3D'
        assert velocity.shape[1] == 3, 'velocity vectors should be 3D'

        # translate the first or last point to the origo
        _, translate_vector = translate_trajectory_to_origo(
            position, relative_to=self._rotate_around)

        # determine the vector that is used for calculating the rotation difference to the target vector
        anchor_vector = self._determine_anchor_vector(position=position,
                                                      velocity=velocity)

        # transform the anchor section using the provided Callable (e.g. project to a plane)
        if self._anchor_section_transform is not None:
            anchor_vector = self._anchor_section_transform(anchor_vector)

        # normalize
        anchor_vector_hat = anchor_vector / np.linalg.norm(anchor_vector)
        target_axis_hat = self._target_axis / np.linalg.norm(self._target_axis)

        # rotate to match with the target axis

        # create the quaternion for the rotation difference
        q = construct_quaternion_from_vectors(anchor_vector_hat,
                                              target_axis_hat)

        return Transform(translation_before_rotation=translate_vector,
                         rotation=q)

    def _determine_anchor_vector(self, position: VectorNx3,
                                 velocity: VectorNx3):
        if self._anchor_vector == 'velocity':
            if self._rotate_around == 'last':
                # rotate the last section to the target axis
                anchor_vector = velocity[-1, :]
            elif self._rotate_around == 'first':
                # rotate the first section to the target axis
                anchor_vector = velocity[0, :]
            else:
                assert False, f'invalid rotate_around: {self._rotate_around}'
        elif self._anchor_vector == 'position_diff':
            if self._rotate_around == 'last':
                # rotate the last section to the target axis
                anchor_vector = position[-1, :] - position[-2, :]
            elif self._rotate_around == 'first':
                # rotate the first section to the target axis
                anchor_vector = position[1, :] - position[0, :]
            else:
                assert False, f'invalid rotate_around: {self._rotate_around}'
        else:
            assert False, f'invalid anchor_vector: {self._anchor_vector}'

        return anchor_vector


def calculate_trajectory_to_origo_translate(
        trajectory: VectorNx3, relative_to: Literal['first',
                                                    'last']) -> Vector3:
    if trajectory.shape[0] == 0:
        return np.array([0., 0., 0])

    assert trajectory.shape[1] == 3, 'position vectors should be 3D'

    translate_vector: Vector3D
    if relative_to == 'last':
        # translate the positions to be the last point at the origin
        translate_vector = -trajectory[-1, :]
    elif relative_to == 'first':
        # translate the positions to be the first point at the origin
        translate_vector = -trajectory[0, :]
    else:
        assert False, f'invalid relative_to: {relative_to}'

    return translate_vector


def translate_trajectory_to_origo(
        trajectory: VectorNx3,
        relative_to: Literal['first', 'last']) -> tuple[VectorNx3, Vector3D]:

    translate_vector = calculate_trajectory_to_origo_translate(
        trajectory, relative_to=relative_to)

    trajectory_translated = trajectory
    if trajectory.shape[0] > 0:
        trajectory_translated = trajectory + translate_vector

    return trajectory_translated, translate_vector


def construct_quaternion_from_vectors(source: Vector3D, target: Vector3D):

    # normalize
    source_hat = source / np.linalg.norm(source)
    target_hat = target / np.linalg.norm(target)

    # calculate the rotation axis, which is the cross product of source_hat and the target_hat vectors
    rotation_axis = np.cross(source_hat, target_hat)

    if np.allclose(rotation_axis, 0.):
        # parallel vectors, return identity quaternion
        return quaternion.quaternion(1)

    # normalize the rotation axis
    rotation_axis_hat = rotation_axis / np.linalg.norm(rotation_axis)

    # calculate the angle, which is the arccos of the dot product of the two unit vectors
    theta = np.arccos(np.dot(source_hat, target_hat))

    # construct the quaternion
    q = quaternion.quaternion(np.cos(theta / 2),
                              *(np.sin(theta / 2) * rotation_axis_hat))

    return q


def rotate_by_quaternion(vectors: VectorNx3, q: quaternion.quaternion):

    assert vectors.ndim == 2, 'only matrix is supported'
    assert vectors.shape[1] == 3, 'only 3D vectors are supported'

    assert np.isclose(np.abs(q), 1.0), 'quaternion should be unit quaternion'

    if vectors.shape[0] == 0:
        return vectors.copy()

    # create the inverse of the quaternion
    # (we can use the conjugate, because it is unit)
    q_inv = q.conjugate()

    # create pure quaternion array from the input
    input_q = np.hstack((np.zeros((vectors.shape[0], 1)), vectors))
    input_q = quaternion.as_quat_array(input_q)

    # rotate
    rotated_q = q * input_q * q_inv

    # only use the vector part from the quaternions
    rotated = quaternion.as_float_array(rotated_q)[:, 1:]

    return rotated
