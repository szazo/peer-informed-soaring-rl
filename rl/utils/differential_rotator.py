import quaternion
import numpy as np

from utils.vector import Vector3


class DifferentialRotator:
    """
    Allow object to incrementally rotated based on the current orientation, but with absolute target orientation.
    """

    _current_quaternion: quaternion.quaternion

    def __init__(self):

        self._current_quaternion = quaternion.quaternion(1.)

    def rotate_to(
            self, yaw_pitch_roll_earth_to_body_rad: Vector3
    ) -> tuple[Vector3, float]:

        # calculate the target quaternion
        target_quaternion = self._quaternion_from_earth_relative_yaw_pitch_roll(
            yaw_pitch_roll_earth_to_body_rad)

        axis, angle_rad = self._quaternion_difference_as_rotation_vector(
            target_quaternion=target_quaternion,
            source_quaternion=self._current_quaternion)

        # update the current orientation
        self._current_quaternion = target_quaternion

        return axis, angle_rad

    def reset(self):

        # return the rotation back to the original orientation
        target_quaternion = quaternion.quaternion(1.)
        axis, angle_rad = self._quaternion_difference_as_rotation_vector(
            target_quaternion=target_quaternion,
            source_quaternion=self._current_quaternion)
        self._current_quaternion = target_quaternion
        return axis, angle_rad

    def _quaternion_difference_as_rotation_vector(
            self, target_quaternion: quaternion.quaternion,
            source_quaternion: quaternion.quaternion) -> tuple[Vector3, float]:

        # calculate the difference between the target and the current orientation's quaternion
        diff = target_quaternion * source_quaternion.conjugate()

        # return the difference as axis and angle in radians
        rotation_vector = quaternion.as_rotation_vector(diff)
        if np.allclose(rotation_vector, 0.):
            return np.zeros(3), 0.

        angle_rad = np.linalg.norm(rotation_vector).item()
        axis = rotation_vector / np.linalg.norm(rotation_vector)

        return axis, angle_rad

    def _quaternion_from_earth_relative_yaw_pitch_roll(
            self, yaw_pitch_roll_earth_to_body_rad: Vector3):
        roll_quaternion = quaternion.from_rotation_vector(
            np.array([1., 0., 0.]) * yaw_pitch_roll_earth_to_body_rad[2])
        pitch_quaternion = quaternion.from_rotation_vector(
            np.array([0., 1., 0.]) * yaw_pitch_roll_earth_to_body_rad[1])
        yaw_quaternion = quaternion.from_rotation_vector(
            np.array([0., 0., 1.]) * yaw_pitch_roll_earth_to_body_rad[0])

        result_quaternion = yaw_quaternion * pitch_quaternion * roll_quaternion
        return result_quaternion
