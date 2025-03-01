import quaternion
import numpy as np

from utils import Vector3
from utils.differential_rotator import DifferentialRotator


def test_rotate_to_should_return_zero_when_no_change():

    # given
    rotator = DifferentialRotator()
    rotator.rotate_to(
        yaw_pitch_roll_earth_to_body_rad=np.array([90., 45., 45.]))

    # when
    axis, angle_rad = rotator.rotate_to(
        yaw_pitch_roll_earth_to_body_rad=np.array([90., 45., 45.]))

    # then
    assert np.allclose(axis, 0.)
    assert angle_rad == 0.


def test_rotate_should_incrementally_rotate():

    # given
    yaw_rotator = DifferentialRotator()
    pitch_rotator = DifferentialRotator()
    roll_rotator = DifferentialRotator()
    yaw_rotator.rotate_to(
        yaw_pitch_roll_earth_to_body_rad=np.array([45., 0., 0.]))
    pitch_rotator.rotate_to(
        yaw_pitch_roll_earth_to_body_rad=np.array([0., 45., 0.]))
    roll_rotator.rotate_to(
        yaw_pitch_roll_earth_to_body_rad=np.array([0., 0., 45.]))

    # when
    yaw_axis, yaw_angle_rad = yaw_rotator.rotate_to(
        yaw_pitch_roll_earth_to_body_rad=np.array([50., 0., 0.]))
    pitch_axis, pitch_angle_rad = pitch_rotator.rotate_to(
        yaw_pitch_roll_earth_to_body_rad=np.array([0., 50., 0.]))
    roll_axis, roll_angle_rad = roll_rotator.rotate_to(
        yaw_pitch_roll_earth_to_body_rad=np.array([0., 0., 50.]))

    # then
    expected_yaw_axis, expected_yaw_angle_rad = \
        expected_axis_angle_from_euler_angles(np.array([5., 0., 0.]))
    expected_pitch_axis, expected_pitch_angle_rad = \
        expected_axis_angle_from_euler_angles(np.array([0., 5., 0.]))
    expected_roll_axis, expected_roll_angle_rad = \
        expected_axis_angle_from_euler_angles(np.array([0., 0., 5.]))

    assert np.allclose(expected_yaw_axis, yaw_axis)
    assert np.isclose(expected_yaw_angle_rad, yaw_angle_rad)
    assert np.allclose(expected_pitch_axis, pitch_axis)
    assert np.isclose(expected_pitch_angle_rad, pitch_angle_rad)
    assert np.allclose(expected_roll_axis, roll_axis)
    assert np.isclose(expected_roll_angle_rad, roll_angle_rad)


def test_reset_should_rotate_back():

    # given
    rotator = DifferentialRotator()
    rotator.rotate_to(yaw_pitch_roll_earth_to_body_rad=np.array([90., 0., 0.]))

    # when
    axis, angle_rad = rotator.reset()

    # then
    expected_axis, expected_angle_rad = expected_axis_angle_from_euler_angles(
        np.array([-90., 0., 0.]))

    assert np.allclose(expected_axis, axis)
    assert np.isclose(expected_angle_rad, angle_rad)


def test_rotate_to_should_rotate_after_reset():

    # given
    rotator = DifferentialRotator()
    rotator.rotate_to(
        yaw_pitch_roll_earth_to_body_rad=np.array([90., 45., 45.]))

    # when
    rotator.reset()
    axis, angle_rad = rotator.rotate_to(
        yaw_pitch_roll_earth_to_body_rad=np.array([90., 45., 45.]))

    # then
    expected_axis, expected_angle_rad = expected_axis_angle_from_euler_angles(
        np.array([90., 45., 45.]))

    assert np.allclose(expected_axis, axis)
    assert np.isclose(expected_angle_rad, angle_rad)


def expected_axis_angle_from_euler_angles(
        yaw_pitch_roll_earth_to_body_rad: Vector3):
    roll_quaternion = quaternion.from_rotation_vector(
        np.array([1., 0., 0.]) * yaw_pitch_roll_earth_to_body_rad[2])
    pitch_quaternion = quaternion.from_rotation_vector(
        np.array([0., 1., 0.]) * yaw_pitch_roll_earth_to_body_rad[1])
    yaw_quaternion = quaternion.from_rotation_vector(
        np.array([0., 0., 1.]) * yaw_pitch_roll_earth_to_body_rad[0])

    q = yaw_quaternion * pitch_quaternion * roll_quaternion
    rotation_vector = quaternion.as_rotation_vector(q)

    if np.allclose(rotation_vector, 0.):
        return np.zeros(3), 0.

    angle_rad = np.linalg.norm(rotation_vector).item()
    axis = rotation_vector / np.linalg.norm(rotation_vector)

    return axis, angle_rad
