import numpy as np
from gymnasium.utils import seeding

from env.glider.base.agent.air_velocity_filter import create_mean_kernel
from env.glider.base.agent.air_velocity_post_processor import (
    AirVelocityFilter, AirVelocityNoise, AirVelocityGaussianNoiseParameters,
    AirVelocityPostProcessor, GaussianNoiseParameters,
    AirVelocityPostProcessorParams)


def test_air_velocity_post_processor_add_noise_then_filter():

    # given
    params = AirVelocityPostProcessorParams(
        filter_kernel=create_mean_kernel(kernel_size=2),
        velocity_noise=AirVelocityGaussianNoiseParameters(
            x=GaussianNoiseParameters(mean=1., sigma=0.),
            y=GaussianNoiseParameters(mean=2., sigma=0.),
            z=GaussianNoiseParameters(mean=3., sigma=0.)))

    np_random, _ = seeding.np_random(seed=42)
    processor = AirVelocityPostProcessor(params=params, np_random=np_random)

    # when
    input1 = np.array([1., 2., 3.])
    result1 = processor.process(air_velocity_earth_xyz_m_s=input1)
    input2 = np.array([2., 3., 4.])
    result2 = processor.process(air_velocity_earth_xyz_m_s=input2)

    # then
    noise_mean = np.array([1., 2., 3.])

    assert np.allclose(result1, input1 + noise_mean)
    assert np.allclose(result2,
                       (input1 + noise_mean + input2 + noise_mean) / 2)


def test_air_velocity_filter_should_use_mean_when_not_fully_filled():

    # given
    kernel = create_mean_kernel(kernel_size=5)

    filter = AirVelocityFilter(kernel)

    # when
    result_1 = filter.filter(air_velocity_earth_xyz_m_s=np.array([1., 2., 3.]))
    result_2 = filter.filter(air_velocity_earth_xyz_m_s=np.array([2., 3., 4.]))

    # then
    assert np.allclose(result_1, np.array([1., 2., 3.]))
    assert np.allclose(result_2, np.array([1.5, 2.5, 3.5]))


def test_air_velocity_filter_should_reset_when_non_empty_state():

    # given
    kernel = create_mean_kernel(kernel_size=5)
    filter = AirVelocityFilter(kernel)

    # when
    filter.filter(air_velocity_earth_xyz_m_s=np.array([1., 2., 3.]))
    filter.reset()
    result_2 = filter.filter(air_velocity_earth_xyz_m_s=np.array([2., 3., 4.]))

    # then
    assert np.allclose(result_2, np.array([2., 3., 4.]))


def test_air_velocity_filter_should_clone():

    # given
    kernel = create_mean_kernel(kernel_size=5)
    filter = AirVelocityFilter(kernel)

    # when
    filter.filter(air_velocity_earth_xyz_m_s=np.array([1., 2., 3.]))
    clone = filter.state_clone()

    result_2_original = filter.filter(
        air_velocity_earth_xyz_m_s=np.array([2., 3., 4.]))
    result_2_clone = clone.filter(
        air_velocity_earth_xyz_m_s=np.array([2., 3., 4.]))

    # then
    assert np.allclose(result_2_original, np.array([1.5, 2.5, 3.5]))
    assert np.allclose(result_2_clone, np.array([1.5, 2.5, 3.5]))


def test_air_velocity_noise_should_use_different_distributions():

    # given
    params = AirVelocityGaussianNoiseParameters(
        x=GaussianNoiseParameters(mean=1., sigma=0.),
        y=GaussianNoiseParameters(mean=2., sigma=0.),
        z=GaussianNoiseParameters(mean=3., sigma=0.))
    np_random, _ = seeding.np_random()
    processor = AirVelocityNoise(params=params, np_random=np_random)
    input = np.array([2.1, 3.2, 4.3])

    # when
    result = processor.add_noise(input)

    # then
    assert np.allclose(result, np.array([3.1, 5.2, 7.3]))


def test_air_velocity_noise_should_use_seed():

    # given
    params = AirVelocityGaussianNoiseParameters(
        x=GaussianNoiseParameters(mean=1., sigma=1.),
        y=GaussianNoiseParameters(mean=2., sigma=1.),
        z=GaussianNoiseParameters(mean=3., sigma=1.))
    np_random, _ = seeding.np_random(seed=42)
    processor1 = AirVelocityNoise(params=params, np_random=np_random)

    np_random, _ = seeding.np_random(seed=42)
    processor2 = AirVelocityNoise(params=params, np_random=np_random)
    input = np.array([2.1, 3.2, 4.3])

    # when
    result1 = processor1.add_noise(input)
    result2 = processor2.add_noise(input)

    # then
    assert np.allclose(result1, result2)
