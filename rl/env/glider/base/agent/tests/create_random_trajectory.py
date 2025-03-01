import numpy as np

from env.glider.base.agent import GliderTrajectory


def create_random_trajectory(step_count: int):
    time_s = np.linspace(0, 10, step_count)
    position_m = np.cumsum(np.random.rand(step_count, 3) - 0.5,
                           axis=0)  # random walk
    velocity_m_per_s = np.gradient(position_m, axis=0)

    yaw_pitch_roll_earth_to_body_rad = np.random.rand(step_count, 3) - 0.5
    air_velocity_m_per_s = velocity_m_per_s + np.array([1., 2., 3.])
    indicated_airspeed_m_per_s = np.random.rand(step_count)

    trajectory = GliderTrajectory(
        time_s=time_s,
        position_earth_xyz_m=position_m,
        velocity_earth_xyz_m_per_s=velocity_m_per_s,
        yaw_pitch_roll_earth_to_body_rad=yaw_pitch_roll_earth_to_body_rad,
        air_velocity_earth_xyz_m_per_s=air_velocity_m_per_s,
        indicated_airspeed_m_per_s=indicated_airspeed_m_per_s)

    return trajectory
