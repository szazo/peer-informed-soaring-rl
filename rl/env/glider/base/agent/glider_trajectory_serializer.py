import h5py
import numpy as np

from env.glider.base.agent.glider_info import GliderTrajectory


class GliderTrajectorySerializer:

    def save(self, trajectory: GliderTrajectory, out_filepath: str):

        with h5py.File(out_filepath, 'w') as f:
            self._serialize(trajectory, f)

    def load(self, filepath: str) -> GliderTrajectory:
        with h5py.File(filepath, 'r') as f:
            trajectory = self._deserialize(f)

            return trajectory

    def _serialize(self, trajectory: GliderTrajectory, group: h5py.Group):

        group.create_dataset('time_s', data=trajectory.time_s)
        group.create_dataset('position_earth_xyz_m',
                             data=trajectory.position_earth_xyz_m)
        group.create_dataset('velocity_earth_xyz_m_per_s',
                             data=trajectory.velocity_earth_xyz_m_per_s)
        group.create_dataset('yaw_pitch_roll_earth_to_body_rad',
                             data=trajectory.yaw_pitch_roll_earth_to_body_rad)
        group.create_dataset('air_velocity_earth_xyz_m_per_s',
                             data=trajectory.air_velocity_earth_xyz_m_per_s)
        group.create_dataset('indicated_airspeed_m_per_s',
                             data=trajectory.indicated_airspeed_m_per_s)

    def _deserialize(self, group: h5py.Group):

        trajectory = GliderTrajectory(
            time_s=self._read_dataset_as_nparray(group, 'time_s'),
            position_earth_xyz_m=self._read_dataset_as_nparray(
                group, 'position_earth_xyz_m'),
            velocity_earth_xyz_m_per_s=self._read_dataset_as_nparray(
                group, 'velocity_earth_xyz_m_per_s'),
            yaw_pitch_roll_earth_to_body_rad=self._read_dataset_as_nparray(
                group, 'yaw_pitch_roll_earth_to_body_rad'),
            air_velocity_earth_xyz_m_per_s=self._read_dataset_as_nparray(
                group, 'air_velocity_earth_xyz_m_per_s'),
            indicated_airspeed_m_per_s=self._read_dataset_as_nparray(
                group, 'indicated_airspeed_m_per_s'))

        return trajectory

    def _read_dataset_as_nparray(self, parent: h5py.Group, key: str):
        dataset = parent[key]
        assert isinstance(dataset, h5py.Dataset)

        return np.array(dataset)
