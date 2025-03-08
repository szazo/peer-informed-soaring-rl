from dataclasses import dataclass
import numpy as np
import pandas as pd

from thermal.api import AirVelocityFieldInterface
from utils.vector import VectorNx2, VectorNx3, VectorN


@dataclass
class TrajectoryFieldMappingParameters:
    time_field: str
    x_field: str
    y_field: str
    z_field: str


@dataclass
class TrajectoryToObservationLogConverterParameters:
    field_mappings: TrajectoryFieldMappingParameters
    filters: dict[str, str | float | int]


class TrajectoryToObservationLogConverter:

    _air_velocity_field: AirVelocityFieldInterface
    _params: TrajectoryToObservationLogConverterParameters

    def __init__(self, air_velocity_field: AirVelocityFieldInterface,
                 params: TrajectoryToObservationLogConverterParameters):

        self._air_velocity_field = air_velocity_field
        self._params = params

    def convert(self, input_df: pd.DataFrame) -> pd.DataFrame:

        for key, value in self._params.filters.items():
            input_df = input_df.loc[input_df[key] == value]

        field_mappings = self._params.field_mappings

        # time
        time_s: VectorN = np.array(
            input_df[field_mappings.time_field].astype('float32'))
        # shift the time to start with zero
        time_s = time_s - time_s.min()

        # position
        position_earth_m_xyz: VectorNx3 = np.column_stack(
            (np.array(input_df[field_mappings.x_field].astype('float32')),
             np.array(input_df[field_mappings.y_field].astype('float32')),
             np.array(input_df[field_mappings.z_field].astype('float32'))))

        # calculate velocity using np.gradient
        velocity_earth_m_per_s_xyz: VectorNx3 = np.gradient(
            position_earth_m_xyz, time_s, axis=0)

        velocity_earth_m_per_s: VectorN = np.linalg.norm(
            velocity_earth_m_per_s_xyz, axis=1)

        # acceleration
        acceleration_earth_m_per_s2_xyz: VectorNx3 = np.gradient(
            velocity_earth_m_per_s_xyz, time_s, axis=0)
        acceleration_earth_m_per_s2: VectorN = np.linalg.norm(
            acceleration_earth_m_per_s2_xyz, axis=1)

        # air velocity field is static
        air_velocity_time_s = 0.

        # core position
        core_position_earth_m_xy: VectorNx2 = self._air_velocity_field.get_thermal_core(
            z_earth_m=position_earth_m_xyz[:, 2], t_s=air_velocity_time_s)

        # velocity vector length
        distance_from_core_m: VectorN = np.linalg.norm(
            position_earth_m_xyz[:, :2] - core_position_earth_m_xy, axis=1)

        # query air velocity
        air_velocity_m_per_s_xyz, _ = self._air_velocity_field.get_velocity(
            x_earth_m=position_earth_m_xyz[:, 0],
            y_earth_m=position_earth_m_xyz[:, 1],
            z_earth_m=position_earth_m_xyz[:, 2],
            t_s=air_velocity_time_s)

        horizontal_air_velocity_earth_m_s = np.linalg.norm(
            air_velocity_m_per_s_xyz[:, :2], axis=1)

        horizontal_air_velocity_direction_deg = np.rad2deg(
            np.arctan2(
                air_velocity_m_per_s_xyz[:, 1],
                air_velocity_m_per_s_xyz[:, 0],
            ))

        output_df = pd.DataFrame({
            'index':
            range(0, time_s.shape[0]),
            'time_s':
            time_s,
            'position_earth_m_x':
            position_earth_m_xyz[:, 0],
            'position_earth_m_y':
            position_earth_m_xyz[:, 1],
            'position_earth_m_z':
            position_earth_m_xyz[:, 2],
            'velocity_earth_m_per_s_x':
            velocity_earth_m_per_s_xyz[:, 0],
            'velocity_earth_m_per_s_y':
            velocity_earth_m_per_s_xyz[:, 1],
            'velocity_earth_m_per_s_z':
            velocity_earth_m_per_s_xyz[:, 2],
            'velocity_earth_m_per_s':
            velocity_earth_m_per_s,
            'acceleration_earth_m_per_s2':
            acceleration_earth_m_per_s2,
            'core_position_m_x':
            core_position_earth_m_xy[:, 0],
            'core_position_m_y':
            core_position_earth_m_xy[:, 1],
            'distance_from_core_m':
            distance_from_core_m,
            'air_velocity_earth_m_per_s_x':
            air_velocity_m_per_s_xyz[:, 0],
            'air_velocity_earth_m_per_s_y':
            air_velocity_m_per_s_xyz[:, 1],
            'air_velocity_earth_m_per_s_z':
            air_velocity_m_per_s_xyz[:, 2],
            'horizontal_air_velocity_earth_m_s':
            horizontal_air_velocity_earth_m_s,
            'horizontal_air_velocity_direction_deg':
            horizontal_air_velocity_direction_deg
        })

        return output_df
