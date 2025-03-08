from typing import Dict
import numpy as np
import tianshou as ts
import pandas as pd
from trainer.tianshou_evaluator.api import ObservationLogger


class GliderObservationLogger(ObservationLogger):

    def __init__(self):
        pass

    def transform_buffer_to_dataframe(self, buffer: ts.data.ReplayBuffer,
                                      output_df: pd.DataFrame):

        info = buffer.info

        output_df["time_s"] = info["t_s"]

        output_df["success"] = info.success
        output_df["cutoff_reason"] = info.cutoff_reason
        output_df["reward"] = buffer.rew

        position_earth_m = (info["position_earth_m"] if "position_earth_m"
                            in info else info["position_earth_xyz_m"])
        output_df["position_earth_m_x"] = position_earth_m[:, 0]
        output_df["position_earth_m_y"] = position_earth_m[:, 1]
        output_df["position_earth_m_z"] = position_earth_m[:, 2]

        velocity_earth_m_per_s = (info["velocity_earth_m_per_s"]
                                  if "velocity_earth_m_per_s" in info else
                                  info["velocity_earth_xyz_m_per_s"])
        output_df["velocity_earth_m_per_s_x"] = velocity_earth_m_per_s[:, 0]
        output_df["velocity_earth_m_per_s_y"] = velocity_earth_m_per_s[:, 1]
        output_df["velocity_earth_m_per_s_z"] = velocity_earth_m_per_s[:, 2]

        output_df["time_s_without_lift"] = info["time_s_without_lift"]
        output_df["distance_from_core_m"] = info["distance_from_core_m"]
        output_df["core_position_m_x"] = info["core_position_earth_m_xy"][:, 0]
        output_df["core_position_m_y"] = info["core_position_earth_m_xy"][:, 1]

        yaw_pitch_roll_earth_to_body_rad = info[
            "yaw_pitch_roll_earth_to_body_rad"]
        output_df["yaw_deg"] = np.rad2deg(yaw_pitch_roll_earth_to_body_rad[:,
                                                                           0])
        output_df["pitch_deg"] = np.rad2deg(
            yaw_pitch_roll_earth_to_body_rad[:, 1])
        output_df["roll_deg"] = np.rad2deg(yaw_pitch_roll_earth_to_body_rad[:,
                                                                            2])

        # current surrounding air velocity
        air_velocity_earth_m_s = (info["air_velocity_earth_m_s"]
                                  if "air_velocity_earth_m_s" in info else
                                  info["air_velocity_earth_xyz_m_s"])

        output_df["air_velocity_earth_m_per_s_x"] = air_velocity_earth_m_s[:,
                                                                           0]
        output_df["air_velocity_earth_m_per_s_y"] = air_velocity_earth_m_s[:,
                                                                           1]
        output_df["air_velocity_earth_m_per_s_z"] = air_velocity_earth_m_s[:,
                                                                           2]

        horizontal_air_velocity_earth_m_s = air_velocity_earth_m_s[:, :2]
        output_df["horizontal_air_velocity_earth_m_s"] = np.linalg.norm(
            horizontal_air_velocity_earth_m_s, axis=1)
        output_df["horizontal_air_velocity_direction_deg"] = np.rad2deg(
            np.arctan2(
                horizontal_air_velocity_earth_m_s[:, 1],
                horizontal_air_velocity_earth_m_s[:, 0],
            ))

        # initial conditions
        initial_conditions = info["initial_conditions"]

        # initial position
        glider_initial_conditions = initial_conditions["glider"]
        initial_position_earth_m = (
            glider_initial_conditions["position_earth_m"]
            if "position_earth_m" in glider_initial_conditions else
            glider_initial_conditions["position_earth_xyz_m"])
        output_df["initial_position_earth_m_z"] = initial_position_earth_m[:,
                                                                           2]

        if "tangent_position" in glider_initial_conditions:
            output_df["tangent_position_distance_from_core"] = (
                glider_initial_conditions["tangent_position"]
                ["distance_from_core_m"])
            output_df["starting_distance_from_tangent_position_m"] = (
                glider_initial_conditions["tangent_position"]
                ["starting_distance_from_tangent_position_m"])
        else:
            output_df["tangent_position_distance_from_core"] = (
                glider_initial_conditions["distance_from_core_m"])
            output_df["starting_distance_from_tangent_position_m"] = (
                glider_initial_conditions[
                    "starting_distance_from_tangent_position_m"])

        # air velocity field
        air_velocity_field = initial_conditions["air_velocity_field"]

        # REVIEW: implement air velocity field specific plugin
        if air_velocity_field is not None:
            if "max_r_m" in air_velocity_field:
                # only if we have data, realistic field has no initial condition log yet
                self._transform_air_velocity_field_initial_conditions_to_dataframe(
                    air_velocity_field, output_df)
            elif "original_thermal_z_min" in air_velocity_field:
                # stacked realistic thermal
                output_df["original_thermal_z_min"] = air_velocity_field[
                    "original_thermal_z_min"]
                output_df["original_thermal_z_max"] = air_velocity_field[
                    "original_thermal_z_max"]

        aerodynamics = info["aerodynamics"]
        output_df["mass_kg"] = aerodynamics["mass_kg"]
        output_df["wing_area_m2"] = aerodynamics["wing_area_m2"]
        output_df["CL"] = aerodynamics["CL"]
        output_df["CD"] = aerodynamics["CD"]
        output_df["rho_kg_per_m3"] = aerodynamics["rho_kg_per_m3"]
        output_df["g_m_per_s2"] = aerodynamics["g_m_per_s2"]

    def _transform_air_velocity_field_initial_conditions_to_dataframe(
            self, air_velocity_field: Dict, df: pd.DataFrame):

        df["thermal_max_r_m"] = air_velocity_field["max_r_m"]
        df["thermal_max_r_altitude_m"] = air_velocity_field["max_r_altitude_m"]
        df["thermal_max_r_m_sigma"] = air_velocity_field["max_r_m_sigma"]
        df["thermal_w_max_m_per_s"] = air_velocity_field["w_max_m_per_s"]
        df['rng_name'] = air_velocity_field["rng_name"]
        df['rng_state'] = air_velocity_field["rng_state"]

        # turbulence noise field
        turbulence_noise_field = air_velocity_field["turbulence_noise_field"]

        has_turbulence = not np.all(turbulence_noise_field == None)
        df.loc[:, "turbulence_noise_multiplier"] = (
            turbulence_noise_field["noise_multiplier"]
            if has_turbulence else 0)
        df.loc[:, "turbulence_gaussian_filter_sigma"] = (
            turbulence_noise_field["noise_multiplier"]
            if has_turbulence else 0)
        turbulence_grid_spacing = (
            turbulence_noise_field["grid_range"][..., -3:] /
            (turbulence_noise_field["grid_resolution"][..., -3:] - 1.0)
            if has_turbulence else None)
        df.loc[:, "turbulence_grid_spacing_x"] = (turbulence_grid_spacing[:, 0]
                                                  if turbulence_grid_spacing
                                                  is not None else 0)
        df.loc[:, "turbulence_grid_spacing_y"] = (turbulence_grid_spacing[:, 1]
                                                  if turbulence_grid_spacing
                                                  is not None else 0)
        df.loc[:, "turbulence_grid_spacing_z"] = (turbulence_grid_spacing[:, 2]
                                                  if turbulence_grid_spacing
                                                  is not None else 0)

        # horizontal wind
        df["horizontal_wind_velocity_earth_2_m_m_per_s"] = air_velocity_field[
            "horizontal_wind_speed_m_per_s"]
        df["horizontal_wind_direction_deg"] = np.rad2deg(
            air_velocity_field["horizontal_wind_direction_rad"])

        # horizontal wind noise field
        horizontal_wind_noise_field = air_velocity_field[
            "horizontal_wind_noise_field"]

        has_horizontal_wind_noise = not np.all(
            horizontal_wind_noise_field == None)
        df.loc[:, "horizontal_wind_noise_multiplier"] = (
            horizontal_wind_noise_field["noise_multiplier"]
            if has_horizontal_wind_noise else 0)
        df.loc[:, "horizontal_wind_gaussian_filter_sigma"] = (
            horizontal_wind_noise_field["noise_multiplier"]
            if has_horizontal_wind_noise else 0)

        horizontal_wind_grid_spacing = (
            horizontal_wind_noise_field["grid_range"][..., -1:] /
            (horizontal_wind_noise_field["grid_resolution"][..., -1:] - 1.0)
            if has_horizontal_wind_noise else None)
        df.loc[:, "horizontal_wind_grid_spacing"] = (
            horizontal_wind_grid_spacing[:, 0]
            if horizontal_wind_grid_spacing is not None else 0)
