from dataclasses import dataclass
import pandas as pd
import numpy as np
from utils import RandomGeneratorState

from .air_velocity_scene_actor import ThermalParams, Scene, Episode
from .glider_agent_actor import AgentParams, AgentEpisode, Trajectory


class ObservationLogSceneLoader:

    def load(self, file_path: str) -> list[Scene]:
        obs_log = pd.read_csv(file_path)

        scenes = self._process_scenes(obs_log)

        return scenes

    def _process_scenes(self, obs_log: pd.DataFrame) -> list[Scene]:

        scenes = []

        # for each scene, collect the episodes
        scene_group_columns = ['thermal', 'rng_name', 'rng_state']

        scene_groupby = obs_log.groupby(scene_group_columns)
        for details, scene_df in scene_groupby:

            assert isinstance(details, tuple)
            (thermal_name, rng_name, rng_state) = details

            thermal_params = ThermalParams(
                thermal_name=thermal_name,
                thermal_params='',
                thermal_random_state=RandomGeneratorState(
                    generator_name=rng_name, encoded_state_values=rng_state))
            episodes = self._process_episodes(scene_df=scene_df)

            scene = Scene(thermal=thermal_params, episodes=episodes)
            scenes.append(scene)

        return scenes

    def _process_episodes(self, scene_df: pd.DataFrame) -> list[Episode]:

        episodes = []

        episode_groupby = scene_df.groupby('episode')

        for _, episode_df in episode_groupby:

            # collect agents for the episode
            episode_agents = self._process_episode_agents(episode_df)

            episode = Episode(agents=episode_agents)
            episodes.append(episode)

        return episodes

    def _process_episode_agents(
            self, episode_df: pd.DataFrame) -> list[AgentEpisode]:

        agent_name_column = None
        if 'agent_id' in episode_df.columns:
            agent_name_column = 'agent_id'
        elif 'bird_name' in episode_df.columns:
            agent_name_column = 'bird_name'

        assert agent_name_column is not None

        agent_column_names = ['agent_type', 'training']
        agent_column_names.append(agent_name_column)

        agent_groupby = episode_df.groupby(agent_column_names)

        result = []
        for agent_details, agent_df in agent_groupby:

            assert isinstance(agent_details, tuple)
            (agent_type, agent_training, agent_name) = agent_details

            agent = AgentParams(agent_type=agent_type,
                                agent_name=agent_name,
                                agent_training=agent_training)
            trajectory = self._convert_agent_episode_df_to_trajectory(
                agent_episode_df=agent_df)

            agent_episode = AgentEpisode(agent=agent, trajectory=trajectory)
            result.append(agent_episode)

        return result

    def _convert_agent_episode_df_to_trajectory(
            self, agent_episode_df: pd.DataFrame) -> Trajectory:

        time_s = agent_episode_df['time_s'].to_numpy()

        position_xyz_m = agent_episode_df[[
            'position_earth_m_x', 'position_earth_m_y', 'position_earth_m_z'
        ]].to_numpy()

        velocity_xyz_m = agent_episode_df[[
            'velocity_earth_m_per_s_x', 'velocity_earth_m_per_s_y',
            'velocity_earth_m_per_s_z'
        ]].to_numpy()

        air_velocity_xyz_m_per_s = agent_episode_df[[
            'air_velocity_earth_m_per_s_x', 'air_velocity_earth_m_per_s_y',
            'air_velocity_earth_m_per_s_z'
        ]].to_numpy()

        yaw_pitch_roll_deg = agent_episode_df[[
            'yaw_deg', 'pitch_deg', 'roll_deg'
        ]].to_numpy()
        yaw_pitch_roll_rad = np.deg2rad(yaw_pitch_roll_deg)

        # should be converted, because created by atan2
        # yaw_north_up_rad = (current_yaw_pitch_roll_rad[0] + np.pi / 2) % (np.pi * 2)
        east_relative_yaw_rad = (yaw_pitch_roll_rad[:, 0] +
                                 2 * np.pi) % (np.pi * 2)
        east_relative_yaw_rad = east_relative_yaw_rad.reshape(-1, 1)

        east_relative_yaw_pitch_roll_rad = np.hstack(
            (east_relative_yaw_rad, yaw_pitch_roll_rad[:, 1:3]))

        trajectory = Trajectory(
            time_s=time_s,
            position_earth_xyz_m=position_xyz_m,
            velocity_earth_xyz_m_per_s=velocity_xyz_m,
            yaw_pitch_roll_earth_to_body_rad=east_relative_yaw_pitch_roll_rad,
            air_velocity_earth_xyz_m_per_s=air_velocity_xyz_m_per_s)

        return trajectory
