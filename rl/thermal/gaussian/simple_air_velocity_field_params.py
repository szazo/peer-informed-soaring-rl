from dataclasses import dataclass
from utils import RandomGeneratorState


@dataclass
class GaussianThermalParameters:

    # maximum radius
    max_r_m_normal_mean: float
    max_r_m_normal_sigma: float
    # altitude of the maximum radius
    max_r_altitude_m_normal_mean: float
    max_r_altitude_m_normal_sigma: float
    # spread of the maximum radius around the maximum radius altitude
    max_r_m_sigma_normal_mean: float
    max_r_m_sigma_normal_sigma: float
    # vertical velocity at the core
    w_max_m_per_s_normal_mean: float
    w_max_m_per_s_normal_sigma: float

    # used for control that the specified sigma should contain the most of the bell volume: sigma'=sigma/k
    sigma_k: float = 2.5
    # because the thermal with the radius is gaussian, we use
    # k for the radius too to limit distribution into the range of specified thermal radius
    radius_k: float = 1.5


@dataclass
class NoiseParameters:
    noise_grid_spacing_m: float

    noise_multiplier_normal_mean: float
    noise_multiplier_normal_sigma: float

    noise_gaussian_filter_sigma_normal_mean_m: float
    noise_gaussian_filter_sigma_normal_sigma_m: float

    # time not used now
    time_low_s: float = 0.0
    time_high_s: float = 0.0
    time_space_s: float = 1.

    sigma_k: float = 1.0

    seed: int | None = None


@dataclass
class SimpleTurbulenceParameters(NoiseParameters):
    pass


@dataclass
class HorizontalWindParameters:
    horizontal_wind_speed_at_2m_m_per_s_normal_mean: float
    horizontal_wind_speed_at_2m_m_per_s_normal_sigma: float
    horizontal_wind_profile_vertical_spacing_m: float
    sigma_k: float = 2.5


@dataclass(kw_only=True)
class SimpleWindParameters(HorizontalWindParameters):
    noise: NoiseParameters | None


@dataclass
class GaussianAirVelocityFieldParameters:
    box_size: list[float]
    thermal: GaussianThermalParameters | None
    turbulence: SimpleTurbulenceParameters | None
    turbulence_episode_regenerate_probability: float
    wind: SimpleWindParameters | None
    seed: int | None = None
    random_state: RandomGeneratorState | None = None
