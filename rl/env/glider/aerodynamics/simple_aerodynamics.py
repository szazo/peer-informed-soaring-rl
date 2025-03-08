import numpy as np
import numpy.typing as npt
from utils import Vector3D
from .api import AerodynamicsInterface, AerodynamicsInfo


class SimpleAerodynamics(AerodynamicsInterface):

    def __init__(
        self,
        mass_kg: float,
        wing_area_m2: float,
        CD: float,
        CL: float,
        rho_kg_per_m3: float,
        g_m_per_s2: float,
    ):
        self._mass_kg = mass_kg
        self._wing_area_m2 = wing_area_m2
        self._CD = CD
        self._CL = CL
        self._rho_kg_per_m3 = rho_kg_per_m3
        self._g_m_per_s2 = g_m_per_s2

    def reset(self) -> AerodynamicsInfo:
        return AerodynamicsInfo(mass_kg=self._mass_kg,
                                wing_area_m2=self._wing_area_m2,
                                CD=self._CD,
                                CL=self._CL,
                                rho_kg_per_m3=self._rho_kg_per_m3,
                                g_m_per_s2=self._g_m_per_s2)

    def get_initial_velocity_earth(
        self,
        heading_earth_to_body_rad: float,
        wind_velocity_earth_m_per_s: npt.NDArray[np.float_],
    ):

        velocity_airmass_relative_m_per_s = self._get_new_velocity_vector_components(
            bank_angle_earth_to_wind_rad=0.,
            heading_earth_to_body_rad=heading_earth_to_body_rad,
            CL=self._CL,
            CD=self._CD)

        # add current wind vector to get earth relative velocity
        velocity_earth_m_per_s = (velocity_airmass_relative_m_per_s +
                                  wind_velocity_earth_m_per_s)

        return velocity_earth_m_per_s, velocity_airmass_relative_m_per_s

    def _get_new_velocity_vector_components(
            self, bank_angle_earth_to_wind_rad: float,
            heading_earth_to_body_rad: float, CL: float, CD: float):

        Vh_m_per_s = self.get_horizontal_velocity(bank_angle_earth_to_wind_rad,
                                                  CL)  # horizontal velocity

        Vz_m_per_s = -self.get_min_sink_rate_from_bank_angle(
            bank_angle_earth_to_wind_rad, CL, CD)

        Vx_m_per_s, Vy_m_per_s = Vh_m_per_s * np.cos(
            heading_earth_to_body_rad), Vh_m_per_s * np.sin(
                heading_earth_to_body_rad)

        return np.array([Vx_m_per_s, Vy_m_per_s, Vz_m_per_s])

    def step(
        self,
        # state variables (position vector, velocity vector, body rotation)
        position_earth_m: Vector3D,  # current position (in NED frame)
        velocity_earth_m_per_s: Vector3D,  # current velocity (in NED frame)
        yaw_pitch_roll_earth_to_body_rad:
        Vector3D,  # current body attitude (we know this from e.g. gyroscope)
        wind_velocity_earth_m_per_s: Vector3D,
        dt_s,  # change in time between the current and next
        velocity_airmass_relative_m_per_s: Vector3D):

        [
            yaw_earth_to_body_rad,
            pitch_earth_to_body_rad,
            roll_earth_to_body_rad,
        ] = yaw_pitch_roll_earth_to_body_rad

        bank_angle_earth_to_wind_rad = roll_earth_to_body_rad

        # we have the heading
        heading_earth_to_body_rad = self.get_new_heading(
            velocity_airmass_relative_m_per_s, bank_angle_earth_to_wind_rad,
            self._CL, dt_s)

        # calculate the new velocity
        next_velocity_airmass_relative_m_per_s = self._get_new_velocity_vector_components(
            bank_angle_earth_to_wind_rad=0.,
            heading_earth_to_body_rad=heading_earth_to_body_rad,
            CL=self._CL,
            CD=self._CD)

        # add current wind vector to get earth relative velocity
        next_velocity_earth_m_per_s = (next_velocity_airmass_relative_m_per_s +
                                       wind_velocity_earth_m_per_s)

        # integrate to get next position
        next_position_earth_m = position_earth_m + velocity_earth_m_per_s * dt_s

        # get the new attitude (yaw is same as heading, pitch, roll does not change)
        next_yaw_pitch_roll_earth_to_body = np.array([
            heading_earth_to_body_rad, pitch_earth_to_body_rad,
            roll_earth_to_body_rad
        ])

        # calculate the airspeed
        true_airspeed_m_per_s = np.linalg.norm(
            next_velocity_airmass_relative_m_per_s)
        indicated_airspeed_m_per_s = (
            true_airspeed_m_per_s  # we use the true airspeed here
        )

        return (next_position_earth_m, next_velocity_earth_m_per_s,
                next_yaw_pitch_roll_earth_to_body, indicated_airspeed_m_per_s,
                next_velocity_airmass_relative_m_per_s)

    def get_new_heading(
        self,
        velocity_m_per_s: npt.NDArray[np.float_],
        bank_angle_earth_to_wind_rad: float,
        CL: float,
        dt_s: float,
    ):

        centripetal_acceleration_m_per_s2 = self.get_centripetal_acceleration(
            velocity_m_per_s, bank_angle_earth_to_wind_rad, CL)
        delta_Vh_m_per_s = centripetal_acceleration_m_per_s2 * dt_s
        Vh_m_per_s = (np.array([velocity_m_per_s[0], velocity_m_per_s[1]]) +
                      delta_Vh_m_per_s)
        return np.arctan2(Vh_m_per_s[1], Vh_m_per_s[0])

    def get_centripetal_acceleration(
        self,
        velocity_m_per_s: npt.NDArray[np.float_],
        bank_angle_earth_to_wind_rad: float,
        CL: float,
    ):

        # Vh_prev is not squared because it cancels out
        lift_force_n = (
            CL * np.sin(bank_angle_earth_to_wind_rad) * (1 / 2) *
            self._rho_kg_per_m3 * self._wing_area_m2 *
            np.linalg.norm(velocity_m_per_s[:2]) *
            np.array([-velocity_m_per_s[1], velocity_m_per_s[0]
                      ]  # this is not unit vector, so V square not required
                     )  # rotate left counterclockwise 90 degrees
        )

        centripetal_acceleration_m_s2 = lift_force_n / self._mass_kg
        return centripetal_acceleration_m_s2

    def get_min_sink_rate_from_bank_angle(self,
                                          bank_angle_earth_to_wind_rad: float,
                                          CL: float, CD: float):

        leveled_min_sink_rate_m_per_s = self.get_leveled_sink_rate(CL, CD)
        if bank_angle_earth_to_wind_rad == np.nan:
            return np.nan
        else:
            # NOTE: "Therefore at shallow glide angles, it's a good approximation to say that the sink rate is proportional to Cd / CL * 1/(Cl^1.5), which works out to Cd / (Cl^1.5), which is the same as (Cd^2 / Cl^3). A derivation of this appears in "Model Aircraft Aerodynamics" by Martin Simons (3rd edition, 1994) on pp. 40-41, or in more detail, on pp. 238-239."
            return leveled_min_sink_rate_m_per_s / (
                np.cos(bank_angle_earth_to_wind_rad)**(3 / 2))

    def get_leveled_sink_rate(self, CL: float, CD: float):

        leveled_horizontal_velocity = self.get_horizontal_velocity(0, CL)

        return CD / CL * leveled_horizontal_velocity

    def get_horizontal_velocity(self, bank_angle_earth_to_wind_rad: float,
                                CL: float):

        wing_load = self._mass_kg / self._wing_area_m2
        result = np.sqrt(
            2 * wing_load * self._g_m_per_s2 /
            (self._rho_kg_per_m3 * CL * np.cos(bank_angle_earth_to_wind_rad)))

        return result
