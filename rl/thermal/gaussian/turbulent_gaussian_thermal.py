import logging
import numpy as np
from typing import Tuple, Union
from .noise_field import NoiseField


class TurbulentGaussianThermal:

    x0: float
    y0: float
    max_r: float
    max_r_altitude: float
    max_r_sigma: float
    w_max: float
    noise_field: NoiseField | None
    _is_time_dependent_noise: bool | None

    def __init__(self, x0: float, y0: float, max_r: float,
                 max_r_altitude: float, max_r_sigma: float, w_max: float,
                 noise_field: NoiseField | None,
                 is_time_dependent_noise: bool | None):

        self._log = logging.getLogger(__class__.__name__)

        self.x0 = x0
        self.y0 = y0
        self.max_r = max_r
        self.max_r_altitude = max_r_altitude
        self.max_r_sigma = max_r_sigma
        self.w_max = w_max
        self._noise_field = noise_field
        self._is_time_dependent_noise = is_time_dependent_noise

        self._log.debug(
            'x0=%s,y0=%s,max_r=%s,max_r_altitude=%s,max_r_sigma=%s,w_max=%s',
            x0, y0, max_r, max_r_altitude, max_r_sigma, w_max)

    def generate_field(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        res: Tuple[float, float, float],
    ):

        res_x = res[0]
        res_y = res[1]
        res_z = res[2]

        x, y, z = np.meshgrid(
            np.arange(x_range[0], x_range[1] + 1, res_x, dtype=np.float32),
            np.arange(y_range[0], y_range[1] + 1, res_y, dtype=np.float32),
            np.arange(z_range[0], z_range[1] + 1, res_z, dtype=np.float32),
        )

        u, v, w = self.get_velocity(x, y, z)

        return x, y, z, u, v, w

    def get_velocity(
        self,
        time_s: Union[float, np.ndarray],
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        x0: Union[float, np.ndarray],
        y0: Union[float, np.ndarray],
    ):

        self._log.debug('get_velocity time_s=%s,x=%s,y=%s,z=%s,x0=%s,y0=%s',
                        time_s, x, y, z, x0, y0)

        # TODO: time
        time_s = 0.

        # radius at altitude
        r = self.thermal_radius(z, self.max_r, self.max_r_altitude,
                                self.max_r_sigma)
        # vertical lift
        w = self.gaussian_lift(x, y, r, r, self.w_max, x0 + self.x0,
                               y0 + self.y0)

        u = np.zeros(w.shape)
        v = np.zeros(w.shape)

        if self._noise_field is not None:
            # turbulence
            noise_x = self._get_noise(noise_index=0,
                                      time_s=time_s,
                                      x=x,
                                      y=y,
                                      z=z)
            noise_y = self._get_noise(noise_index=1,
                                      time_s=time_s,
                                      x=x,
                                      y=y,
                                      z=z)
            noise_z = self._get_noise(noise_index=2,
                                      time_s=time_s,
                                      x=x,
                                      y=y,
                                      z=z)

            w_with_noise = w + (noise_z)

            u_with_noise = u + w * noise_x
            v_with_noise = v + w * noise_y

            u_with_noise = u_with_noise if u_with_noise.size > 1 else u_with_noise.item(
            )
            v_with_noise = v_with_noise if v_with_noise.size > 1 else v_with_noise.item(
            )
            w_with_noise = w_with_noise if w_with_noise.size > 1 else w_with_noise.item(
            )

            return u_with_noise, v_with_noise, w_with_noise
        else:
            return u, v, w

    def _get_noise(self, noise_index: int, time_s: Union[float, np.ndarray],
                   x: Union[float, np.ndarray], y: Union[float, np.ndarray],
                   z: Union[float, np.ndarray]):

        assert self._noise_field is not None, 'missing _noise_field'

        has_scalar_coord = np.isscalar(x) or np.isscalar(y) or np.isscalar(
            z) or np.isscalar(time_s)
        if has_scalar_coord:
            coordinates = (np.asarray(x), np.asarray(y), np.asarray(z))
            if self._is_time_dependent_noise:
                coordinates = (np.asarray(time_s), *coordinates)

            return self._noise_field.get_interpolated_noise(
                noise_index=noise_index, coordinates=coordinates)
        else:
            raise Exception('coding error: missing coordinates')

    def thermal_radius(self, altitude: float, max_r: float,
                       max_r_altitude: float, max_r_sigma: float):
        """Calculate thermal radius based on altitude (m) with
            Gaussian distribution.

        Parameters
        ----------
        altitude: float
            The altitude we find radius for (can be vector)
        max_r: float
            The maximum of the r.
        max_r_altitude: float
            The altitude where the r will be the maximum.
        max_r_sigma: float
            The width of the distribution.

        Returns
        -------
        float
            The radius at altitude.
        """
        a = max_r
        b = max_r_altitude
        c = max_r_sigma

        r = a * np.exp(-((altitude - b)**2) / (2 * c**2))

        return r

    def gaussian_lift(
        self,
        x: float,
        y: float,
        r_x: float,
        r_y: float,
        amplitude: float,
        x0: float,
        y0: float,
    ):
        """Generates gaussian lift for 2d input

        Parameters
        ----------
        x, y : float
            The x and y coordinates
        r_x, r_y: float
            The x and y distribution (radius) of the lift
        amplitude:
            The maximum of the lift
        x0, y0:
            Coordinates of the maximum of the lift

        Returns
        -------
        float
            The lift for specific (x,y) point/vector.
        """

        a = 1 / (2 * r_x**2)
        c = 1 / (2 * r_y**2)

        if not np.isscalar(x0) and x0.ndim < x.ndim:
            x0 = np.broadcast_to(
                np.expand_dims(x0, axis=[d for d in range(1, x.ndim)]),
                x.shape)
        if not np.isscalar(y0) and y0.ndim < y.ndim:
            y0 = np.broadcast_to(
                np.expand_dims(y0[np.newaxis, :],
                               axis=[d for d in range(2, y.ndim)]), y.shape)

        f_xy = amplitude * np.exp(-(a * (x - x0)**2 + c * (y - y0)**2))

        return f_xy
