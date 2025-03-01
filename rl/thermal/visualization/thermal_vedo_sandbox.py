from typing import Any
from dataclasses import dataclass
import logging
from omegaconf import MISSING
import numpy as np
import pandas as pd
from gymnasium.utils import seeding
from thermal.api import AirVelocityFieldInterface
from utils.custom_job_api import CustomJobBase, CustomJobBaseConfig
from vedo import Points, show, Arrows, mag
import vedo


@dataclass
class StreamlinesParams:
    seed_count: int
    seed_radius_m: float


@dataclass(kw_only=True)
class ThermalVedoSandboxJobConfig(CustomJobBaseConfig):
    streamlines: StreamlinesParams
    air_velocity_field: Any = MISSING

    _target_: str = 'thermal.visualization.thermal_vedo_sandbox.ThermalVedoSandbox'


class ThermalVedoSandbox(CustomJobBase):

    _log: logging.Logger

    _streamlines_params: StreamlinesParams

    _air_velocity_field: AirVelocityFieldInterface
    _np_random: np.random.Generator

    def __init__(self,
                 streamlines: StreamlinesParams,
                 air_velocity_field: AirVelocityFieldInterface,
                 seed: int | None = None):
        self._log = logging.getLogger(__class__.__name__)

        self._log.debug('initialized; air_velocity_field=%s',
                        air_velocity_field)

        self._streamlines_params = streamlines
        self._air_velocity_field = air_velocity_field

        print(air_velocity_field)

        self._np_random, seed = seeding.np_random(seed)

    def run(self, output_dir: str):

        self._air_velocity_field.reset()

        # data = "https://raw.githubusercontent.com/plotly/datasets/master/vortex.csv"
        # df = pd.read_csv(data)

        x = np.linspace(-1000, 1000, num=150)
        y = np.linspace(-1000, 1000, num=150)
        z = np.linspace(0, 1500, num=200)

        X, Y, Z = np.meshgrid(x, y, z)

        coords = [item.ravel() for item in (X, Y, Z)]
        x = coords[0]
        y = coords[1]
        z = coords[2]

        # seed_points
        seeds_xyz_m = self._generate_random_seeds()
        print('BEFORE', x.shape, seeds_xyz_m[:, 0].shape)
        x = np.concatenate((x, seeds_xyz_m[:, 0]))
        y = np.concatenate((y, seeds_xyz_m[:, 1]))
        z = np.concatenate((z, seeds_xyz_m[:, 2]))
        print('AFTER', x.shape)

        # = Points(xyz_m)
        # print('SEEDS2', seeds)

        # print(xyz_m)
        # return seeds

        print('X shape', X.shape, coords[0].shape)

        uvw, _ = self._air_velocity_field.get_velocity(x_earth_m=x,
                                                       y_earth_m=y,
                                                       z_earth_m=z,
                                                       t_s=0)
        u = uvw[0]
        v = uvw[1]
        w = uvw[2]

        mask = w > -1000.
        print(mask)
        print('shape', uvw.shape, u.shape, v.shape, w.shape)
        print(w)
        x = x[mask]
        y = y[mask]
        z = z[mask]
        u = u[mask]
        v = v[mask]
        w = w[mask]

        pts = np.c_[x, y, z]
        wind = np.c_[u, v, w]
        # pts = np.c_[df['x'], df['y'], df['z']]
        # wind = np.c_[df['u'], df['v'], df['w']]

        print(pts.shape)
        print(wind.shape)

        # vpts = Points(pts, r=10)
        vpts = Points(pts, r=30)
        vpts.pointdata['Wind'] = wind
        vpts.pointdata['scals'] = w

        print('min max', np.min(pts), np.max(pts))
        # converts points to a volume to create a domain for the streamlines
        vol = vpts.tovolume(kernel='shepard', n=4, dims=(150, 150, 200))
        # vol.pointdata.select('Wind')
        # vol.cmap('b').mode(1)

        vol.cmap('jet')
        vol.alpha([(0., 0), (0.5, 0.2), (1.0, 0.9), (2.0, 0.95)])
        vol.alpha_unit(500)

        # vol_pts = vol.coordinates
        # print('vol_pts', vol_pts.shape)
        # iwind = vol.pointdata['Wind']
        # arrs = Arrows(vol_pts, vol_pts + iwind, alpha=0.2)

        print(vpts)

        seeds = Points(seeds_xyz_m).color('g')

        #        seeds = vpts.clone().subsample(0.05).color('g')
        # print('seeds', seeds.coordinates)
        # print('SEEDS', seeds)

        streamlines = vol.compute_streamlines(seeds.vertices, direction='both')

        streamlines.pointdata["wind_intensity"] = mag(
            streamlines.pointdata["Wind"])
        streamlines.cmap("Reds").add_scalarbar()

        #vol.add_scalarbar('vz')

        plane = vedo.Mesh(vedo.dataurl + 'cessna.vtk')
        plane.color('green')
        plane.scale(5.)

        plane.pos(0, 0, 400.)
        flagpost = plane.flagpost(
            f"Heigth:\nz={42}m",
            offset=(0., 0., 30.),
            alpha=0.5,
            c=(0, 0, 255),
            bc=(255, 0, 255),
            lw=1,
            vspacing=1.2,
            s=0.8,
            font='VictorMono',
        )

        show(vol,
             seeds,
             streamlines,
             plane,
             flagpost,
             __doc__,
             axes=1,
             viewup='z').close()
        # print(output_dir)

    def _generate_random_seeds(self):

        xy_m = self._generate_random_points_in_circle(
            R=self._streamlines_params.seed_radius_m,
            n=self._streamlines_params.seed_count)

        z = np.zeros(xy_m.shape[0])

        print('xy_m.shape', xy_m.shape)
        print('z.shape', z.reshape(-1, 1).shape)

        xyz_m = np.hstack((xy_m, z.reshape(-1, 1)))

        return xyz_m

    def _generate_random_points_in_circle(self, R: float, n: int):

        xy = np.zeros((n, 2))

        for i in range(n):

            x, y = self._generate_random_point_in_circle(R)

            xy[i, 0] = x
            xy[i, 1] = y

        return xy

    def _generate_random_point_in_circle(self, R: float):

        # r^2 should be uniform, so we use sqrt of the uniform to generate r
        r = R * np.sqrt(np.random.uniform())

        # the theta can be uniform
        theta = 2 * np.pi * np.random.uniform()

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return x, y
