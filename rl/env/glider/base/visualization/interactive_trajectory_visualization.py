import pyvista as pv
import numpy as np

from env.glider.base import GliderTrajectory


class InteractiveTrajectoryVisualization:

    _plotter: pv.Plotter
    _line: pv.PolyData | None

    def __init__(self):
        self._plotter = pv.Plotter()
        self._line = None

        self._show_bounds()

    def _show_bounds(self):
        self._plotter.show_bounds(self._plotter,
                                  grid='back',
                                  location='outer',
                                  ticks='both',
                                  n_xlabels=3,
                                  n_ylabels=3,
                                  n_zlabels=3,
                                  xtitle='x (m)',
                                  ytitle='y (m)',
                                  ztitle='z (m)',
                                  bounds=[-1000, 1000, -1000, 1000, 0, 1500])

    def update(self, trajectory: GliderTrajectory):

        if self._plotter._closed:
            return

        points = np.copy(trajectory.position_earth_xyz_m)

        if points.shape[0] > 0:

            if self._line is None:
                # create the line
                self._line = pv.PolyData(points)
                self._line_actor = self._plotter.add_mesh(self._line,
                                                          line_width=3,
                                                          reset_camera=False)
                self._show_bounds()
            else:
                # update the line
                self._line.points = points
                # hacky update (https://github.com/pyvista/pyvista-support/issues/104)
                self._line.verts = self._line._make_vertex_cells(
                    self._line.n_points)

        else:
            if self._line is not None:
                raise Exception('not supported: you cannot remove trajectory')

    def show(self):
        self._plotter.show(interactive_update=True)

    def render(self):
        self._plotter.update()

    def close(self):
        self._plotter.close()
