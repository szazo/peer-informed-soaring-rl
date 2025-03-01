from dataclasses import dataclass
import numpy as np
import h5py
from utils import Vector2D, Vector3D, VectorNxN, VectorNxNxN


@dataclass
class Box:
    range_low_xyz_m: Vector3D
    range_high_xyz_m: Vector3D
    resolution: Vector3D


@dataclass
class Box2D:
    range_low_xy_m: Vector2D
    range_high_xy_m: Vector2D
    resolution_xy_m: Vector2D
    z_m: float


@dataclass
class VerticalThermalPlotData:
    u: VectorNxNxN
    v: VectorNxNxN
    w: VectorNxNxN
    box: Box


@dataclass
class HorizontalThermalPlotData:
    u: VectorNxN
    v: VectorNxN
    w: VectorNxN
    box: Box2D


class ThermalPlotDataSerializer:

    def save(self, horizontal_data: HorizontalThermalPlotData | None,
             vertical_data: VerticalThermalPlotData | None, out_filepath: str):

        with h5py.File(out_filepath, 'w') as f:
            if horizontal_data is not None:
                self.serialize_horizontal(horizontal_data,
                                          target=f.create_group('horizontal'))
            if vertical_data is not None:
                self.serialize_vertical(vertical_data,
                                        target=f.create_group('vertical'))

    def load(self, filepath: str):

        with h5py.File(filepath, 'r') as f:
            # horizontal
            horizontal_group = f['horizontal']
            assert isinstance(horizontal_group, h5py.Group)
            horizontal = self._deserialize_horizontal(horizontal_group)

            # vertical
            vertical_group = f['vertical']
            assert isinstance(vertical_group, h5py.Group)
            vertical = self._deserialize_vertical(vertical_group)

            return horizontal, vertical

    def serialize_vertical(self, data: VerticalThermalPlotData,
                           target: h5py.Group):

        velocity_group = target.create_group('velocity')
        velocity_group.create_dataset('u', data=data.u)
        velocity_group.create_dataset('v', data=data.v)
        velocity_group.create_dataset('w', data=data.w)

        box_group = target.create_group('box')
        box_group.create_dataset('range_low_xyz_m',
                                 data=data.box.range_low_xyz_m)
        box_group.create_dataset('range_high_xyz_m',
                                 data=data.box.range_high_xyz_m)
        box_group.create_dataset('resolution_xyz_m', data=data.box.resolution)

    def _deserialize_vertical(self, source: h5py.Group):
        # read velocities
        velocity_group = source['velocity']
        assert isinstance(velocity_group, h5py.Group)
        u = self._read_dataset_as_nparray(velocity_group, 'u')
        v = self._read_dataset_as_nparray(velocity_group, 'v')
        w = self._read_dataset_as_nparray(velocity_group, 'w')

        # read box
        box_group = source['box']
        assert isinstance(box_group, h5py.Group)
        range_low_xyz_m = self._read_dataset_as_nparray(
            box_group, 'range_low_xyz_m')
        range_high_xyz_m = self._read_dataset_as_nparray(
            box_group, 'range_high_xyz_m')
        resolution_xyz_m = self._read_dataset_as_nparray(
            box_group, 'resolution_xyz_m')

        data = VerticalThermalPlotData(u=u,
                                       v=v,
                                       w=w,
                                       box=Box(
                                           range_low_xyz_m=range_low_xyz_m,
                                           range_high_xyz_m=range_high_xyz_m,
                                           resolution=resolution_xyz_m))

        return data

    def serialize_horizontal(self, data: HorizontalThermalPlotData,
                             target: h5py.Group):

        velocity_group = target.create_group('velocity')
        velocity_group.create_dataset('u', data=data.u)
        velocity_group.create_dataset('v', data=data.v)
        velocity_group.create_dataset('w', data=data.w)

        box_group = target.create_group('box')
        box_group.create_dataset('range_low_xy_m',
                                 data=data.box.range_low_xy_m)
        box_group.create_dataset('range_high_xy_m',
                                 data=data.box.range_high_xy_m)
        box_group.create_dataset('resolution_xy_m',
                                 data=data.box.resolution_xy_m)
        box_group.create_dataset('z_m', data=data.box.z_m)

    def _deserialize_horizontal(self, source: h5py.Group):
        # read velocities
        velocity_group = source['velocity']
        assert isinstance(velocity_group, h5py.Group)
        u = self._read_dataset_as_nparray(velocity_group, 'u')
        v = self._read_dataset_as_nparray(velocity_group, 'v')
        w = self._read_dataset_as_nparray(velocity_group, 'w')

        # read box
        box_group = source['box']
        assert isinstance(box_group, h5py.Group)
        range_low_xy_m = self._read_dataset_as_nparray(box_group,
                                                       'range_low_xy_m')
        range_high_xy_m = self._read_dataset_as_nparray(
            box_group, 'range_high_xy_m')
        resolution_xy_m = self._read_dataset_as_nparray(
            box_group, 'resolution_xy_m')
        z_m = self._read_dataset_as_nparray(box_group, 'z_m').item()

        data = HorizontalThermalPlotData(u=u,
                                         v=v,
                                         w=w,
                                         box=Box2D(
                                             range_low_xy_m=range_low_xy_m,
                                             range_high_xy_m=range_high_xy_m,
                                             resolution_xy_m=resolution_xy_m,
                                             z_m=z_m))

        return data

    def _read_dataset_as_nparray(self, parent: h5py.Group, key: str):
        dataset = parent[key]
        assert isinstance(dataset, h5py.Dataset)

        return np.array(dataset)
