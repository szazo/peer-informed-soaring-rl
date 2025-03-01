import tempfile
from deepdiff import DeepDiff

from .create_random_trajectory import create_random_trajectory
from env.glider.base.agent.glider_trajectory_serializer import GliderTrajectorySerializer


def test_serialize_should_create_h5py_file():

    # given
    step_count = 11
    trajectory = create_random_trajectory(step_count)

    serializer = GliderTrajectorySerializer()

    # when
    _, filepath = tempfile.mkstemp(suffix='.h5')
    serializer.save(trajectory, filepath)

    # then
    loaded = serializer.load(filepath)
    deep_diff = DeepDiff(trajectory, loaded)
    assert deep_diff == {}
