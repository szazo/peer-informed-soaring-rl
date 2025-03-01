import numpy as np

from utils.random_state import deserialize_random_state, serialize_random_state


def test_serialize_deserialize_return_the_same_state():

    # given
    seed = np.uint64(424328759829)
    pcg = np.random.PCG64(seed)
    rng = np.random.Generator(pcg)
    rng.bytes(64 * 10000)  # some mixing

    random_numbers = rng.uniform(0., 1., 1000)

    # when
    state = serialize_random_state(generator=rng)

    # then
    restored_rng = deserialize_random_state(state)

    random_numbers_restored = restored_rng.uniform(0., 1., 1000)

    np.allclose(random_numbers, random_numbers_restored)
