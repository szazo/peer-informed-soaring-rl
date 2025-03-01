from dataclasses import dataclass
import base64
import pickle
import numpy as np


@dataclass
class RandomGeneratorState:
    generator_name: str
    encoded_state_values: str


def serialize_random_state(generator: np.random.Generator):
    generator_data = generator.bit_generator.state
    generator_name = generator_data['bit_generator']

    state_values = [
        generator_data['has_uint32'], generator_data['uinteger'],
        generator_data['state']['state'], generator_data['state']['inc']
    ]

    encoded_state_values = base64.urlsafe_b64encode(
        pickle.dumps(state_values)).decode('ascii')

    return RandomGeneratorState(generator_name=generator_name,
                                encoded_state_values=encoded_state_values)


def deserialize_random_state(state: RandomGeneratorState):

    assert state.generator_name == 'PCG64', 'only PCG64 state supported now'

    # decode the values
    state_values = pickle.loads(
        base64.urlsafe_b64decode(state.encoded_state_values.encode('ascii')))

    state_dict = {
        'bit_generator': state.generator_name,
        'has_uint32': state_values[0],
        'uinteger': state_values[1],
        'state': {
            'state': state_values[2],
            'inc': state_values[3]
        }
    }

    bit_generator = np.random.PCG64()
    generator = np.random.Generator(bit_generator)

    generator.bit_generator.state = state_dict

    return generator
