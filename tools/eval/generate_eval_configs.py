import os
import argparse
import numpy as np
from ruamel.yaml import YAML

from utils.random_state import serialize_random_state

DEFAULT_BASE_DIR = os.path.join(os.path.dirname(__file__),
                                '../../config/eval/gaussian')

AIR_VELOCITY_FIELD_OUT_DIR = os.path.join(DEFAULT_BASE_DIR,
                                          'air_velocity_fields')
POLICY_OUT_DIR = os.path.join(DEFAULT_BASE_DIR, 'models/policy')
MULTI_AGENT_POLICY_OUT_DIR = os.path.join(DEFAULT_BASE_DIR,
                                          'models/multi_agent_policy')
DEFAULT_SEED = 42
DEFAULT_COUNT = 10


def create_air_velocity_field_config(seed: int, index: int, out_filepath: str):

    # seeds for the different air velocity fields will be independent (spawn_key)
    seed_seq = np.random.SeedSequence(seed, spawn_key=[index])
    generator = np.random.Generator(np.random.PCG64(seed_seq))

    random_state = serialize_random_state(generator)

    print(
        f'generating air velocity field with random state {random_state.encoded_state_values} in directory {out_filepath}...'
    )

    defaults = [{
        'override /env/glider/air_velocity_field@_here_':
        'multi_agent_thermal_wind',
    }, '_self_']

    air_velocity_field = {
        'random_state': {
            'generator_name': random_state.generator_name,
            'encoded_state_values': random_state.encoded_state_values
        }
    }

    config = {'defaults': defaults, **air_velocity_field}

    create_yaml(config, out_filepath, insert_package_global=True)


def create_policy_eval_config(experiment_name: str, air_velocity_name: str,
                              aerodynamics_name: str, out_filepath: str):

    print(f'generating POLICY config {out_filepath}...')

    defaults = [{
        '/eval/gaussian/initial_conditions@evaluators.main.env.env.params':
        'gaussian_thermal_initial_conditions'
    }, {
        '/eval/gaussian/air_velocity_fields@evaluators.main.env.env.air_velocity_field':
        air_velocity_name
    }, '_self_']

    self_config = {
        'experiment_name': experiment_name,
        'evaluators': {
            'main': {
                'observation_logger': {
                    'additional_columns': {
                        'thermal': air_velocity_name,
                        'bird_name': aerodynamics_name
                    }
                }
            }
        }
    }

    config = {'defaults': defaults, **self_config}

    create_yaml(config, out_filepath, insert_package_global=True)


def create_multi_agent_policy_eval_config(experiment_name: str,
                                          air_velocity_name: str,
                                          aerodynamics_name: str,
                                          out_filepath: str):

    print(f'generating POLICY config {out_filepath}...')

    defaults = [{
        '/eval/gaussian/air_velocity_fields@evaluators.main.env.env.air_velocity_field':
        air_velocity_name
    }, '_self_']

    self_config = {
        'experiment_name': experiment_name,
        'evaluators': {
            'main': {
                'observation_logger': {
                    'additional_columns': {
                        'thermal': air_velocity_name,
                        'bird_name': aerodynamics_name
                    }
                }
            }
        }
    }

    config = {'defaults': defaults, **self_config}

    create_yaml(config, out_filepath, insert_package_global=True)


def create_yaml(config: dict, out_filepath: str, insert_package_global: bool):
    os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 4096  # prevent line break
    with open(out_filepath, 'w') as f:
        if insert_package_global:
            f.write('# @package _global_\n')
        yaml.dump(config, f)


def generate(count: int, seed: int):

    print(
        f'generating {count} air velocity fields and policy evaluation configs...'
    )

    aerodynamics_name = 'train_glider'

    for i in range(count):

        air_velocity_name = f'eval{i}'

        # air velocity field
        output_filepath = os.path.abspath(
            os.path.join(AIR_VELOCITY_FIELD_OUT_DIR,
                         f'{air_velocity_name}.yaml'))
        create_air_velocity_field_config(seed=seed,
                                         index=i,
                                         out_filepath=output_filepath)

        # policy
        experiment_name = f'policy_eval{i}'
        output_filepath = os.path.abspath(
            os.path.join(POLICY_OUT_DIR, f'{experiment_name}.yaml'))
        create_policy_eval_config(experiment_name=experiment_name,
                                  air_velocity_name=air_velocity_name,
                                  aerodynamics_name=aerodynamics_name,
                                  out_filepath=output_filepath)

        # multi agent policy
        experiment_name = f'multi_agent_policy_eval{i}'
        output_filepath = os.path.abspath(
            os.path.join(MULTI_AGENT_POLICY_OUT_DIR,
                         f'{experiment_name}.yaml'))
        create_multi_agent_policy_eval_config(
            experiment_name=experiment_name,
            air_velocity_name=air_velocity_name,
            aerodynamics_name=aerodynamics_name,
            out_filepath=output_filepath)


parser = argparse.ArgumentParser(
    description='The tool generates thermal seeds for comparison evaluation.')
parser.add_argument('--seed',
                    type=int,
                    default=DEFAULT_SEED,
                    help='the initial seed for the generation')
parser.add_argument('--count',
                    type=int,
                    default=DEFAULT_COUNT,
                    help='the number of air velocity fields to generate')
args = parser.parse_args()

generate(count=args.count, seed=args.seed)
