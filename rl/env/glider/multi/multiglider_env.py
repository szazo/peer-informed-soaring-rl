from .make_multiglider_env import make_multiglider_env

# https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/atari/base_atari_env.py
# https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/atari/base_atari_env.py


def create_env_constructor():
    def env_fn(**kwargs):
        env = make_multiglider_env(env_name='multiglider_v0', **kwargs)
        return env

    return env_fn


env = create_env_constructor()
