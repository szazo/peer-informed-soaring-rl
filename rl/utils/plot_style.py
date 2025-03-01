import scienceplots
import matplotlib.pyplot as plt
import contextlib


@contextlib.contextmanager
def plot_style(additional_rc_params: dict = {}):

    additional_rc_params = {
        'font.family': 'sans-serif',
        'font.size': 7.0,
        'svg.fonttype': 'path',
        **additional_rc_params
    }

    with plt.style.context(['science', additional_rc_params]):
        # plt.rcParams['errorbar.capsize'] = 5

        # for key, value in plt.rcParams.items():
        #     print(f'{key}: {value}')

        yield
