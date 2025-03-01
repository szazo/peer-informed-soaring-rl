import scipy
import numpy as np
from dataclasses import dataclass


@dataclass
class FilterInstanceParamsBase:
    _target_: str = 'missing'


@dataclass(kw_only=True)
class ExponentialFilterParams(FilterInstanceParamsBase):
    kernel_size: int
    alpha: float
    _target_: str = 'env.glider.base.agent.air_velocity_filter.create_exponential_filter_kernel'


@dataclass(kw_only=True)
class GaussianFilterParams(FilterInstanceParamsBase):
    kernel_size: int
    # used for control that the specified kernel size should contain
    # the most of the bell volume: sigma'=kernel_size/k
    kernel_sigma_k: float = 2.5
    _target_: str = 'env.glider.base.agent.air_velocity_filter.create_gaussian_filter_kernel'


def create_gaussian_filter_kernel(kernel_size: int, kernel_sigma_k: float):
    assert kernel_sigma_k > 0
    gaussian_kernel = scipy.signal.windows.gaussian(kernel_size,
                                                    std=kernel_size /
                                                    kernel_sigma_k)
    gaussian_kernel = gaussian_kernel / \
        np.sum(gaussian_kernel)  # normalize
    return gaussian_kernel


def create_exponential_filter_kernel(kernel_size: int, alpha: float):
    x = np.arange(0, kernel_size)
    w = alpha * (1 - alpha)**(x)
    w = w / np.sum(w)  # normalize
    return w


def create_mean_kernel(kernel_size: int):
    kernel = np.array([1.] * kernel_size) / kernel_size

    return kernel
