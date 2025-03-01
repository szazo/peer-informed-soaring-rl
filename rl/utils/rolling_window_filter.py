from typing import Self
import numpy as np
import numpy.typing as npt
from copy import deepcopy


class RollingWindowFilter:

    _kernel: npt.NDArray[np.float64]

    _buffer: npt.NDArray[np.float64] = np.array([])

    def __init__(self, kernel: npt.NDArray[np.float64]):
        self._kernel = kernel

    def clone_state_from(self, other: Self):
        self._buffer = deepcopy(other._buffer)

    def reset(self):
        self._buffer = np.array([])

    def feed(self, value: float):

        kernel_size = self._kernel.shape[0]

        # append to the end and use only the last 'kernel_size' elements
        self._buffer = np.append(self._buffer, value)[-kernel_size:]

        # prepare the buffer we will use for the convolution
        work_buffer = self._buffer
        if work_buffer.shape[0] < kernel_size:
            # fill with the mean, if we have less data than the kernel size
            mean_values = np.full(shape=kernel_size - work_buffer.shape[0],
                                  fill_value=np.mean(self._buffer))
            work_buffer = np.append(mean_values, work_buffer)
        assert work_buffer.size == kernel_size

        # calculate only the overlapping region
        convolved = np.convolve(work_buffer, self._kernel, mode='valid')

        # return the last element of the convolved result
        return convolved[-1]
