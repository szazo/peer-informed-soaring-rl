import math
import sys
import numpy.typing as npt
import numpy as np


# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm
class RunningStats:

    _n: int
    _mean: float
    _sum_of_squares_of_diffs: float
    _min: float
    _max: float

    def __init__(self):
        self._n = 0
        self._mean = 0.0
        self._sum_of_squares_of_diffs = 0.0
        self._min = sys.float_info.max
        self._max = sys.float_info.min

    def push(self, x: float | list[float] | npt.NDArray[np.float_]):

        values = x if isinstance(x, (list, np.ndarray)) else [x]

        for value in values:
            self._n += 1
            if self._n == 1:
                self._mean = value
                self._sum_of_squares_of_diffs = 0.0
            else:
                new_mean = self._mean + (value - self._mean) / self._n
                self._sum_of_squares_of_diffs = (
                    self._sum_of_squares_of_diffs + (value - self._mean) *
                    (value - new_mean))
                self._mean = new_mean

            if value < self._min:
                self._min = value

            if value > self._max:
                self._max = value

    def min(self):
        return self._min

    def max(self):
        return self._max

    def mean(self):
        return self._mean

    def var(self):
        return (self._sum_of_squares_of_diffs /
                (self._n) if self._n > 1 else 0.0)

    def std(self):
        return math.sqrt(self.var())
