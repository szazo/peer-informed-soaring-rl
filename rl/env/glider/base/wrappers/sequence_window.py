import logging

import numpy as np


class SequenceWindow:

    _max_sequence_length: int
    _window: np.ndarray | None

    def __init__(self, max_sequence_length: int):
        super().__init__()

        assert max_sequence_length > 0, 'please set max_sequence_length'

        self._log = logging.getLogger(__class__.__name__)

        self._window = None
        self._max_sequence_length = max_sequence_length

    def reset(self):
        self._window = None

    def add(self, item: np.ndarray):

        # create dimension for the row
        item = np.expand_dims(item, axis=0)

        # add the item at the end of the stack
        window = self._window
        self._log.debug("stack before; stack=%s", window)

        if window is None:
            window = item
        else:
            # insert at the end
            window = np.vstack((window, item))

        # use only the last max_sequence_length items
        window = window[-self._max_sequence_length:]

        # save the stack before the post process
        self._window = window

        return np.copy(window)

    @property
    def window(self):
        return self._window
