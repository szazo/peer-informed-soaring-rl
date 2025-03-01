from typing import Sequence
from functools import reduce
import numpy as np


def create_and_stack_sample_items(shape: Sequence[int], item_axis: int,
                                  count: int):
    items = create_sample_items(shape=shape, count=count)
    result = np.stack(items, axis=item_axis)

    return result


def create_sample_items(shape: Sequence[int], count: int):

    items = []

    item_size = reduce(lambda x, y: x * y, shape)
    for i in range(count):
        item = (np.arange(item_size) + float(i * item_size))
        item = item.reshape(shape)

        items.append(item)

    return items


def create_sample_items_with_shapes(*shapes: Sequence[int]):

    items = []

    offset = 0
    for shape in shapes:
        item = create_sample_item(shape, offset=offset)
        items.append(item)

        offset += item.size

    return items


def create_sample_item(shape: Sequence[int], offset: int = 0):
    item_size = reduce(lambda x, y: x * y, shape)
    item = (np.arange(item_size) + float(offset))
    item = item.reshape(shape)
    return item
