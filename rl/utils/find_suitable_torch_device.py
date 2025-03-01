import torch
import logging
from torch.types import Device


def find_suitable_torch_device(device_str: str) -> torch.device:

    log = logging.getLogger(__name__)
    log.debug('find_suitable_torch_device; device_str=%s', device_str)

    if device_str == 'cpu':
        log.debug('using cpu')
        return torch.device(device_str)

    if device_str == 'max_memory_cuda':
        # find suitable gpu
        log.debug('finding suitable gpu based on memory')
        max_memory = 0
        best_cuda_index = -1
        best_name = None
        for i in range(torch.cuda.device_count()):
            # wait for it to finish its job if any
            torch.cuda.synchronize(i)

            # query memory
            properties = torch.cuda.get_device_properties(i)
            total_memory = properties.total_memory
            if total_memory > max_memory:
                max_memory = total_memory
                best_cuda_index = i
                best_name = properties.name

        if best_cuda_index == -1:
            log.debug('no cuda device found, using cpu')
            return torch.device('cpu')

        device = torch.device(f'cuda:{best_cuda_index}')
        log.debug('found best cuda device "%s" (%s) with memory %.0fGB',
                  best_name, device, max_memory / 1024 / 1024 / 1024)

        return device

    return torch.device(device_str)
