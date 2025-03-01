import numpy as np


# convert all numpy specific type to pure python (e.g. np.float64 to float)
def cleanup_numpy_from_dictionary(dictionary: dict):

    result = {}
    for key, value in dictionary.items():

        processed_value = None

        if isinstance(value, dict):
            processed_value = cleanup_numpy_from_dictionary(value)
        elif np.isscalar(value):
            processed_value = np.array(value).item()
        elif isinstance(value, np.ndarray):
            processed_value = value.tolist()
        elif isinstance(value, list):
            processed_value = [
                item.item() if isinstance(item, np.generic) else item
                for item in value
            ]
        else:
            processed_value = value

        result[key] = processed_value

    return result
