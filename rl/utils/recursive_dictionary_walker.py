# generator which recursively walks a dictionary and return ([keys], value) collection
def recursive_dictionary_walker(dictionary: dict,
                                current_path: list[str] | None = None):
    current_path = current_path[:] if current_path else []

    for key, value in dictionary.items():
        next_path = current_path + [key]

        if isinstance(value, dict):
            yield from recursive_dictionary_walker(dictionary=value,
                                                   current_path=next_path)
        elif isinstance(value, list) or isinstance(value, tuple):
            for item in value:
                yield (next_path, item)
        else:
            yield (next_path, value)
