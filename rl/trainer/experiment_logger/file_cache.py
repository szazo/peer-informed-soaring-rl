import os
import shutil
import logging


class FileCache:

    _log: logging.Logger
    _file_path_cache: dict[str, str]

    def __init__(self):
        self._file_path_cache = {}
        self._log = logging.getLogger(__class__.__name__)

    def set(self, key: str, path: str):
        self._log.debug('set; key=%s,path=%s', key, path)

        self._file_path_cache[key] = path

    def get(self, key: str, destination_path: str) -> bool:
        self._log.debug('get; key=%s,destination_path=%s', key,
                        destination_path)

        if key in self._file_path_cache:
            path = self._file_path_cache[key]
            try:
                os.link(path, destination_path)
            except OSError:
                self._log.debug('os.link error, file has been copied')
                shutil.copyfile(path, destination_path)
            self._log.debug('file found with key=%s', key)

            return True

        self._log.debug('file not found with key=%s', key)
        return False
