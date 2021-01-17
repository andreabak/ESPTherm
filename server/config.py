"""Configuration classes and globals"""

import json
import os
import warnings
from typing import MutableMapping, Iterator, Any, Tuple

from .base import LiveFileData, JSONETDDecoder, JSONETDEncoder, DELETED, EditTrackingDict


__all__ = [
    "LOGS_PATH",
    "LOGS_EXT",
    "LOGS_DB_PATH",
    "LOGS_DB_METADATA",
    "LOGS_ROTATE_SIZE",
    "CONFIGS_PATH",
    "CONFIGS_EXT",
    "SERVER_CONFIG_BASENAME",
    "DEFAULT_CONFIG_BASENAME",
    "ServerConfig",
    "server_config"
]


# ### GLOBAL CONFIGS ### #
# TODO: some of these could be moved to the server's config file,
#       but they're expected not to change often and it's low priority for now.

LOGS_PATH: str = 'logs'
"""Directory path for log files"""

LOGS_EXT: str = '.log'
"""File extension of log files"""

LOGS_DB_PATH: str = os.path.join(LOGS_PATH, 'database.sqlite')
"""Path of the logs database"""

LOGS_DB_METADATA: str = '_devices'
"""Metadata table name within the logs database"""

LOGS_ROTATE_SIZE: int = 4 * 1024 * 1024  # 4 MB
"""Maximum log file in bytes before it's rotated"""

CONFIGS_PATH: str = 'configs'
"""Directory path for config files"""

CONFIGS_EXT: str = '.json'
"""File extension of config files"""

SERVER_CONFIG_BASENAME: str = 'server'
"""Server's config file base name"""

DEFAULT_CONFIG_BASENAME: str = '_default'
"""Device-type default config basename"""


class ServerConfig(LiveFileData, MutableMapping[str, Any]):
    """
    Class that holds server config data.
    Acts like a dict, and it's backed by default by the server's config file.
    """
    @classmethod
    def file_path(cls, *args, **kwargs) -> str:
        return os.path.join(CONFIGS_PATH, SERVER_CONFIG_BASENAME + CONFIGS_EXT)

    @classmethod
    def _load_file(cls) -> Tuple[EditTrackingDict[str, Any], str, float]:
        """
        Class method that loads data from the server's config file
        :return: a tuple comprised of (data, file_path, file_mtime), where:
                 "data" is an instance of `EditTrackingDict`,
                 "file_path" is the config's file path,
                 "file_mtime" is the mtime timestamp of the config file at load time.
        """
        file_path: str = cls.file_path()
        with open(file_path, 'r') as fp:
            file_mtime: float = cls.file_mtime(path=file_path)
            data: EditTrackingDict[str, Any] = json.load(fp, cls=JSONETDDecoder)
        return data, file_path, file_mtime

    def __init__(self, *args, **kwargs):
        """
        Initialization for `ServerConfig`
        :param args: additional positional arguments to be passed to `super().__init__(...)`
        :param kwargs: additional keyword arguments to be passed to `super().__init__(...)`
        """
        self._data: EditTrackingDict[str, Any] = EditTrackingDict()
        file_path: str
        file_mtime: float
        file_path, file_mtime = self.load()
        super().__init__(*args, loaded_path=file_path, loaded_mtime=file_mtime, **kwargs)

    def load(self) -> Tuple[str, float]:
        """
        Loads data from the server's config file into the instance.
        :return: a tuple of (file_path, file_mtime), where:
                 "file_path" is the config's file path,
                 "file_mtime" is the mtime timestamp of the config file at load time.
        """
        data: EditTrackingDict[str, Any]
        file_path: str
        file_mtime: float
        data, file_path, file_mtime = self._load_file()
        self._data = data
        self._loaded_mtime = file_mtime
        return file_path, file_mtime

    def save(self, force: bool = False) -> None:
        """
        Saves the data stored in the instance to the server's config file.
        :param force: will force save the data stored in the instance, ignoring warnings.
        :raise RuntimeWarning: if trying to save modified data but the loaded data of the instance
                               is not up-to-date anymore with the config's file and `force` is not True.
        """
        if not force and not self.uptodate and self._data.modified:
            raise RuntimeWarning(f'Not saving {self.__class__.__name__} because source file has been updated since. '
                                 f'Use `.merge()` or pass `force=True` instead.')
        file_path: str = self.file_path()
        with open(file_path, 'w') as fp:
            json.dump(self._data, fp, cls=JSONETDEncoder)
            fp.flush()
        file_mtime: float = self.file_mtime(path=file_path)
        self._loaded_mtime = file_mtime

    def merge(self) -> None:
        """
        Saves the data stored in the instance, reloading the server's config file first
        and merging the edited values together, with the latter taking precedence.
        """
        file_data: EditTrackingDict[str, Any]
        file_data, _, _ = self._load_file()
        for k, v in self._data.iteredits():
            if v is DELETED:
                del file_data[k]
            else:
                file_data[k] = v
        file_data.reset_edits()
        self._data = file_data
        self.save(force=True)

    def ensure_uptodate(self) -> None:
        """
        Ensures the loaded data is up-to-date with the file.
        Will not load from file and print a warning instead
        if the stored data has been modified and is unsaved.
        """
        if not self.uptodate:
            if self._data.modified:
                warnings.warn(f'Not reloading {self.__class__.__name__} from file because there\'s unsaved data',
                              RuntimeWarning)
            else:
                self.load()

    def __setitem__(self, k: str, v: Any) -> None:
        self.ensure_uptodate()
        self._data[k] = v

    def __delitem__(self, k: str) -> None:
        self.ensure_uptodate()
        del self._data[k]

    def __getitem__(self, k: str) -> Any:
        self.ensure_uptodate()
        return self._data[k]

    def __len__(self) -> int:
        self.ensure_uptodate()
        return len(self._data)

    def __iter__(self) -> Iterator[Any]:
        self.ensure_uptodate()
        return iter(self._data)


server_config: ServerConfig = ServerConfig()
"""Global `ServerConfig` instance"""
