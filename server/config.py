import json
import os
import warnings
from typing import MutableMapping, Iterator, Any, Tuple

from .base import LiveFileData, JSONETDDecoder, JSONETDEncoder, DELETED, EditTrackingDict


LOGS_PATH: str = 'logs'
LOGS_EXT: str = '.log'
LOGS_DB_PATH: str = os.path.join(LOGS_PATH, 'database.sqlite')
LOGS_DB_METADATA: str = '_devices'
LOGS_ROTATE_SIZE: int = 4 * 1024 * 1024  # 4 MB
CONFIGS_PATH: str = 'configs'
CONFIGS_EXT: str = '.json'
SERVER_CONFIG_BASENAME: str = 'server'
DEFAULT_CONFIG_BASENAME: str = '_default'


class ServerConfig(LiveFileData, MutableMapping[str, Any]):
    @classmethod
    def file_path(cls, *args, **kwargs) -> str:
        return os.path.join(CONFIGS_PATH, SERVER_CONFIG_BASENAME + CONFIGS_EXT)

    @classmethod
    def _load_file(cls) -> Tuple[EditTrackingDict[str, Any], str, float]:
        file_path: str = cls.file_path()
        with open(file_path, 'r') as fp:
            file_mtime: float = cls.file_mtime(file_path)
            data: EditTrackingDict[str, Any] = json.load(fp, cls=JSONETDDecoder)
        return data, file_path, file_mtime

    def __init__(self, *args, **kwargs):
        self._data: EditTrackingDict[str, Any] = EditTrackingDict()
        file_path: str
        file_mtime: float
        file_path, file_mtime = self.load()
        super().__init__(*args, loaded_path=file_path, loaded_mtime=file_mtime, **kwargs)

    def load(self) -> Tuple[str, float]:
        data: EditTrackingDict[str, Any]
        file_path: str
        file_mtime: float
        data, file_path, file_mtime = self._load_file()
        self._data = data
        self._loaded_mtime = file_mtime
        return file_path, file_mtime

    def save(self, force: bool = False) -> None:
        if not force and not self.uptodate and self._data.modified:
            raise RuntimeWarning(f'Not saving {self.__class__.__name__} because source file has been updated since. '
                                 f'Use `.merge()` or pass `force=True` instead.')
        file_path: str = self.file_path()
        with open(file_path, 'w') as fp:
            json.dump(self._data, fp, cls=JSONETDEncoder)
            fp.flush()
        file_mtime: float = self.file_mtime(file_path)
        self._loaded_mtime = file_mtime

    def merge(self) -> None:
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
