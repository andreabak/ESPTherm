import itertools
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, Any, MutableMapping, TypeVar, Iterator, Tuple, Union, Sequence, Iterable, List

ST = TypeVar('ST')


def chunked(it: Iterable[ST], n: int) -> Iterator[Sequence[ST]]:
    it = iter(it)
    while True:
        chunk: List[ST] = list(itertools.islice(it, n))
        if not chunk:
            break
        yield chunk


def ensure_tz(dt: datetime, default_tz: Optional[timezone] = timezone.utc) -> datetime:
    return dt.replace(tzinfo=default_tz) if dt.tzinfo is None else dt


class LiveFileData(ABC):
    @classmethod
    @abstractmethod
    def file_path(cls, *args, **kwargs) -> str: ...

    @classmethod
    def file_mtime(cls, path: Optional[str] = None, *file_path_args, **file_path_kwargs) -> float:
        path_arg_valid = path is not None
        path_gen_args_valid = all((file_path_args, file_path_kwargs))
        if not path_arg_valid != path_gen_args_valid:
            raise AttributeError('Must specify either "path" or "file_path_args" / "file_path_kwargs" arguments')
        if path is None:
            path = cls.file_path(*file_path_args, **file_path_kwargs)
        return os.path.getmtime(path)

    def __init__(self, loaded_path: str, loaded_mtime: Optional[float] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loaded_path: str = loaded_path
        if loaded_mtime is None:
            loaded_mtime = self.file_mtime(path=self._loaded_path)
        self._loaded_mtime: float = loaded_mtime

    @property
    def loaded_mtime(self) -> float:
        return self._loaded_mtime

    @property
    def uptodate(self) -> bool:
        return self.loaded_mtime == self.file_mtime(path=self._loaded_path)


class UniqueHashable(ABC):
    @property
    @abstractmethod
    def unique_id(self) -> Any: ...

    def __init__(self, freeze: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frozen: bool = freeze

    def freeze(self):
        self._frozen = True

    def __setitem__(self, key, value):
        if self._frozen:
            raise PermissionError('Cannot set items to frozen instance')
        # noinspection PyUnresolvedReferences
        super().__setitem__(key, value)

    def __hash__(self) -> int:
        return hash((self.__class__, self.unique_id))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.unique_id == other.unique_id


class JSONETDDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('object_hook', self.object_hook)
        super().__init__(*args, **kwargs)

    @staticmethod
    def object_hook(d: MutableMapping[str, Any]):
        return EditTrackingDict(d)


class JSONETDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, EditTrackingDict):
            return dict(obj)
        return super().default(obj)


class _DELETED:
    """Sentinel object for deleted values"""


DELETED = _DELETED()
_KT = TypeVar('_KT')
_VT = TypeVar('_VT')


class EditTrackingDict(MutableMapping[_KT, _VT]):
    def __init__(self, *args, **kwargs):
        self._dict: MutableMapping[_KT, _VT] = dict(*args, **kwargs)
        self._edits: MutableMapping[_KT, _VT] = {}
        self.replace_dicts()

    def replace_dicts(self) -> None:
        for k, v in self._dict.items():
            self[k] = v
        self.reset_edits()

    def reset_edits(self) -> None:
        for v in self.values():
            if isinstance(v, EditTrackingDict):
                v.reset_edits()
        self._edits = {}

    @property
    def modified(self) -> bool:
        return self._edits or any(v.modified for v in self.values() if isinstance(v, EditTrackingDict))

    def iteredits(self) -> Iterator[Tuple[_KT, Union[_VT, _DELETED]]]:
        yield from self._edits.items()

    def __setitem__(self, k: _KT, v: _VT) -> None:
        if isinstance(v, dict):
            v = EditTrackingDict(v)
        self._dict[k] = v
        self._edits[k] = v

    def __delitem__(self, k: _KT) -> None:
        del self._dict[k]
        self._edits[k] = DELETED

    def __getitem__(self, k: _KT) -> _VT:
        return self._dict[k]

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator[_KT]:
        return iter(self._dict)
