"""Base classes and common functions"""

from __future__ import annotations

import itertools
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, Any, MutableMapping, TypeVar, Iterator, Tuple, Union, Sequence, Iterable, List


__all__ = [
    "chunked",
    "ensure_tz",
    "LiveFileData",
    "UniqueHashable",
    "DELETED",
    "JSONETDDecoder",
    "JSONETDEncoder",
    "EditTrackingDict"
]


ST = TypeVar('ST')


def chunked(it: Iterable[ST], n: int) -> Iterator[Sequence[ST]]:
    """
    Split up an iterable object into n-sized chunks.
    :param it: an iterable object
    :param n: the maximun size of the chunks
    :return: an iterator of sequences (chunks) of items from the input iterable object
    """
    it = iter(it)  # Ensure we're acting on a iterator
    while True:
        chunk: List[ST] = list(itertools.islice(it, n))
        if not chunk:
            break
        yield chunk


def ensure_tz(dt: datetime, default_tz: Optional[timezone] = timezone.utc) -> datetime:
    """
    Return a datetime that is timezone-aware
    :param dt: the input datetime object
    :param default_tz: the default timezone to be applied if `dt` is not timezone-aware
    :return: a timezone-aware datetime object (could be a copy or the same instance as `dt`)
    """
    return dt.replace(tzinfo=default_tz) if dt.tzinfo is None else dt


class LiveFileData(ABC):
    """
    A mixin abstract class that provides helper functionality to subclasses
    that store data that's loaded from a file and needs to be kept up-to-date.
    """
    @classmethod
    @abstractmethod
    def file_path(cls, *args, **kwargs) -> str:
        """
        An abstract class method to be implemented by subclasses that must
        build and return the path of the file associated to it.
        :param args: additional positional arguments to build the file path
        :param kwargs: additional keyword arguments to build the file path
        :return: the file path as a string
        """

    @classmethod
    def file_mtime(cls, *path_args, path: Optional[str] = None, **path_kwargs) -> float:
        """
        Get the file's modification time (mtime).
        :param path: if specified, use the provided path to get the file mtime,
                     otherwise the class' `.file_path(...)` method will be used instead
        :param path_args: when `path` is not provided, additional positional arguments to be passed to `.file_path(...)`
        :param path_kwargs: when `path` is not provided, additional keyword arguments to be passed to `.file_path(...)`
        :return: the file modification timestamp as float
        """
        if path and (path_args or path_kwargs):
            raise AttributeError('Must specify either "path" or "path_args" / "path_kwargs" arguments')
        if path is None:
            path = cls.file_path(*path_args, **path_kwargs)
        return os.path.getmtime(path)

    def __init__(self, *args, loaded_path: str, loaded_mtime: Optional[float] = None, **kwargs):
        """
        Initialization for `LiveFileData` instances
        :param loaded_path: the path of the loaded file associated to this instance
        :param loaded_mtime: the modification time of the file associated with this instance at load time,
                             if not specified, it will be set to the current file's mtime
        :param args: additional positional arguments to be passed to `super().__init__(...)`
        :param kwargs: additional keyword arguments to be passed to `super().__init__(...)`
        """
        super().__init__(*args, **kwargs)
        self._loaded_path: str = loaded_path
        if loaded_mtime is None:
            loaded_mtime = self.file_mtime(path=self._loaded_path)
        self._loaded_mtime: float = loaded_mtime

    @property
    def loaded_mtime(self) -> float:
        """Get the loaded mtime of the file associated with this instance"""
        return self._loaded_mtime

    @property
    def uptodate(self) -> bool:
        """Is True if the instance is up-to-date (loaded mtime matches current file's mtime)"""
        return self.loaded_mtime == self.file_mtime(path=self._loaded_path)


class UniqueHashable(ABC):
    """
    A mixin abstract class that provides functionality to make instances
    of subclasses hashable and comparable, based on a unique id.
    The unique id doesn't have to literally be "unique" per instance,
    but it should represent the uniqueness of the data held.
    Instances of this class can be optionally "frozen":
    trying to set items to a frozen instance will subsequently fail.
    (N.B. be mindful about parents classes order for this to work)
    """
    @property
    @abstractmethod
    def unique_id(self) -> Any:
        """
        An abstract property to be implemented by subclasses that must return
        an unique id for the instance.
        The unique id doesn't have to literally be "unique" per instance,
        but it should represent the uniqueness of the data held.
        """

    def __init__(self, *args, freeze: bool = False, **kwargs):
        """
        Initialization for `UniqueHashable` instances
        :param freeze: if True, freeze the instance as soon as it's initialized
        :param args: additional positional arguments to be passed to `super().__init__(...)`
        :param kwargs: additional keyword arguments to be passed to `super().__init__(...)`
        """
        super().__init__(*args, **kwargs)
        self._frozen: bool = freeze

    def freeze(self) -> None:
        """Freezes the instance."""
        self._frozen = True

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Override of the __setitem__ method to allow for freezing.
        :param key: the key of the item to set
        :param value: the value of the item to set
        """
        if self._frozen:
            raise PermissionError('Cannot set items to frozen instance')
        if hasattr(super(), '__setitem__'):
            # noinspection PyUnresolvedReferences
            super().__setitem__(key, value)  # type: ignore[misc]  # pylint: disable=no-member

    def __hash__(self) -> int:
        """Returns the hash of the instance"""
        return hash((self.__class__, self.unique_id))

    def __eq__(self, other: Any) -> bool:
        """Checks if the instance equals another, based on their unique ids"""
        if not isinstance(other, self.__class__):
            return False
        return self.unique_id == other.unique_id


class JSONETDDecoder(json.JSONDecoder):
    """Custom `JSONDecoder` that replaces dicts with `EditTrackingDict`s"""
    def __init__(self, *args, **kwargs):
        """
        Initialization for `JSONETDDecoder`
        :param args: additional positional arguments to be passed to `super().__init__(...)`
        :param kwargs: additional keyword arguments to be passed to `super().__init__(...)`
        """
        kwargs.setdefault('object_hook', self._object_hook)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _object_hook(d: MutableMapping[str, Any]) -> EditTrackingDict:
        """
        Hook function to intercept decoded dict objects and convert them to `EditTrackingDict`s
        :param d: the json-decoded dict
        :return: the EditTrackingDict instance
        """
        return EditTrackingDict(d)


class JSONETDEncoder(json.JSONEncoder):
    """Custom `JSONEncoder` that supports `EditTrackingDict`s for json encoding"""
    def default(self, o: Any) -> Any:
        """
        `JSONEncoder.default` override method used to convert
        `EditTrackingDict`s to dicts during json encoding.
        :param o: the input object
        :return: the output object
        """
        if isinstance(o, EditTrackingDict):
            return dict(o)
        return super().default(o)


class _DELETED:
    """Sentinel object for deleted values. Using a class for better repr"""


DELETED = _DELETED()
KT = TypeVar('KT')
VT = TypeVar('VT')
EVT = Union[VT, _DELETED]


# pylint: disable=no-member
class EditTrackingDict(MutableMapping[KT, VT]):
    """
    A dict-like class that tracks edits done to the map values.
    N.B. edits done indirectly within complex objects might not be tracked.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialization for `EditTrackingDict`
        :param args: additional positional arguments to be passed to `super().__init__(...)`
        :param kwargs: additional keyword arguments to be passed to `super().__init__(...)`
        """
        self._dict: MutableMapping[KT, VT] = dict(*args, **kwargs)
        self._edits: MutableMapping[KT, EVT] = {}
        self.replace_dicts()

    def replace_dicts(self) -> None:
        """
        Recursively replaces any values stored as dicts to `EditTrackingDict` instances.
        Also, resets edits, so after calling this the instance will appear as unmodified.
        """
        for k, v in self._dict.items():
            self[k] = v
        self.reset_edits()

    def reset_edits(self) -> None:
        """
        Recursively resets edits, setting this instance and any child `EditTrackingDict` instance as unmodified.
        """
        for v in self.values():
            if isinstance(v, EditTrackingDict):
                v.reset_edits()
        self._edits = {}

    @property
    def modified(self) -> bool:
        """True if any edits are present, False if the instance appears to be unmodified"""
        return bool(self._edits) or any(v.modified for v in self.values() if isinstance(v, EditTrackingDict))

    def iteredits(self) -> Iterator[Tuple[KT, EVT]]:
        """
        Iterates the edits applied to this instance.
        :return: an iterator of (key, value) tuples, of only the modified keys.
                 A value might be `DELETED`, if the key was originally present, but has since been deleted.
        """
        yield from self._edits.items()

    def __setitem__(self, k: KT, v: VT) -> None:
        if isinstance(v, dict):
            # noinspection PyUnresolvedReferences
            v = EditTrackingDict(v)  # type: ignore[assignment]
        self._dict[k] = v
        self._edits[k] = v

    def __delitem__(self, k: KT) -> None:
        del self._dict[k]
        self._edits[k] = DELETED

    def __getitem__(self, k: KT) -> VT:
        return self._dict[k]

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator[KT]:
        return iter(self._dict)
