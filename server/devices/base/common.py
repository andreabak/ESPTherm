"""Common code for base device classes"""

from __future__ import annotations

import os
from abc import ABC
from typing import Iterator, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .device import Device


__all__ = [
    "iterate_type_id_paths",
    "DeviceBound",
]


def iterate_type_id_paths(base_path: str, ext: str) -> Iterator[Tuple[str, str]]:
    """
    Helper function to iterate a directory structure like `base_path/type_name/id_name.ext`
    :param base_path: the base path to start from
    :param ext: the file extension to filter `id_name` files by.
                If omitted or None, disables the filtering behaviour.
    :return: an iterator of tuples of str with (type_name, id_name)
    """
    type_name: str
    for type_name in os.listdir(base_path):
        type_dir: str = os.path.join(base_path, type_name)
        if os.path.isdir(type_dir):
            filename: str
            for filename in os.listdir(type_dir):
                filepath: str = os.path.join(type_dir, filename)
                id_name: str
                file_ext: str
                id_name, file_ext = os.path.splitext(filename)
                if os.path.isfile(filepath) and file_ext == ext:
                    yield type_name, id_name


class DeviceBound(ABC):
    """
    Abstract mixin class that keeps a read only attribute
    linking to a `Device` object set at initialization
    """
    def __init__(self, device: Device, *args, **kwargs):
        """
        Initialization for `DeviceBound`
        :param device: the `Device` object to link this instance to
        :param args: additional positional arguments to be passed to `super().__init__(...)`
        :param kwargs: additional keyword arguments to be passed to `super().__init__(...)`
        """
        super().__init__(*args, **kwargs)
        self._device: Device = device

    @property
    def device(self) -> Device:
        """The `Device` object linked to this instance"""
        return self._device
