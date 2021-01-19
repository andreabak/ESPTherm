"""Base classes for devices configs"""

from __future__ import annotations

import json
import os
from abc import ABC
from typing import Optional, Iterator, Tuple, IO, MutableMapping, Any, TYPE_CHECKING, Type

from ...base import UniqueHashable, LiveFileData
from ...config import DEFAULT_CONFIG_BASENAME, CONFIGS_PATH, CONFIGS_EXT

from .common import iterate_type_id_paths, DeviceTypeRegistrar

if TYPE_CHECKING:
    from .device import Device


__all__ = [
    "DeviceConfig",
]


class DeviceConfig(DeviceTypeRegistrar, UniqueHashable, LiveFileData, dict, ABC):
    """
    Class that represents a device's configuration, subclasses dict
    """
    # pylint: disable=arguments-differ
    @classmethod
    def file_path(cls, device_type: str, device_id: Optional[str] = None) -> str:
        """
        Builds and returns the path of a config file for a device.
        :param device_type: the device type
        :param device_id: the device id. If omitted or None, the path of the default
                          config file for the device type is returned instead.
        :return: the file path as a string
        """
        config_basename: str = DEFAULT_CONFIG_BASENAME if device_id is None else device_id
        config_path: str = os.path.join(CONFIGS_PATH, device_type, config_basename + CONFIGS_EXT)
        return config_path

    @staticmethod
    def iterate_available_configs() -> Iterator[Tuple[str, Optional[str]]]:
        """
        Iterates device type, id for all available config files
        :return: an iterator of tuples of str, Optional[str] of (device_type, device_id)
        """
        device_type: str
        device_id: Optional[str]
        for device_type, device_id in iterate_type_id_paths(CONFIGS_PATH, ext=CONFIGS_EXT):
            if device_id == DEFAULT_CONFIG_BASENAME:
                device_id = None
            yield device_type, device_id

    @classmethod
    def device_id_config_exists(cls, device_type: str, device_id: str) -> bool:
        """
        Check if a config file for a specific device_type and device_id exists
        :param device_type: the device type
        :param device_id: the device id
        :return: True if the config file exists, else False
        """
        config_path: str = cls.file_path(device_type, device_id)
        return os.path.exists(config_path)

    @classmethod
    def from_type_id(cls, device_type: str, device_id: Optional[str] = None) -> DeviceConfig:
        """
        Build a `DeviceConfig` instance loading data for a device type and id
        :param device_type: the device type
        :param device_id: the device id.
                          If omitted or None, the default config for the device type will be used instead.
        :return: the built `DeviceConfig` instance
        """
        config_path: str = cls.file_path(device_type, device_id)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Device config file does not exist: {config_path}')
        config_mtime: float = cls.file_mtime(path=config_path)
        config_cls: Type[DeviceConfig] = cls.get_class_for_device_type(device_type)
        config: DeviceConfig = config_cls(device_type=device_type, device_id=device_id,
                                          loaded_path=config_path, loaded_mtime=config_mtime)
        fp: IO
        with open(config_path, 'r') as fp:
            config_data: MutableMapping[str, Any] = json.load(fp)
        config.update(config_data)
        config.freeze()
        return config

    @classmethod
    def from_device(cls, device: Device) -> DeviceConfig:
        """
        Build a `DeviceConfig` instance from a `Device` object, loading from the correct config file
        :param device: the `Device` object
        :return: the built `DeviceConfig` instance
        """
        device_type: str
        device_id: Optional[str]
        device_type, device_id = device.full_device_id
        if not cls.device_id_config_exists(device_type, device_id):
            device_id = None
        return cls.from_type_id(device_type, device_id)

    def __init__(self, device_type: str, device_id: Optional[str], *args, **kwargs):
        """
        Initialization for `DeviceConfig`
        :param device_type: the device type
        :param device_id: the device id
        :param args: additional positional arguments to be passed to `super().__init__(...)`
        :param kwargs: additional keyword arguments to be passed to `super().__init__(...)`
        """
        super().__init__(*args, **kwargs)
        self.device_type: str = device_type
        self.device_id: Optional[str] = device_id

    @property
    def unique_id(self) -> Any:
        return (self.device_type, self.device_id), self.loaded_mtime
