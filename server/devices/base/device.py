"""Base class for devices definitions"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Tuple, Optional, TypeVar, Generic, ClassVar, MutableMapping, Type, Iterator

from .common import DeviceTypeRegistrar
from .config import DeviceConfig
from .log import DeviceLog


__all__ = [
    "Device",
    "DeviceFullID",
    "DeviceOptionalID",
]


DeviceFullID = Tuple[str, str]
DeviceOptionalID = Tuple[str, Optional[str]]
CT = TypeVar('CT', bound=DeviceConfig)
LT = TypeVar('LT', bound=DeviceLog)


# noinspection SyntaxError
class Device(DeviceTypeRegistrar, ABC, Generic[CT, LT]):
    """
    Abstract base class that represents a device and its related data
    """
    _known_devices: ClassVar[MutableMapping[DeviceFullID, Device]] = {}
    _cached_configs: ClassVar[MutableMapping[DeviceOptionalID, CT]] = {}

    @staticmethod
    @abstractmethod
    def _get_config_class() -> Type[DeviceConfig]:
        """
        Abstract static method that subclasses must implement to return
        the associated `DeviceConfig` subclass for their specific device type.
        """

    @staticmethod
    @abstractmethod
    def _get_log_class() -> Type[DeviceLog]:
        """
        Abstract static method that subclasses must implement to return
        the associated `DeviceLog` subclass for their specific device type.
        """

    @classmethod
    def _register_instance(cls, instance: Device) -> None:
        """
        Internal class method that registers an instantiated `Device` object as a known device
        with its full id, ensuring there are no conflicts.
        :param instance: the instantiated `Device` instance
        """
        instance_full_id: DeviceFullID = instance.full_device_id
        conflicting_device: Optional[Device] = cls._known_devices.get(instance_full_id)
        if conflicting_device is not None:
            raise ValueError(f'Instantiating new device {repr(instance)} with same full id "{instance_full_id}" '
                             f'as already existing instance {repr(conflicting_device)}')
        cls._known_devices[instance_full_id] = instance

    @classmethod
    def iterate_known_devices(cls) -> Iterator[Device]:
        """
        Iterates all known devices.
        :return: an iterator of `Device` objects
        """
        for device in cls._known_devices.values():
            yield device

    @classmethod
    def get_known_device(cls, device_type: str, device_id: str) -> Device:
        """
        Get a known device by type and id.
        :param device_type: the device type
        :param device_id: the device id
        :raise KeyError: if there's no known device with the given type and id
        :return: the known `Device` object
        """
        return cls._known_devices[(device_type, device_id)]

    @classmethod
    def load_logged_devices(cls) -> None:
        """Preloads all known devices from available log files."""
        for device_type, device_id in DeviceLog.iterate_available_logs():
            cls.for_type(device_type=device_type, device_id=device_id, preload=True)

    @classmethod
    def get_config(cls, device_type: str, device_id: Optional[str]) -> CT:
        """
        Class method to get a config for the specified device type and id.
        Configs are internally cached and kept up-to-date.
        :param device_type: the device type
        :param device_id: the device id
        :return: the device config object
        """
        if not DeviceConfig.device_id_config_exists(device_type, device_id):
            device_id = None
        full_id = device_type, device_id
        cached_config: Optional[CT] = cls._cached_configs.get(full_id)
        if cached_config is None or not cached_config.uptodate:
            cls._cached_configs[full_id] = DeviceConfig.from_type_id(device_type, device_id)
        return cls._cached_configs[full_id]

    @classmethod
    def load_known_configs(cls) -> None:
        """Preloads all known device configs, caching them."""
        for device_type, device_id in DeviceConfig.iterate_available_configs():
            cls.get_config(device_type, device_id)

    @classmethod
    def load_known_devices(cls):
        """Preloads all known devices and their configs."""
        cls.load_known_configs()
        cls.load_logged_devices()

    def __init__(self, device_id: str, preload: bool = False):
        """
        Initialization for `Device`
        :param device_id: the device id
        :param preload: if True, preloads device's config and log data. Defaults to False
        """
        self._device_id: str = device_id
        self._config: Optional[CT] = None

        self.__class__._register_instance(self)

        if preload:
            self._load_config()
            self._get_log_class().reflect_to_db(device_type=self.device_type, device_id=self.device_id)

    @property
    def device_type(self) -> str:
        """The device type of this instance."""
        return self._get_device_type()

    @property
    def device_id(self) -> str:
        """The device id of this instance."""
        return self._device_id

    @property
    def full_device_id(self) -> DeviceFullID:
        """The complete device id as a tuple of (device_type, device_id)."""
        return self.device_type, self._device_id

    def _load_config(self) -> None:
        """Internal method that loads the device's config into the instance."""
        self._config = self.get_config(self.device_type, self.device_id)

    @property
    def config(self) -> CT:
        """The device's config"""
        if self._config is None or not self._config.uptodate:
            self._load_config()
        return self._config

    def get_log(self, **filters) -> LT:
        """
        Gets a device log, applying any specified loading filters
        :param filters: additional keyword arguments passed to the log class to filter the records
        :return: the loaded device log
        """
        # noinspection PyUnresolvedReferences,SyntaxError,Annotator
        return self._get_log_class().from_device(self, **filters)  # type: ignore[return-value]

    @property
    def full_log(self) -> LT:
        """The full log of the device, with all the existing records.
        N.B. This might take a long time to load."""
        return self.get_log()

    @property
    def log_daily(self) -> LT:
        """The device log of only the last 24 hours"""
        # TODO: UTC conversion? Check
        # Rounding to hour to allow caching
        daily_min_dt: datetime = (datetime.now() - timedelta(hours=24)).replace(minute=0, second=0, microsecond=0)
        return self.get_log(timestamp_min=daily_min_dt)
