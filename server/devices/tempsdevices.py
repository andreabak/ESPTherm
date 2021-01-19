"""Base devices definitions for devices that log temperatures"""
import dataclasses
from abc import ABC, abstractmethod
from typing import TypeVar, Optional, List

from .base import DeviceConfig, DeviceLog, Device


__all__ = [
    "TempsDeviceConfig",
    "TempsDeviceLogRecord",
    "TempSetLogRecord",
    "TempsDeviceLog",
    "TempsDevice",
]


class TempsDeviceConfig(DeviceConfig, ABC):
    """
    Base abstract class for `DeviceConfig` subclasses of devices that log temperatures
    """
    @property
    @abstractmethod
    def sched_temps(self) -> List[float]:
        """
        Helper property to get the config schedule temperatures from a given device config dict.
        :return: the list of hourly scheduled temperatures
        """


@dataclasses.dataclass(frozen=True)
class TempsDeviceLogRecord:
    """
    Dataclass mixin for LogRecords that store temperatures
    """
    temp_current: Optional[float]
    """The current raw unfiltered temperature value"""

    temp_average: Optional[float]
    """The averaged temperature value, as per device's config"""


@dataclasses.dataclass(frozen=True)
class TempSetLogRecord:
    """
    Dataclass mixin for LogRecords that have set temperatures
    """
    temp_set: Optional[float]
    """The set threshold temperature for the device at log record time,
    usually derived from the configured schedule"""


class TempsDeviceLog(DeviceLog, ABC):
    """Base abstract class for `DeviceLog` subclasses of devices that log temperatures"""


CT = TypeVar('CT', bound=TempsDeviceConfig)
LT = TypeVar('LT', bound=TempsDeviceLog)


class TempsDevice(Device[CT, LT], ABC):
    """Base abstract class for `Device` subclasses of devices that log temperatures"""
