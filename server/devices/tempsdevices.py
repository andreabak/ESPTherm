"""Base devices definitions for devices that log temperatures"""
import dataclasses
from abc import ABC
from typing import TypeVar, Optional

from .base import DeviceConfig, DeviceLog, Device


__all__ = [
    "TempsDeviceLogRecord",
    "TempSetLogRecord",
    "TempsDeviceLog",
    "TempsDevice",
]


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


CT = TypeVar('CT', bound=DeviceConfig)
TLT = TypeVar('TLT', bound=TempsDeviceLog)


class TempsDevice(Device[CT, TLT], ABC):
    """Base abstract class for `Device` subclasses of devices that log temperatures"""
