"""Therm devices definitions"""

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from datetime import datetime
from typing import Optional, List, Type, Mapping, Any

from dateutil.parser import parse as dateutil_parser

from .base import DeviceConfig, LogRecord, DeviceLog
from .tempsdevices import TempsDeviceLogRecord, TempSetLogRecord, TempsDeviceLog, TempsDevice

__all__ = [
    "ThermLogRecord",
    "ThermDeviceLog",
    "ThermDevice",
]


@dataclasses.dataclass(frozen=True)
class ThermLogRecord(TempSetLogRecord, TempsDeviceLogRecord, LogRecord):
    """
    `LogRecord` subclass for "therm" device types
    """
    @staticmethod
    def _get_device_type() -> str:
        return 'therm'

    operating_mode: str
    """Operating mode of the therm device at log record time"""

    temp_set: Optional[float]
    """The set threshold temperature for the device at log record time,
    usually derived from the configured schedule"""

    therm_state: bool
    """The thermostat switch state at log record time"""

    tampered: bool
    """Whether the tamper switch has been detected"""

    grace_given: bool
    """Whether the thermostat is operating in "grace" mode (obsolete)"""

    @classmethod
    def from_raw_logline(cls, raw_line: str) -> LogRecord:
        line_items: List[str] = raw_line.rstrip('\r\n').split(' ')
        timestamp: Optional[datetime] = dateutil_parser(line_items[0]) if line_items[0] else None
        offset: int
        has_opmode: bool
        try:
            float(line_items[1])
        except ValueError:
            has_opmode = True
            offset = 0
        else:
            has_opmode = False
            offset = -1
        return cls(
            timestamp=timestamp,
            timestamp_missing=timestamp is None,
            operating_mode=line_items[1] if has_opmode else 'schedule',
            temp_current=float(line_items[2+offset]) if line_items[2+offset] else None,
            temp_average=float(line_items[3+offset]) if line_items[3+offset] else None,
            temp_set=float(line_items[4+offset]) if line_items[4+offset] else None,
            therm_state=line_items[5+offset] == 'on',
            wlan_active=line_items[6+offset] == 'wlan',
            tampered=line_items[7+offset] == 'tampered',
            grace_given=line_items[8+offset] == 'gracious',
            battery_voltage=None,
        )


class ThermDeviceLog(TempsDeviceLog):
    """
    `DeviceLog` subclass for "therm" device types
    """
    @staticmethod
    @abstractmethod
    def get_logrecord_class() -> Type[LogRecord]:
        return ThermLogRecord

    @staticmethod
    def get_sched_temps(config: Mapping[str, Any]) -> List[float]:
        """
        Helper static method to get the schedule temperatures from a given device config dict.
        :param config: a mapping-like object (config or dict)
        :return: the list of hourly scheduled temperatures
        """
        return config['thermostat']['sched_temps']


# pylint: disable=missing-function-docstring,unsubscriptable-object
class ThermDevice(TempsDevice[DeviceConfig, ThermDeviceLog]):
    """
    `Device` subclass for "therm" device types
    """
    @staticmethod
    def _get_device_type() -> str:
        return 'therm'

    @staticmethod
    def _get_log_class() -> Type[DeviceLog]:
        return ThermDeviceLog
