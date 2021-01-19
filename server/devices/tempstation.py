"""TempStation devices definitions"""

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from datetime import datetime
from typing import Optional, List, Type, MutableMapping, Any

from dateutil.parser import parse as dateutil_parser

from .base import DeviceConfig, LogRecord, DeviceLog
from .tempsdevices import TempsDeviceLogRecord, TempSetLogRecord, TempsDeviceLog, TempsDevice

__all__ = [
    "TempStationLogRecord",
    "TempStationDeviceLog",
    "TempStationDevice",
]


@dataclasses.dataclass(frozen=True)
class TempStationLogRecord(TempSetLogRecord, TempsDeviceLogRecord, LogRecord):
    """
    `LogRecord` subclass for "tempstation" device types
    """
    @staticmethod
    def _get_device_type() -> str:
        return 'tempstation'

    humidity_current: Optional[float]
    """The current raw unfiltered temperature value"""

    humidity_average: Optional[float]
    """The averaged temperature value, as per device's config"""

    @classmethod
    def from_raw_logline(cls, raw_line: str) -> LogRecord:
        line_items: List[str] = raw_line.rstrip('\r\n').split(' ')
        timestamp: Optional[datetime] = dateutil_parser(line_items[0]) if line_items[0] else None
        return cls(
            timestamp=timestamp,
            timestamp_missing=timestamp is None,
            temp_current=float(line_items[1]) if line_items[1] else None,
            temp_average=float(line_items[2]) if line_items[2] else None,
            humidity_current=float(line_items[3]) if line_items[3] else None,
            humidity_average=float(line_items[4]) if line_items[4] else None,
            temp_set=float(line_items[5]) if line_items[5] else None,
            wlan_active=line_items[6] == 'wlan',
            battery_voltage=float(line_items[7]) if len(line_items) > 7 and line_items[7] else None,
        )


class TempStationDeviceLog(TempsDeviceLog):
    """
    `DeviceLog` subclass for "tempstation" device types
    """
    @staticmethod
    @abstractmethod
    def get_logrecord_class() -> Type[LogRecord]:
        return TempStationLogRecord

    @staticmethod
    def get_sched_temps(config: MutableMapping[str, Any]) -> List[float]:
        """
        Helper static method to get the schedule temperatures from a given device config dict.
        :param config: a mapping-like object (config or dict)
        :return: the list of hourly scheduled temperatures
        """
        return config['station']['sched_temps']


# pylint: disable=missing-function-docstring,unsubscriptable-object
class TempStationDevice(TempsDevice[DeviceConfig, TempStationDeviceLog]):
    """
    `Device` subclass for "tempstation" device types
    """
    @staticmethod
    def _get_device_type() -> str:
        return 'tempstation'

    @staticmethod
    def _get_log_class() -> Type[DeviceLog]:
        return TempStationDeviceLog
