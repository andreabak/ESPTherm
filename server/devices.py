"""Devices datastructures and code"""

from __future__ import annotations

import json
import os
import dataclasses
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import (ClassVar, MutableMapping, Type, Optional, List, Tuple, Union, Any, Iterator, TypeVar, IO, Generator,
                    Set, ContextManager, Iterable, Mapping, ForwardRef, Hashable, Generic, Sequence)

import dataset
from dataset.util import ResultIter
from sqlalchemy import Integer, UnicodeText, Float, Boolean, DateTime, Column
from sqlalchemy.sql.type_api import TypeEngine
from dateutil.parser import parse as dateutil_parser

from . import app
from .base import LiveFileData, UniqueHashable, ensure_tz, chunked
from .config import LOGS_PATH, LOGS_EXT, CONFIGS_PATH, CONFIGS_EXT, DEFAULT_CONFIG_BASENAME, LOGS_DB_PATH, LOGS_ROTATE_SIZE


__all__ = [
    "DeviceConfig",
    "LogRecord",
    "DeviceLog",
    "DeviceFullID",
    "DeviceOptionalID",
    "Device",
    "ThermLogRecord",
    "ThermDeviceLog",
    "ThermDevice",
    "TempStationLogRecord",
    "TempStationDeviceLog",
    "TempStationDevice",
    "TempsDeviceLogRecord",
    "TempsDeviceLog",
    "TempsDevice"
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


class DeviceConfig(UniqueHashable, LiveFileData, dict):
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
        config: DeviceConfig = cls(device_type=device_type, device_id=device_id,
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


# TODO: Refactor classes that register against device_type with some mixin or metaclass?

# pylint: disable=invalid-name
@dataclasses.dataclass(frozen=True)
class LogRecord(ABC):
    """
    Abstract base dataclass for device log records
    """
    DB_TYPES_MAP: ClassVar[MutableMapping[Type, TypeEngine]] = {
        int: Integer,
        str: UnicodeText,
        float: Float,
        bool: Boolean,
        datetime: DateTime,
    }
    """Mapping from field types to SQLAlchemy column types"""

    # TODO: Replace fields class variables with factory methods to make overridable?
    
    DB_PKEY_FIELDS: ClassVar[Set[str]] = {'timestamp'}  
    """Set of field names that are primary keys columns in the database device-type's table"""

    DB_IDX_FIELDS: ClassVar[Set[str]] = {'timestamp'}
    """Set of field names that are indexed columns in the database device-type's table"""

    _device_types_classes: ClassVar[MutableMapping[str, Type[LogRecord]]] = {}

    @staticmethod
    @abstractmethod
    def _get_device_type() -> str: 
        """
        Abstract static method that subclasses must implement
        to return the associated device type as string.
        N.B. if two subclasses are defined with the same device type,
             a NameError will be raised.
        """

    @classmethod
    def get_known_types_classes(cls) -> Iterator[Tuple[str, Type[LogRecord]]]:
        """
        Iterates through all the defined `LogRecord` classes for each device type.
        :return: an iterator of tuples of (device_type, log_record_class)
        """
        for k, v in cls._device_types_classes.items():
            yield k, v

    def __init_subclass__(cls): 
        """
        Initializes a defined `LogRecord` subclass, registering it's associated device type.
        :raise NameError: if a subclass is being initialized with the same device type as an another one
        """
        device_type: str = cls._get_device_type()
        existing_device_type_class: Type[LogRecord] = cls._device_types_classes.get(device_type)
        if existing_device_type_class is not None:
            raise NameError(f'More than one {LogRecord.__name__} class with same device type "{device_type}" defined: '
                            f'{existing_device_type_class.__name__} and {cls.__name__}')
        cls._device_types_classes[device_type] = cls

    @classmethod
    def get_class_for_device_type(cls, device_type: str) -> Type[LogRecord]: 
        """
        Gets the `LogRecord` class for the given device type.
        :param device_type: the device type for which to get the `LogRecord` class
        :raise NameError: if no class for the given device type is found
        :return: the registered `LogRecord` class for the device type
        """
        log_class: Optional[Type[LogRecord]] = cls._device_types_classes.get(device_type)
        if log_class is None:
            raise NameError(f'No log record class found for device type "{device_type}"')
        return log_class

    @classmethod
    def for_type(cls, device_type: str, *args, **kwargs) -> LogRecord: 
        """
        Creates a `LogRecord` instance using the appropriate subclass for the given device type
        :param device_type: the device type for which to instantiate the correct `LogRecord` subclass
        :param args: additional positional arguments to be passed to the `LogRecord` subclass' `__init__(...)`
        :param kwargs: additional keyword arguments to be passed to the `LogRecord` subclass' `__init__(...)`
        :return: the instantiated `LogRecord` subclass' instance
        """
        log_class = cls.get_class_for_device_type(device_type)
        # noinspection PyArgumentList
        return log_class(*args, **kwargs)

    @classmethod
    @abstractmethod
    def from_raw_logline(cls, raw_line: str) -> LogRecord:
        """
        Abstract class method that subclasses must implement
        to create a `LogRecord` instance from a raw log line.
        :param raw_line: the raw log line string
        :return: the `LogRecord` instance created from the log line.
        """

    @staticmethod
    def parse_timestamp(timestamp_raw: str, timestamp_override: Optional[datetime] = None) -> datetime:
        """
        Helper static method for parsing a raw timestamp string
        :param timestamp_raw: the raw timestamp string
        :param timestamp_override: if specified, return this directly
        :raise ValueError: if the raw timestamp couldn't be parsed
        :return: the parsed timestamp as a datetime object
        """
        timestamp: datetime
        if timestamp_override is None:
            parsed_timestamp: Optional[datetime] = dateutil_parser(timestamp_raw) if timestamp_raw else None
            if not parsed_timestamp:
                raise ValueError('Missing timestamp')
            timestamp = parsed_timestamp
        else:
            timestamp = timestamp_override
        return timestamp

    @classmethod
    def fields_to_db_columns(cls) -> List[Column]:
        """
        Converts the fields defined in the dataclass to a list of database columns definitions.
        :return: the list of columns
        """
        columns: List[Column] = []
        field: dataclasses.Field
        for field in dataclasses.fields(cls):
            ftype: Any = field.type
            if isinstance(ftype, str):
                # noinspection PyProtectedMember
                # pylint: disable=protected-access
                ftype = ForwardRef(ftype)._evaluate(globalns=globals(), localns=locals())
            is_union: bool = (getattr(ftype, '__origin__', None) is Union) or 'typing.Union' in repr(ftype)
            is_optional: bool = is_union and type(None) in ftype.__args__
            if is_union and (not is_optional or len(ftype.__args__) > 2):
                raise TypeError(f'Cannot derive unique DB type from field type: {repr(ftype)}')
            actual_type: Type
            if is_union:
                for subtype in ftype.__args__:
                    if subtype not in (None, type(None)):
                        actual_type = subtype
                        break
                else:
                    raise TypeError(f'Did not find suitable DB type from field type: {repr(ftype)}')
            else:
                actual_type = ftype
            # noinspection PyTypeChecker
            db_type: Optional[TypeEngine] = cls.DB_TYPES_MAP.get(actual_type)
            if db_type is None:
                raise KeyError(f'Field type "{repr(ftype)}" does not map to any DB type!')
            is_pkey: bool = field.name in cls.DB_PKEY_FIELDS
            has_idx: bool = field.name in cls.DB_IDX_FIELDS
            columns.append(Column(field.name, db_type, nullable=is_optional, primary_key=is_pkey, index=has_idx))
        return columns

    timestamp: Optional[datetime]
    """The timestamp as datetime of when the record was generated"""

    timestamp_missing: bool
    """True if the timestamp was missing from the source log file"""

    wlan_active: bool
    """True if when the log line was generated the device was in wlan mode"""

    battery_voltage: Optional[float]
    """The battery voltage, if available, else None"""

    def copy(self, **overrides) -> LogRecord:
        """
        Returns a new `LogRecord` instance from the current one, applying any given values overrides
        :param overrides: additional keyword arguments to replace values
                          with the give ones in the copied instance
        :return: the new `LogRecord` instance
        """
        values: MutableMapping[str, Any] = dataclasses.asdict(self)
        values.update(overrides)
        # noinspection PyArgumentList
        return self.__class__(**values)


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


# TODO: Refactor some methods to remove redundant device_type args
class DeviceLog(UniqueHashable, DeviceBound, LiveFileData, list, ABC):
    """
    Abstract base class that represents a loaded device log's data, subclasses list
    """
    _db_thread_lock: RLock = RLock()

    @staticmethod
    @abstractmethod
    def get_logrecord_class() -> Type[LogRecord]:
        """
        Abstract static method that subclasses must implement to return
        the associated `LogRecord` subclass for their specific device type.
        """

    # pylint: disable=arguments-differ
    @classmethod
    def file_path(cls, device_type: str, device_id: str) -> str:
        """
        Builds and returns the path of a log file for a device.
        :param device_type: the device type
        :param device_id: the device id
        :return: the file path as a string
        """
        log_path: str = os.path.join(LOGS_PATH, device_type, device_id + LOGS_EXT)
        return log_path

    @staticmethod
    def iterate_available_logs() -> Iterator[Tuple[str, str]]:
        """
        Iterates device type, id for all available log files
        :return: an iterator of tuples of str, Optional[str] of (device_type, device_id)
        """
        # TODO: Add DB logs
        yield from iterate_type_id_paths(LOGS_PATH, ext=LOGS_EXT)

    # noinspection PyProtectedMember
    # pylint: disable=protected-access
    @classmethod
    def ensure_db_schema(cls, db: dataset.Database) -> None:
        """
        Ensures the db schema exist, creating it if it doesn't.
        :param db: the opened `dataset.Database` object
        """
        db.query('PRAGMA journal_mode = WAL').close()
        device_type: str
        log_record_type: Type[LogRecord]
        for device_type, log_record_type in LogRecord.get_known_types_classes():
            table_columns: List[Column] = [Column('device_id', UnicodeText, primary_key=True)] \
                                          + log_record_type.fields_to_db_columns()
            device_type_table: dataset.Table = db.create_table(device_type, primary_id=False)
            device_type_table._sync_table(table_columns)
        meta_table: dataset.Table = db.create_table('_meta', primary_id=False)
        meta_table._sync_table([
            Column('device_type', UnicodeText, nullable=False, primary_key=True),
            Column('device_id', UnicodeText, nullable=False, primary_key=True),
            Column('log_position', Integer, nullable=False),
            Column('log_mtime', Float, nullable=False),
        ])

    @classmethod
    @contextmanager
    def logs_db_context(cls) -> Iterator[dataset.Database]:
        """
        Context manager function that wraps a database connection and transaction
        :return: the opened `dataset.Database` object
        """
        with cls._db_thread_lock:
            db = dataset.Database(url=f'sqlite:///{LOGS_DB_PATH}',
                                  engine_kwargs=dict(connect_args=dict(check_same_thread=False)))
            cls.ensure_db_schema(db)
            try:
                yield db
            except Exception as exc:
                db.rollback()
                app.logger.warning(f'Error while in db transaction, rolling back:\n{exc}')
                raise
            else:
                db.commit()
            finally:
                db.close()

    @classmethod
    def upload(cls, raw_data: bytes, device_type: str, device_id: str, encoding: str = 'utf8') -> str:
        """
        Class method to store raw data received from a device into the appropriate log file.
        :param raw_data: the raw log data received from the device as encoded bytes
        :param device_type: the device type
        :param device_id: the device id
        :param encoding: the encoding of the raw data, defaults to 'utf8'
        :return: the decoded and log file-stored data
        """
        log_path: str = cls.file_path(device_type, device_id)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_data: str = raw_data.decode(encoding)
        if not log_data.endswith('\n'):
            log_data += '\n'
        with open(log_path, 'a') as fp:
            fp.write(log_data)
        return log_data

    @classmethod
    def parse_records(cls, str_or_iterable: Union[str, Iterable], device_type: str, device_id: Optional[str] = None,
                      last_known_date: Optional[datetime] = None,
                      interval_avg_max_deviation: float = 0.2, ema_alpha: float = 0.25) \
            -> Generator[LogRecord, None, Tuple[Optional[float], Optional[float]]]:
        """
        Generator method for parsing raw or database-loaded log records from an iterable or multiline string.
        :param str_or_iterable: iterable or multiline string to parse the log records from
        :param device_type: the device type
        :param device_id: the device id
        :param last_known_date: the timestamp of the last record before the first that will be parsed, if known
        :param interval_avg_max_deviation: max deviation ratio from the config-defined update/sync intervals
                                           above which records intervals are excluded from averages calculation
        :param ema_alpha: exponential moving average alpha-factor used to calculate intervals averages
        :return: a generator that yields instances of the appropriate `LogRecord` subclass,
                 and finally returns a tuple of floats or None as (average_records_interval, average_sync_interval)
        """
        lines_iterator: Iterable[Union[str, Mapping[str, Any]]]
        if isinstance(str_or_iterable, str):
            lines_iterator = str_or_iterable.splitlines()
        else:
            lines_iterator = str_or_iterable

        device_cfg: DeviceConfig = Device.get_config(device_type=device_type, device_id=device_id)
        expected_records_interval: Optional[float] = device_cfg.get('update_interval')
        if expected_records_interval is None:
            app.logger.warning(f'Failed getting "update_interval" from config for device_type="{device_type}"'
                               + (f', device_id="{device_id}"' if device_id is not None else ''))
        average_records_interval: Optional[float] = None
        expected_sync_interval: Optional[float] = device_cfg.get('sync_interval')
        if expected_sync_interval is None:
            app.logger.warning(f'Failed getting "sync_interval" from config for device_type="{device_type}"'
                               + (f', device_id="{device_id}"' if device_id is not None else ''))
        average_sync_interval: Optional[float] = None
        max_deviation_ratio = 1.0 + interval_avg_max_deviation
        deviation_bounds: Tuple[float, float] = ((1.0 / max_deviation_ratio), max_deviation_ratio)

        log_record_cls: Type[LogRecord] = LogRecord.get_class_for_device_type(device_type)

        last_date: Optional[datetime] = last_known_date
        last_sync: Optional[datetime] = None

        for n, record in enumerate(lines_iterator):
            log_record: LogRecord
            try:
                if isinstance(record, str):
                    line: str = record.rstrip('\r\n')
                    if not line:
                        continue
                    log_record = log_record_cls.from_raw_logline(line)
                elif isinstance(record, (dict, OrderedDict)):
                    record = dict(record)
                    record.pop('device_id')
                    # TODO: lineno from db rowid for logger?
                    # noinspection PyArgumentList
                    log_record = log_record_cls(**record)
                else:
                    raise TypeError(f'Unknown record type for log: {repr(record)}')

            except Exception as exc:  # pylint: disable=broad-except
                app.logger.warning(f'Error while parsing log line {n} (skipped):\n{record}\n{exc}')

            else:
                if log_record.timestamp_missing:
                    if last_date is None:
                        raise ValueError('Missing log dates from the start')
                    estimated_records_interval: Optional[float] = average_records_interval or expected_sync_interval
                    if estimated_records_interval is None:
                        raise ValueError('Missing log date and couldn\'t get update interval to estimate')
                    record_date: datetime = last_date + timedelta(seconds=estimated_records_interval)
                    log_record = log_record.copy(timestamp=record_date)
                if expected_records_interval is not None and last_date is not None:
                    last_record_interval: float = (log_record.timestamp - last_date).total_seconds()
                    if deviation_bounds[0] < (last_record_interval / expected_records_interval) < deviation_bounds[1]:
                        if average_records_interval is None:
                            average_records_interval = last_record_interval
                        average_records_interval = ((1.0 - ema_alpha) * average_records_interval
                                                    + ema_alpha * last_record_interval)
                last_date = log_record.timestamp

                if log_record.wlan_active:
                    if expected_sync_interval is not None and last_sync is not None:
                        last_sync_interval: float = (log_record.timestamp - last_sync).total_seconds()
                        if deviation_bounds[0] < (last_sync_interval / expected_sync_interval) < deviation_bounds[1]:
                            if average_sync_interval is None:
                                average_sync_interval = last_sync_interval
                            average_sync_interval = ((1.0 - ema_alpha) * average_sync_interval
                                                     + ema_alpha * last_sync_interval)
                    last_sync = log_record.timestamp

                yield log_record

        if average_records_interval is None:
            average_records_interval = expected_records_interval
        if average_sync_interval is None:
            average_sync_interval = expected_sync_interval
        return average_records_interval, average_sync_interval

    @classmethod
    def parse_iter(cls, iterable: Iterable[Any], device_type: str, device_id: str, **parser_kwargs) \
            -> Tuple[List[LogRecord], Optional[float], Optional[float]]:
        """
        Helper method used to parse raw or database-loaded log records from an iterable.
        :param iterable: iterable or multiline string to parse the log records from
        :param device_type: the device type
        :param device_id: the device id
        :param parser_kwargs: additional keyword arguments for log lines parser
        :return: a tuple with (log_records, average_records_interval, average_sync_interval)
        """
        average_records_interval: Optional[float]
        average_sync_interval: Optional[float]
        log_records: List[LogRecord] = []
        records_generator: Generator[LogRecord, None, Tuple[Optional[float], Optional[float]]]
        records_generator = cls.parse_records(iterable, device_type=device_type, device_id=device_id, **parser_kwargs)
        while True:
            try:
                record: LogRecord = next(records_generator)
            except StopIteration as stop:
                average_records_interval, average_sync_interval = stop.value
                break
            else:
                log_records.append(record)
        return log_records, average_records_interval, average_sync_interval

    @classmethod
    def parse_file(cls, file_path_or_fp: Union[str, IO], device_type: str, device_id: str, **parser_kwargs) \
            -> Tuple[List[LogRecord], Optional[float], Optional[float]]:
        """
        Class method to parse records from a log file.
        :param file_path_or_fp: a file path, or an already-opened IO file object
        :param device_type: the device type
        :param device_id: the device id
        :param parser_kwargs: additional keyword arguments for log lines parser
        :return: a tuple with (log_records, average_records_interval, average_sync_interval)
        """
        file_context: ContextManager[IO]
        if isinstance(file_path_or_fp, str):
            file_context = open(file_path_or_fp, 'r')
        else:
            file_context = nullcontext(file_path_or_fp)
        with file_context as fp:
            return cls.parse_iter(fp, device_type=device_type, device_id=device_id, **parser_kwargs)

    @classmethod
    def parse_db(cls, device_type: str, device_id: str,
                 timestamp_min: Optional[datetime] = None, timestamp_max: Optional[datetime] = None,
                 bound_exclusive: Optional[bool] = None, **parser_kwargs) \
            -> Tuple[List[LogRecord], Optional[float], Optional[float]]:
        """
        Class method to load records from logs database.
        :param device_type: the device type
        :param device_id: the device id
        :param timestamp_min: optional, filters records returning only ones newer than the given datetime
        :param timestamp_max: optional, filters records returning only ones older than the given datetime
        :param bound_exclusive: if True, the filters include the threshold value/timestamp, defaults to False
        :param parser_kwargs: additional keyword arguments for log lines parser
        :return: a tuple with (log_records, average_records_interval, average_sync_interval)
        """
        def ensure_utc_unaware(dt: datetime) -> datetime:
            return ensure_tz(dt, default_tz=None).astimezone(tz=timezone.utc).replace(tzinfo=None)

        with cls.logs_db_context() as db:
            device_type_table: dataset.Table = db[device_type]
            filters: MutableMapping[str, Mapping[str, Any]] = {}
            timestamp_filter: MutableMapping[str, Any] = {}
            if bound_exclusive is None:
                bound_exclusive = False
            if timestamp_min is not None:
                timestamp_filter['>' if bound_exclusive else '>='] = ensure_utc_unaware(timestamp_min)
            if timestamp_max is not None:
                timestamp_filter['<' if bound_exclusive else '<='] = ensure_utc_unaware(timestamp_max)
            if timestamp_filter:
                filters['timestamp'] = timestamp_filter
            db_results: ResultIter = device_type_table.find(device_id=device_id, **filters, order_by='timestamp')
            return cls.parse_iter(db_results, device_type=device_type, device_id=device_id, **parser_kwargs)

    @classmethod
    def reflect_to_db(cls, device_type: str, device_id: str) -> None:
        """
        Reflects a device's log file to the logs database,
        keeping track of the last reflected position in the log file,
        merging duplicate records (by timestamp) keeping the newest,
        and rotating the log file (truncating) when its size exceeds the maximum set.
        :param device_type: the device type
        :param device_id: the device id
        """
        db: dataset.Database
        with cls.logs_db_context() as db:
            metadata_table: dataset.Table = db['_meta']
            metadata: Optional[MutableMapping[str, Any]]
            metadata = metadata_table.find_one(device_type=device_type, device_id=device_id)

            log_path: str = cls.file_path(device_type, device_id)
            log_mtime: float = cls.file_mtime(path=log_path)
            if not metadata or log_mtime > metadata['log_mtime']:  # Reflect if file is newer
                with open(log_path, 'r+') as fp:
                    log_position: int
                    if metadata:
                        log_position = metadata['log_position'] or 0
                        fp.seek(log_position)
                    device_type_table: dataset.Table = db[device_type]
                    last_db_record: Optional[MutableMapping[str, Any]]
                    last_db_record = device_type_table.find_one(device_id=device_id, order_by='-timestamp')
                    last_record_datetime: Optional[datetime]
                    last_record_datetime = ensure_tz(last_db_record['timestamp']) if last_db_record else None
                    log_records: List[LogRecord]
                    log_records, *_ = cls.parse_file(fp, device_type=device_type, device_id=device_id,
                                                     last_known_date=last_record_datetime)
                    records_chunk: Sequence[LogRecord]
                    for records_chunk in chunked(log_records, 720):
                        # TODO: Sanity check all records must have timestamp
                        records_by_ts: MutableMapping[datetime, LogRecord] = {ensure_tz(r.timestamp): r
                                                                              for r in records_chunk}
                        records_chunk = list(records_by_ts.values())  # Merge same-timestamp records by latest
                        log_timestamps: List[datetime] = list(records_by_ts.keys())
                        existing_rows: Iterable[MutableMapping[str, Any]]
                        existing_rows = device_type_table.find(device_id=device_id, timestamp=log_timestamps)
                        existing_ts: Set[datetime] = {ensure_tz(row['timestamp']) for row in existing_rows}
                        records_for_db: MutableMapping[datetime, MutableMapping[str, Any]]
                        records_for_db = {record.timestamp: dict(device_id=device_id, **dataclasses.asdict(record))
                                          for record in records_chunk}
                        new_records: List[MutableMapping[str, Any]] = [r for ts, r in records_for_db.items()
                                                                       if ts not in existing_ts]
                        update_records: List[MutableMapping[str, Any]] = [r for ts, r in records_for_db.items()
                                                                          if ts in existing_ts]
                        device_type_table.insert_many(new_records, chunk_size=720, ensure=False)
                        for record in update_records:
                            device_type_table.upsert(record, keys=['device_id', 'timestamp'], ensure=False)
                        db.commit()
                    log_position = fp.tell()
                    if log_position > LOGS_ROTATE_SIZE:  # Check file rotation and truncate
                        # TODO: Store inside a 7z archive?
                        fp.seek(0)
                        fp.truncate()
                        log_position = 0
                        fp.flush()
                log_mtime = cls.file_mtime(path=log_path)
                metadata_table.upsert(dict(
                    device_type=device_type, device_id=device_id,
                    log_position=log_position, log_mtime=log_mtime
                ), keys=['device_type', 'device_id'])

    @classmethod
    def from_device(cls, device: Device, **filters) -> DeviceLog:
        """
        Creates a `DeviceLog` instance from a `Device` object,
        ensuring log database is up-to-date with the log file,
        and loading records applying any given filters.
        :param device: the `Device` object associated with the log
        :param filters: additional keyword arguments passed to `.parse_db(...)` to filter the records
        :return: the newly-created `DeviceLog` instance
        """
        device_type, device_id = device.full_device_id
        log_path: str = cls.file_path(device_type, device_id)
        cls.reflect_to_db(device_type=device_type, device_id=device_id)
        log: DeviceLog = cls(device=device, loaded_path=log_path, load_filters=filters)
        average_records_interval: Optional[float]
        average_sync_interval: Optional[float]
        log_records, average_records_interval, average_sync_interval = cls.parse_db(device_type=device_type,
                                                                                    device_id=device_id,
                                                                                    **filters)
        log.extend(log_records)
        if average_records_interval is not None:
            log.average_records_interval = average_records_interval
        if average_sync_interval is not None:
            log.average_sync_interval = average_sync_interval
        log.freeze()
        return log

    def __init__(self, *args, average_records_interval: Optional[float] = None,
                 average_sync_interval: Optional[float] = None, load_filters: Any = None, **kwargs):
        """
        Initialization for `DeviceLog`
        :param average_records_interval: the average interval between records,
                                         if omitted or None, the default interval value
                                         from the device's config is used instead
        :param average_sync_interval: the average interval between sync records,
                                      if omitted or None, the default interval value
                                      from the device's config is used instead
        :param load_filters: the filters used to load the log records in this instance, if any
        :param args: additional positional arguments to be passed to `super().__init__(...)`
        :param kwargs: additional keyword arguments to be passed to `super().__init__(...)`
        """
        super().__init__(*args, **kwargs)
        if isinstance(load_filters, dict):
            load_filters = list(load_filters.items())
        if isinstance(load_filters, list):
            load_filters = tuple(load_filters)
        self._load_filters: Hashable = load_filters
        if average_records_interval is None:
            average_records_interval = self.device.config.get('update_interval')
        self.average_records_interval: Optional[float] = average_records_interval
        if average_sync_interval is None:
            average_sync_interval = self.device.config.get('sync_interval')
        self.average_sync_interval: Optional[float] = average_sync_interval

    @property
    def unique_id(self) -> Any:
        return self.device.full_device_id, self.loaded_mtime, self._load_filters

    @property
    def device_type(self) -> str:
        """Shortcut property for the device type of the associated `Device` object"""
        return self.device.device_type

    @property
    def device_id(self) -> str:
        """Shortcut property for the device id of the associated `Device` object"""
        return self.device.device_id


DeviceFullID = Tuple[str, str]
DeviceOptionalID = Tuple[str, Optional[str]]
CT = TypeVar('CT', bound=DeviceConfig)
LT = TypeVar('LT', bound=DeviceLog)


# noinspection SyntaxError
class Device(ABC, Generic[CT, LT]):
    """
    Abstract base class that represents a device and its related data
    """
    _device_types_classes: ClassVar[MutableMapping[str, Type[Device]]] = {}
    _known_devices: ClassVar[MutableMapping[DeviceFullID, Device]] = {}
    _cached_configs: ClassVar[MutableMapping[DeviceOptionalID, CT]] = {}

    @staticmethod
    @abstractmethod
    def _get_device_type() -> str: 
        """
        Abstract static method that subclasses must implement
        to return the associated device type as string.
        N.B. if two subclasses are defined with the same device type,
             a NameError will be raised.
        """

    @staticmethod
    def _get_config_class() -> Type[DeviceConfig]:
        """
        Static method that returns the class used as `DeviceConfig`.
        """
        return DeviceConfig

    @staticmethod
    @abstractmethod
    def _get_log_class() -> Type[DeviceLog]:
        """
        Abstract static method that subclasses must implement to return
        the associated `DeviceLog` subclass for their specific device type.
        """

    def __init_subclass__(cls, **kwargs): 
        """
        Initializes a defined `Device` subclass, registering it's associated device type.
        :raise NameError: if a subclass is being initialized with the same device type as an another one
        """
        device_type: str = cls._get_device_type()
        existing_device_type_class: Type[Device] = cls._device_types_classes.get(device_type)
        if existing_device_type_class is not None:
            raise NameError(f'More than one {Device.__name__} class with same device type "{device_type}" defined: '
                            f'{existing_device_type_class.__name__} and {cls.__name__}')
        cls._device_types_classes[device_type] = cls

    @classmethod
    def get_class_for_device_type(cls, device_type) -> Type[Device]: 
        """
        Gets the `Device` class for the given device type.
        :param device_type: the device type for which to get the `Device` class
        :raise NameError: if no class for the given device type is found
        :return: the registered `Device` class for the device type
        """
        device_type_cls: Optional[Type[Device]] = cls._device_types_classes.get(device_type)
        if device_type_cls is None:
            raise KeyError(f'No {Device.__name__} subclass found for device type "{device_type}"')
        return device_type_cls

    @classmethod
    def for_type(cls, device_type: str, *args, **kwargs) -> Device: 
        """
        Creates a `Device` instance using the appropriate subclass for the given device type
        :param device_type: the device type for which to instantiate the correct `Device` subclass
        :param args: additional positional arguments to be passed to the `Device` subclass' `__init__(...)`
        :param kwargs: additional keyword arguments to be passed to the `Device` subclass' `__init__(...)`
        :return: the instantiated `Device` subclass' instance
        """
        device_cls = cls.get_class_for_device_type(device_type)
        return device_cls(*args, **kwargs)

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
class ThermLogRecord(TempsDeviceLogRecord, LogRecord):
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


# TODO: Refactor get_sched_temps in DeviceLog subclasses into a temps_set mixin

class ThermDeviceLog(DeviceLog):
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
class ThermDevice(Device[DeviceConfig, ThermDeviceLog]):
    """
    `Device` subclass for "therm" device types
    """
    @staticmethod
    def _get_device_type() -> str:
        return 'therm'

    @staticmethod
    def _get_log_class() -> Type[DeviceLog]:
        return ThermDeviceLog


@dataclasses.dataclass(frozen=True)
class TempStationLogRecord(TempsDeviceLogRecord, LogRecord):
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

    temp_set: Optional[float]
    """The set threshold temperature for the device at log record time,
    usually derived from the configured schedule"""

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


class TempStationDeviceLog(DeviceLog):
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
class TempStationDevice(Device[DeviceConfig, TempStationDeviceLog]):
    """
    `Device` subclass for "tempstation" device types
    """
    @staticmethod
    def _get_device_type() -> str:
        return 'tempstation'

    @staticmethod
    def _get_log_class() -> Type[DeviceLog]:
        return TempStationDeviceLog


TempsDeviceLog = Union[ThermDeviceLog, TempStationDeviceLog]
TempsDevice = Union[ThermDevice, TempStationDevice]
