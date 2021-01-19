"""Base classes for devices logs"""

from __future__ import annotations

import dataclasses
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import (ClassVar, MutableMapping, Type, Set, Iterator, Tuple, Optional, List, Any, ForwardRef, Union,
                    Iterable, Generator, Mapping, IO, ContextManager, Sequence, Hashable, TYPE_CHECKING)

import dataset
from dataset.util import ResultIter
from dateutil.parser import parse as dateutil_parser
from sqlalchemy import Integer, UnicodeText, Float, Boolean, DateTime, Column
from sqlalchemy.sql.type_api import TypeEngine

from ... import app
from ...base import UniqueHashable, LiveFileData, ensure_tz, chunked
from ...config import LOGS_PATH, LOGS_EXT, LOGS_DB_PATH, LOGS_ROTATE_SIZE

from .common import DeviceBound, iterate_type_id_paths, DeviceTypeRegistrar

if TYPE_CHECKING:
    from .config import DeviceConfig
    from .device import Device


__all__ = [
    "LogRecord",
    "DeviceLog",
]


# TODO: Refactor some methods to remove redundant device_type args
# pylint: disable=invalid-name
@dataclasses.dataclass(frozen=True)
class LogRecord(DeviceTypeRegistrar, ABC):
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
        from .device import Device

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
