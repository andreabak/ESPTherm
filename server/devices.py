from __future__ import annotations

import json
import os
import dataclasses
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import ClassVar, MutableMapping, Type, Optional, List, Tuple, Union, Any, Iterator, TypeVar, IO, Generator, \
    Set, ContextManager, Iterable, Mapping, ForwardRef, Hashable

import dataset
from dataset.util import ResultIter
from sqlalchemy import Integer, UnicodeText, Float, Boolean, DateTime, Column
from sqlalchemy.sql.type_api import TypeEngine
from dateutil.parser import parse as dateutil_parser

from . import app
from .base import LiveFileData, UniqueHashable, ensure_tz, chunked
from .config import LOGS_PATH, LOGS_EXT, CONFIGS_PATH, CONFIGS_EXT, DEFAULT_CONFIG_BASENAME, LOGS_DB_PATH, \
    LOGS_ROTATE_SIZE


def iterate_type_id_paths(base_path: str) -> Iterator[Tuple[str, str]]:
    for device_type in os.listdir(base_path):
        device_type_dir = os.path.join(base_path, device_type)
        if os.path.isdir(device_type_dir):
            for filename in os.listdir(device_type_dir):
                filepath = os.path.join(device_type_dir, filename)
                device_id, ext = os.path.splitext(filename)
                if os.path.isfile(filepath) and ext == LOGS_EXT:
                    yield device_type, device_id


class DeviceConfig(UniqueHashable, LiveFileData, dict):
    @classmethod
    def file_path(cls, device_type: str, device_id: Optional[str] = None) -> str:
        config_basename: str = DEFAULT_CONFIG_BASENAME if device_id is None else device_id
        config_path: str = os.path.join(CONFIGS_PATH, device_type, config_basename + CONFIGS_EXT)
        return config_path

    @staticmethod
    def iterate_available_configs() -> Iterator[Tuple[str, str]]:
        for device_type, device_id in iterate_type_id_paths(CONFIGS_PATH):
            if device_id == DEFAULT_CONFIG_BASENAME:
                device_id = None
            yield device_type, device_id

    @classmethod
    def device_id_config_exists(cls, device_type: str, device_id: str):
        config_path: str = cls.file_path(device_type, device_id)
        return os.path.exists(config_path)

    @classmethod
    def from_type_id(cls, device_type: str, device_id: Optional[str] = None) -> DeviceConfig:
        config_path: str = cls.file_path(device_type, device_id)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Device config file does not exist: {config_path}')
        config_mtime: float = cls.file_mtime(config_path)
        config: DeviceConfig = cls(device_type=device_type, device_id=device_id,
                                   loaded_path=config_path, loaded_mtime=config_mtime)
        with open(config_path, 'r') as fp:
            config_data: MutableMapping[str, Any] = json.load(fp)
        config.update(config_data)
        config.freeze()
        return config

    @classmethod
    def from_device(cls, device: Device) -> DeviceConfig:
        device_type, device_id = device.full_device_id
        if not cls.device_id_config_exists(device_type, device_id):
            device_id = None
        return cls.from_type_id(device_type, device_id)

    def __init__(self, device_type: str, device_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device_type: str = device_type
        self.device_id: str = device_id

    @property
    def unique_id(self) -> Any:
        return (self.device_type, self.device_id), self.loaded_mtime


@dataclasses.dataclass(frozen=True)
class LogRecord(ABC):
    DB_TYPES_MAP: ClassVar[MutableMapping[Type, TypeEngine]] = {
        int: Integer,
        str: UnicodeText,
        float: Float,
        bool: Boolean,
        datetime: DateTime,
    }
    DB_PKEY_FIELDS: ClassVar[Set[str]] = {'timestamp'}
    DB_IDX_FIELDS: ClassVar[Set[str]] = {'timestamp'}
    _device_types_classes: ClassVar[MutableMapping[str, Type[LogRecord]]] = {}

    @staticmethod
    @abstractmethod
    def _device_type() -> str: ...

    @classmethod
    def get_known_types_classes(cls) -> Iterator[Tuple[str, Type[LogRecord]]]:
        for k, v in cls._device_types_classes.items():
            yield k, v

    def __init_subclass__(cls, **kwargs):
        device_type: str = cls._device_type()
        existing_device_type_class: Type[LogRecord] = cls._device_types_classes.get(device_type)
        if existing_device_type_class is not None:
            raise NameError(f'More than one {LogRecord.__name__} class with same device type "{device_type}" defined: '
                            f'{existing_device_type_class.__name__} and {cls.__name__}')
        cls._device_types_classes[device_type] = cls

    @classmethod
    def cls_for_device_type(cls, device_type) -> Type[LogRecord]:
        log_class: Optional[Type[LogRecord]] = cls._device_types_classes.get(device_type)
        if log_class is None:
            raise NameError(f'No log record class found for device type "{device_type}"')
        return log_class

    @classmethod
    def for_type(cls, device_type: str, *args, **kwargs) -> LogRecord:
        log_class = cls.cls_for_device_type(device_type)
        # noinspection PyArgumentList
        return log_class(*args, **kwargs)

    @classmethod
    @abstractmethod
    def from_raw_logline(cls, raw_line: str) -> LogRecord:
        ...

    @staticmethod
    def parse_timestamp(timestamp_raw: str, timestamp_override: Optional[datetime] = None) -> datetime:
        timestamp: datetime
        if timestamp_override is None:
            parsed_timestamp: Optional[datetime] = dateutil_parser(timestamp_raw) if timestamp_raw else None
            if not parsed_timestamp:
                raise ValueError(f'Missing timestamp')
            timestamp = parsed_timestamp
        else:
            timestamp = timestamp_override
        return timestamp

    @classmethod
    def fields_to_db_columns(cls) -> List[Column]:
        columns: List[Column] = []
        field: dataclasses.Field
        for field in dataclasses.fields(cls):
            ftype: Any = field.type
            if isinstance(ftype, str):
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
    timestamp_missing: bool
    wlan_active: bool
    battery_voltage: Optional[float]

    def copy(self, **overrides) -> LogRecord:
        values: MutableMapping[str, Any] = dataclasses.asdict(self)
        values.update(overrides)
        return self.__class__(**values)


class DeviceBound(ABC):
    def __init__(self, device: Device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device: Device = device

    @property
    def device(self) -> Device:
        return self._device


class DeviceLog(UniqueHashable, DeviceBound, LiveFileData, list, ABC):
    db_thread_lock: RLock = RLock()

    @staticmethod
    @abstractmethod
    def get_logrecord_class() -> Type[LogRecord]: ...

    @classmethod
    def file_path(cls, device_type: str, device_id: str) -> str:
        log_path: str = os.path.join(LOGS_PATH, device_type, device_id + LOGS_EXT)
        return log_path

    @staticmethod
    def iterate_available_logs() -> Iterator[Tuple[str, str]]:
        # TODO: Add DB logs
        yield from iterate_type_id_paths(LOGS_PATH)

    @classmethod
    def ensure_db_schema(cls, db: dataset.Database):
        db.query('PRAGMA journal_mode = WAL').close()
        device_type: str
        log_record_type: Type[LogRecord]
        for device_type, log_record_type in LogRecord.get_known_types_classes():
            table_columns: List[Column] = [Column('device_id', UnicodeText, primary_key=True)] \
                                          + log_record_type.fields_to_db_columns()
            device_type_table: dataset.Table = db.create_table(device_type, primary_id=False)
            # noinspection PyProtectedMember
            device_type_table._sync_table(table_columns)
        meta_table: dataset.Table = db.create_table('_meta', primary_id=False)
        # noinspection PyProtectedMember
        meta_table._sync_table([
            Column('device_type', UnicodeText, nullable=False, primary_key=True),
            Column('device_id', UnicodeText, nullable=False, primary_key=True),
            Column('log_position', Integer, nullable=False),
            Column('log_mtime', Float, nullable=False),
        ])

    @classmethod
    @contextmanager
    def logs_db_context(cls) -> Iterator[dataset.Database]:
        with cls.db_thread_lock:
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
    def parse_records(cls, str_or_iterable: Union[str, Iterable], device_type: str, device_id: Optional[str] = None,
                      interval_avg_max_deviation: float = 0.2, ema_alpha: float = 0.25) \
            -> Generator[LogRecord, None, Tuple[Optional[float], Optional[float]]]:

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

        log_record_cls: Type[LogRecord] = LogRecord.cls_for_device_type(device_type)

        last_date: Optional[datetime] = None
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
                    # TODO: lineno from db rowid?
                    log_record = log_record_cls(**record)
                else:
                    raise TypeError(f'Unknown record type for log: {repr(record)}')

            except Exception as exc:
                app.logger.warning(f'Error while parsing log line {n} (skipped):\n{record}\n{exc}')

            else:
                if last_date is None and log_record.timestamp_missing:
                    raise ValueError(f'Missing log dates from the start')
                if log_record.timestamp_missing:
                    estimated_records_interval: Optional[float] = average_records_interval or expected_sync_interval
                    if estimated_records_interval is None:
                        raise ValueError(f'Missing log date and couldn\'t get update interval to estimate')
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
    def upload(cls, raw_data: bytes, device_type: str, device_id: str, encoding: str = 'utf8') -> str:
        log_path: str = cls.file_path(device_type, device_id)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_data: str = raw_data.decode(encoding)
        if not log_data.endswith('\n'):
            log_data += '\n'
        with open(log_path, 'a') as fp:
            fp.write(log_data)
        return log_data

    @classmethod
    def parse_iter(cls, iterable: Iterable[Any], device_type: str, device_id: str):
        average_records_interval: float
        average_sync_interval: float
        log_records: List[LogRecord] = []
        records_generator: Generator[LogRecord, None, Tuple[Optional[float], Optional[float]]]
        records_generator = cls.parse_records(iterable, device_type=device_type, device_id=device_id)
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
    def parse_file(cls, file_path_or_fp: Union[str, IO], device_type: str, device_id: str):
        file_context: ContextManager[IO]
        if isinstance(file_path_or_fp, str):
            file_context = open(file_path_or_fp, 'r')
        else:
            file_context = nullcontext(file_path_or_fp)
        with file_context as fp:
            return cls.parse_iter(fp, device_type=device_type, device_id=device_id)

    @classmethod
    def parse_db(cls, device_type: str, device_id: str,
                 timestamp_min: Optional[datetime] = None, timestamp_max: Optional[datetime] = None,
                 bound_exclusive: Optional[bool] = None):
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
            return cls.parse_iter(db_results, device_type=device_type, device_id=device_id)

    @classmethod
    def reflect_to_db(cls, device_type: str, device_id: str):
        db: dataset.Database
        with cls.logs_db_context() as db:
            metadata_table: dataset.Table = db['_meta']
            metadata: Optional[MutableMapping[str, Any]]
            metadata = metadata_table.find_one(device_type=device_type, device_id=device_id)

            log_path: str = cls.file_path(device_type, device_id)
            log_mtime: float = cls.file_mtime(log_path)
            if not metadata or log_mtime > metadata['log_mtime']:  # Reflect if file is newer
                with open(log_path, 'r+') as fp:
                    log_position: int
                    if metadata:
                        log_position = metadata['log_position'] or 0
                        fp.seek(log_position)
                    device_type_table: dataset.Table = db[device_type]
                    log_records: List[LogRecord]
                    log_records, *_ = cls.parse_file(fp, device_type=device_type, device_id=device_id)
                    records_chunk: List[LogRecord]
                    for records_chunk in chunked(log_records, 720):
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
                        fp.seek(0)
                        fp.truncate()
                        log_position = 0
                        fp.flush()
                log_mtime = cls.file_mtime(log_path)
                metadata_table.upsert(dict(
                    device_type=device_type, device_id=device_id,
                    log_position=log_position, log_mtime=log_mtime
                ), keys=['device_type', 'device_id'])

    @classmethod
    def from_device(cls, device: Device, **filters) -> DeviceLog:
        device_type, device_id = device.full_device_id
        log_path: str = cls.file_path(device_type, device_id)
        cls.reflect_to_db(device_type=device_type, device_id=device_id)
        log: DeviceLog = cls(device=device, loaded_path=log_path, load_filters=filters)
        average_records_interval: float
        average_sync_interval: float
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

    def __init__(self, average_records_interval: Optional[float] = None, average_sync_interval: Optional[float] = None,
                 load_filters: Any = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(load_filters, dict):
            load_filters = [(k, v) for k, v in load_filters.items()]
        if isinstance(load_filters, list):
            load_filters = tuple(load_filters)
        self._load_filters: Hashable = load_filters
        if average_records_interval is None:
            average_records_interval = self.device.config.get('update_interval')
        self.average_records_interval: float = average_records_interval
        if average_sync_interval is None:
            average_sync_interval = self.device.config.get('sync_interval')
        self.average_sync_interval: float = average_sync_interval

    @property
    def unique_id(self) -> Any:
        return self.device.full_device_id, self.loaded_mtime, self._load_filters

    @property
    def device_type(self) -> str:
        return self.device.device_type

    @property
    def device_id(self) -> str:
        return self.device.device_id

    # TODO: Override uptodate with if check for mode
    # TODO: Add unified interface to check log start/end timestamps


DeviceFullID = Tuple[str, str]
DeviceOptionalID = Tuple[str, Optional[str]]
_CT = TypeVar('_CT', bound=DeviceConfig)
_LT = TypeVar('_LT', bound=DeviceLog)


class Device(ABC):
    _device_types_classes: ClassVar[MutableMapping[str, Type[Device]]] = {}
    _known_devices: ClassVar[MutableMapping[DeviceFullID, Device]] = {}
    _cached_configs: ClassVar[MutableMapping[DeviceOptionalID, _CT]] = {}

    @staticmethod
    @abstractmethod
    def _get_device_type() -> str: ...

    @staticmethod
    def _get_config_class() -> Type[_CT]:
        return DeviceConfig

    @staticmethod
    @abstractmethod
    def _get_log_class() -> Type[_LT]: ...

    def __init_subclass__(cls, **kwargs):
        device_type: str = cls._get_device_type()
        existing_device_type_class: Type[Device] = cls._device_types_classes.get(device_type)
        if existing_device_type_class is not None:
            raise NameError(f'More than one {Device.__name__} class with same device type "{device_type}" defined: '
                            f'{existing_device_type_class.__name__} and {cls.__name__}')
        cls._device_types_classes[device_type] = cls

    @classmethod
    def get_class_for_device_type(cls, device_type) -> Type[Device]:
        device_type_cls: Optional[Type[Device]] = cls._device_types_classes.get(device_type)
        if device_type_cls is None:
            raise KeyError(f'No {Device.__name__} subclass found for device type "{device_type}"')
        return device_type_cls

    @classmethod
    def for_type(cls, device_type: str, *args, **kwargs) -> Device:
        device_cls = cls.get_class_for_device_type(device_type)
        return device_cls(*args, **kwargs)

    @classmethod
    def _register_instance(cls, instance: Device):
        instance_full_id: DeviceFullID = instance.full_device_id
        conflicting_device: Optional[Device] = cls._known_devices.get(instance_full_id)
        if conflicting_device is not None:
            raise ValueError(f'Instantiating new device {repr(instance)} with same full id "{instance_full_id}" '
                             f'as already existing instance {repr(conflicting_device)}')
        cls._known_devices[instance_full_id] = instance

    @classmethod
    def iterate_known_devices(cls) -> Iterator[Device]:
        for device in cls._known_devices.values():
            yield device

    @classmethod
    def get_known_device(cls, device_type: str, device_id: str) -> Device:
        return cls._known_devices[(device_type, device_id)]

    @classmethod
    def load_logged_devices(cls):
        for device_type, device_id in DeviceLog.iterate_available_logs():
            cls.for_type(device_type=device_type, device_id=device_id, preload=True)

    @classmethod
    def get_config(cls, device_type: str, device_id: Optional[str]):
        if not DeviceConfig.device_id_config_exists(device_type, device_id):
            device_id = None
        full_id = device_type, device_id
        cached_config: Optional[_CT] = cls._cached_configs.get(full_id)
        if cached_config is None or not cached_config.uptodate:
            cls._cached_configs[full_id] = DeviceConfig.from_type_id(device_type, device_id)
        return cls._cached_configs[full_id]

    @classmethod
    def load_known_configs(cls):
        for device_type, device_id in DeviceConfig.iterate_available_configs():
            cls.get_config(device_type, device_id)

    @classmethod
    def load_known_devices(cls):
        cls.load_known_configs()
        cls.load_logged_devices()

    def __init__(self, device_id: str, preload: bool = False):
        self._device_id: str = device_id
        self._config: Optional[_CT] = None

        conflicting_device: Optional[Device] = self.__class__._known_devices.get(self.full_device_id)
        if conflicting_device is not None:
            raise ValueError(f'Instantiating new device {repr(self)} with same full id "{self.full_device_id}" '
                             f'as already existing instance {repr(conflicting_device)}')
        self.__class__._known_devices[self.full_device_id] = self

        if preload:
            self._load_config()
            self._get_log_class().reflect_to_db(device_type=self.device_type, device_id=self.device_id)

    @property
    def device_type(self) -> str:
        return self.__class__._get_device_type()

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def full_device_id(self) -> DeviceFullID:
        return self.device_type, self._device_id

    def _load_config(self):
        self._config = self.get_config(self.device_type, self.device_id)

    @property
    def config(self) -> _CT:
        if self._config is None or not self._config.uptodate:
            self._load_config()
        return self._config

    def get_log(self, **filters) -> _LT:
        return self._get_log_class().from_device(self, **filters)

    @property
    def full_log(self) -> _LT:
        return self.get_log()

    @property
    def log_daily(self) -> _LT:
        return self.get_log(timestamp_min=datetime.now() - timedelta(hours=24))  # TODO: UTC conversion


@dataclasses.dataclass(frozen=True)
class TempsDeviceLogRecordMixin:
    temp_current: Optional[float]
    temp_average: Optional[float]


@dataclasses.dataclass(frozen=True)
class ThermLogRecord(TempsDeviceLogRecordMixin, LogRecord):
    @staticmethod
    def _device_type() -> str:
        return 'therm'

    operating_mode: str
    temp_set: Optional[float]
    therm_state: bool
    tampered: bool
    grace_given: bool

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


class ThermDeviceLog(DeviceLog):
    @staticmethod
    @abstractmethod
    def get_logrecord_class() -> Type[LogRecord]:
        return ThermLogRecord

    @staticmethod
    def get_sched_temps(config: MutableMapping[str, Any]) -> List[float]:
        return config['thermostat']['sched_temps']


class ThermDevice(Device):
    @staticmethod
    def _get_device_type() -> str:
        return 'therm'

    @staticmethod
    def _get_log_class() -> Type[DeviceLog]:
        return ThermDeviceLog


@dataclasses.dataclass(frozen=True)
class TempStationLogRecord(TempsDeviceLogRecordMixin, LogRecord):
    @staticmethod
    def _device_type() -> str:
        return 'tempstation'

    humidity_current: Optional[float]
    humidity_average: Optional[float]
    temp_set: Optional[float]

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
    @staticmethod
    @abstractmethod
    def get_logrecord_class() -> Type[LogRecord]:
        return TempStationLogRecord

    @staticmethod
    def get_sched_temps(config: MutableMapping[str, Any]) -> List[float]:
        return config['station']['sched_temps']


class TempStationDevice(Device):
    @staticmethod
    def _get_device_type() -> str:
        return 'tempstation'

    @staticmethod
    def _get_log_class() -> Type[DeviceLog]:
        return TempStationDeviceLog


TempsDeviceLog = Union[ThermDeviceLog, TempStationDeviceLog]
TempsDevice = Union[ThermDevice, TempStationDevice]
