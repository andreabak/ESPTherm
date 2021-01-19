"""Module with server routes used by devices for synchronization and data upload"""

from calendar import timegm
from datetime import datetime, timedelta
from typing import Optional, Dict, MutableMapping, Any, Sequence

from flask import request, jsonify, abort, Response

from . import app
from .config import server_config
from .devices.base import DeviceConfig, LogRecord, DeviceLog, Device


@app.route('/<string:device_type>/upload_log', methods=['POST'])
def upload_log(device_type: str, device_id: Optional[str] = None) -> str:
    """
    Route for log data upload by a device.
    The route endpoint is `/<string:device_type>/upload_log?deviceid=<device_id>`,
    the request method must be POST, and the request's data must be the log data being uploaded.
    :param device_type: the device type
    :param device_id: the device id, optionally specified by another caller method,
                      otherwise derived from url params.
    :return: the response for the device, a string of the length of the received raw data
    """
    if not device_id:
        device_id = request.args.get('deviceid')
    if not device_id:
        abort(400)
    raw_data: bytes = request.data
    saved_data: str = DeviceLog.upload(raw_data=raw_data, device_type=device_type, device_id=device_id)
    app.logger.info(f'Received {device_type} log from {request.remote_addr}:\n{saved_data.strip()}')
    config_data: DeviceConfig = Device.get_config(device_type, device_id)
    next_sync_time: datetime = datetime.now() + timedelta(seconds=config_data["sync_interval"])
    next_sync_time = next_sync_time.replace(microsecond=0)
    webrepl_cfg: Optional[MutableMapping[str, Any]] = config_data.get('webrepl')
    if webrepl_cfg and isinstance(webrepl_cfg, dict) and webrepl_cfg.get('enabled'):
        webrepl_time_window: int = webrepl_cfg["time_window"]
        app.logger.warning(f'WEBREPL is enabled, device will be listening at {config_data["ifconfig"]["address"]}:8266 '
                           f'for {webrepl_time_window} seconds from now')
        next_sync_time += timedelta(seconds=webrepl_time_window)
    app.logger.info(f'Next sync time expected after {next_sync_time.isoformat()}')
    return str(len(raw_data))


NTP_DELTA: int = timegm((2000, 1, 1, 0, 0, 0, 5, 1, 0))
"""Micropython embedded-to-unix epoch seconds delta"""


@app.route('/<string:device_type>/sync_config')
def sync_config(device_type: str) -> Response:
    """
    Route for config synchronization by a device.
    The route endpoint is `/<string:device_type>/sync_config`.
    Optional url parameters are:
    - `deviceid`: the device id, if omitted the default config for the device type is used
    - `only_timestamp`: sends only the timestamp of the local (server's) config file
                        instead of all the config's data
    :param device_type: the device type
    :return: the response for the device, json data with an object with the config's mtime and (optionally) its data
    """
    only_timestamp: bool = 'only_timestamp' in request.args
    device_id: Optional[str] = request.args.get('deviceid')

    device_config: DeviceConfig = Device.get_config(device_type, device_id)
    config_mtime: float = device_config.loaded_mtime

    mtime = int(config_mtime - NTP_DELTA)
    data: Dict = {'mtime': mtime}
    if not only_timestamp:
        data['config'] = dict(device_config)

    return jsonify(data)


class RemoteThermException(Exception):
    """Exception class used to signal remote therm config/log exceptions"""


class RemoteThermDisabled(RemoteThermException):
    """Exception class used to signal that remote therm is disabled"""


@app.route('/<string:device_type>/state_set')
def state_set(device_type: str) -> Response:
    """
    Route for state synchronization by a device.
    Can be used to remotely set states to a device (like thermostat switching)
    The route endpoint is `/<string:device_type>/state_set`.
    Optional url parameters are:
    - `deviceid`: the device id, if omitted some remote state settings might not be applied
    :param device_type: the device type
    :return: the response for the device, json data with an object of state values overrides or empty
    """
    device_id: Optional[str] = request.args.get('deviceid')

    data: Dict = {}

    # Check remote therm switch linking
    if device_type == 'therm':
        try:
            # pylint: disable=no-member  # pylint issue #3684
            remote_therm_cfg: MutableMapping[str, Any] = server_config.get('remote_therm')
            if not remote_therm_cfg:
                raise RemoteThermDisabled
            remote_therm_enabled: bool = remote_therm_cfg.get('enabled', False)
            if not remote_therm_enabled:
                raise RemoteThermDisabled
            switch_link_full_id: Optional[str] = remote_therm_cfg.get('switch_link_id')
            if not switch_link_full_id:
                raise RemoteThermException('Remote therm mode is enabled, but no "switch_link_id" is configured')
            link_device_type: str
            link_device_id: str
            link_device_type, link_device_id = switch_link_full_id.split('/')
            try:
                link_device: Device = Device.get_known_device(link_device_type, link_device_id)
            except KeyError as exc:
                raise RemoteThermException(f'Remote therm has switch_link_id="{switch_link_full_id}", '
                                           f'but no known device with that id exists') from exc
            link_device_max_log_age: int = remote_therm_cfg.get('max_log_age', 30)
            recent_log: DeviceLog = link_device.get_log(
                timestamp_min=datetime.now() - timedelta(minutes=link_device_max_log_age))
            if not recent_log:
                raise RemoteThermException(f'Remote therm linked device {repr(link_device)} '
                                           f'has no log data within the last {link_device_max_log_age} minutes')
            link_last_log: LogRecord = recent_log[-1]
            link_last_temp: Optional[float] = getattr(link_last_log, 'temp_average', None)
            link_set_temp: Optional[float] = getattr(link_last_log, 'temp_set', None)
            bad_attrs: Sequence[str] = [f'"{a}"' for a, v in (('temp_average', link_last_temp),
                                                              ('temp_set', link_set_temp))
                                        if v is None]
            if bad_attrs:
                raise RemoteThermException(f'Remote therm linked device {repr(link_device)} '
                                           f'has no valid {" nor ".join(bad_attrs)} in its last log record')
            remote_switch_state: bool = link_last_temp < link_set_temp
            data['operating_mode'] = 'remote'
            data['therm_switch'] = remote_switch_state

        except RemoteThermException as exc:
            if not isinstance(exc, RemoteThermDisabled):
                app.logger.warning(f'Remote therm mode error, reverting to schedule operating mode:\n{exc}')
            data['operating_mode'] = 'schedule'

    return jsonify(data)
