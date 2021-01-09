import json
import math
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from inspect import signature, Parameter
from typing import List, MutableMapping, Optional, Tuple, Any, Callable, Iterator, Union, Sequence

import numpy as np
import plotly
from plotly import graph_objs as plotly_go
from plotly.basedatatypes import BaseTraceType
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter, find_peaks
from dateutil.parser import parse as dateutil_parser
from flask import request, url_for, jsonify, render_template, send_from_directory, redirect, Response

from . import app, compress
from .config import server_config
from .base import ensure_tz
from .devices import DeviceLog, ThermDeviceLog, TempStationDeviceLog, TempsDeviceLog, Device, DeviceConfig


SESSION_COOKIE_NAME: str = 'flasksession'

MAIN_DEVICE: Tuple[str, str] = ('therm', 'main')


@dataclass
class UserSession:
    valid: bool = True


authed_sessions: MutableMapping[str, UserSession] = {}


def get_session_cookie() -> Optional[str]:
    return request.cookies.get(SESSION_COOKIE_NAME)


def check_session_valid() -> bool:
    global authed_sessions
    cookie_id: Optional[str] = get_session_cookie()
    if not cookie_id or cookie_id not in authed_sessions:
        return False
    return authed_sessions[cookie_id].valid


def create_session_cookie(response: Optional[Response] = None) -> str:
    global authed_sessions
    session_id: str = uuid.uuid4().hex
    if response is not None:
        response.set_cookie(SESSION_COOKIE_NAME, session_id)
    authed_sessions[session_id] = UserSession()
    return session_id


@app.route('/auth', methods=['GET', 'POST'])
@compress.compressed()
def auth_page() -> Response:
    if request.method == 'POST':  # Check auth
        response: MutableMapping[str, Any] = {}
        received_code: Optional[str] = request.form.get('secret_code') if request.form else None
        if received_code and received_code == server_config['ui_secret_code']:
            response['status'] = 'SUCCESS'
            response['redirect'] = url_for('homepage')
            session_id: str = create_session_cookie()
            response['session_cookie'] = SESSION_COOKIE_NAME
            response['session_id'] = session_id
        else:
            response['status'] = 'ERROR'
        return jsonify(response)

    if check_session_valid():
        return redirect(url_for('homepage'))

    return render_template('pinpad.jinja2')


def format_time(seconds: float, precision: int = 2, with_seconds: bool = True, with_letters: bool = True) -> str:
    """
    Format a duration in seconds into a timer-like string
    :param seconds: The number of seconds
    :param precision: The amount of digits for the decimal part of the seconds
    :param with_seconds: If False, exclude seconds from the output. Default True
    :param with_letters: If True, use letter formatting instead of colons. Default True
    :return: The formatted "timer" string
    """
    s_fmt_l: int = precision + 3 if precision else 2
    s_fmt_spec_full: str = f'{{:0{s_fmt_l}.{precision}f}}'
    s_fmt_spec_min: str = f'{{:.{precision}f}}'
    h: int = int(seconds) // 3600
    m: int = (int(seconds) // 60) % 60
    s: float = seconds % 60
    h_fmt: str = f'{h:d}'
    m_fmt: str = ('{:02d}' if seconds >= 3600 else '{:d}').format(m)
    s_fmt: str = (s_fmt_spec_full if seconds >= 60 else s_fmt_spec_min).format(s) + ('s' if with_letters else '')
    separator: str = ' ' if with_letters else ':'
    ms_pre_fmt: str = f'{m_fmt}m' if with_letters else f'{m_fmt}'
    ms_fmt: str = separator.join([ms_pre_fmt] + ([s_fmt] if with_seconds else []))
    hms_fmt: str = separator.join((f'{h_fmt}h' if with_letters else f'{h_fmt}', ms_fmt))
    timer: str = (hms_fmt if seconds >= 3600 else (ms_fmt if (not with_seconds or seconds >= 60) else s_fmt))
    return timer


def get_percentile_range(data: np.ndarray, delta: float = 1.0) -> Tuple[Optional[float], Optional[float]]:
    if not data.size:
        return None, None
    return np.percentile(data, 0.0 + delta), np.percentile(data, 100.0 - delta)


PlotFnResult = Any


@dataclass
class CachedPlotFigure:
    cache_key: Tuple
    log: DeviceLog
    log_mtime: float
    results: PlotFnResult


plot_figures_cache = {}


def plot_figures_cache_cleanup():
    # Cleanup cache
    k: Tuple
    pf: CachedPlotFigure
    for k, pf in list(plot_figures_cache.items()):
        if not pf.log.uptodate:
            del plot_figures_cache[k]


def log_cacheable(plot_fn: Callable[..., PlotFnResult]) -> Callable[..., PlotFnResult]:
    fn_sig = signature(plot_fn)
    required_kwargs = ('log', )
    for kwarg in required_kwargs:
        try:
            if kwarg not in fn_sig.parameters:
                raise AttributeError(f'missing required "{kwarg}" keyword parameter')
            param = fn_sig.parameters[kwarg]
            if param.kind != Parameter.KEYWORD_ONLY:
                raise AttributeError(f'"{kwarg}" must be a keyword-only parameter')
            if param.default != Parameter.empty:
                raise AttributeError(f'"{kwarg}" parameter must not have any default value')
        except AttributeError as exc:
            raise AttributeError(f'plot function {repr(plot_fn)} decorated with "log_cacheable" {exc}.\n'
                                 f'Got signature: {fn_sig}')

    @wraps(plot_fn)
    def wrapper(*args, **kwargs):
        nonlocal plot_fn

        def mkhash(obj):
            return hash(obj) if not isinstance(obj, np.ndarray) else id(obj)

        log: DeviceLog = kwargs['log']
        log_mtime: float = log.loaded_mtime
        cache_key: Tuple = (log.device.full_device_id, plot_fn.__name__, str(id(plot_fn)),
                            *[mkhash(a) for a in args], '*',
                            *[(k, mkhash(v)) for k, v in kwargs.items()])
        cached_plot_figure: Optional[CachedPlotFigure] = plot_figures_cache.get(cache_key)
        if cached_plot_figure is not None and cached_plot_figure.log_mtime == log_mtime:
            return cached_plot_figure.results

        results: PlotFnResult = plot_fn(*args, **kwargs)
        plot_figures_cache[cache_key] = CachedPlotFigure(cache_key=cache_key, log=log, log_mtime=log_mtime,
                                                         results=results)
        return results

    return wrapper


def log_device_full_id(log: DeviceLog):
    return '/'.join(log.device.full_device_id)


@log_cacheable
def make_timeaxis(*, log: DeviceLog, non_roundable_interval: float = 90) -> np.ndarray:
    timeaxis = np.array([ensure_tz(record.timestamp).astimezone(tz=None).timestamp() for record in log])
    non_roundable = np.where(timeaxis[1:] - timeaxis[:-1] > non_roundable_interval)[0]
    non_roundable = list(np.union1d(non_roundable, non_roundable+1).tolist()) + [0, timeaxis.size - 1]
    roundable = np.setdiff1d(np.arange(timeaxis.size), np.array(non_roundable))
    timeaxis[roundable] = np.round(timeaxis[roundable] / 60.0) * 60.0
    return np.array([datetime.fromtimestamp(ts) for ts in timeaxis])


@log_cacheable
def make_np_array(*, log: DeviceLog, field_name: str, cast: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
    def iterlog() -> Iterator[float]:
        nonlocal log
        for record in log:
            value = getattr(record, field_name)
            yield (cast(value) if cast is not None else value) if value is not None else float('-inf')

    values: np.ndarray = np.fromiter(iterlog(), dtype='float')
    mask: np.ndarray = np.where(values != float('-inf'))[0]
    return values, mask


@log_cacheable
def make_masked_arrays(*arrays: np.ndarray, mask: np.ndarray, log: DeviceLog) -> Sequence[np.ndarray]:
    return tuple(a[mask] for a in arrays)


@log_cacheable
def mktrace_temps_avg(*, log: TempsDeviceLog, timeaxis: np.ndarray, hue: float = 210, line_dash: str = 'solid',
                      opacity: float = 1.0) \
        -> Tuple[BaseTraceType, np.ndarray, np.ndarray, Tuple[float, float]]:
    temps_avg, temps_avg_mask = make_np_array(log=log, field_name='temp_average')
    timeaxis_masked: np.ndarray
    temps_avg_masked: np.ndarray
    timeaxis_masked, temps_avg_masked = make_masked_arrays(timeaxis, temps_avg, mask=temps_avg_mask, log=log)
    if temps_avg_mask.size:
        trace = plotly_go.Scatter(
            x=timeaxis_masked, y=temps_avg_masked, yaxis='y2',
            name=f'{log.device_id} temp',
            meta=dict(id=f'{log_device_full_id(log)}.temp_avg'),
            legendgroup=log_device_full_id(log),
            line=dict(shape='spline', width=2, color=f'hsla({hue:.0f}, 100%, 75%, 0.75)', dash=line_dash),
            opacity=opacity
        )
    else:
        trace = None
    return trace, temps_avg_masked, timeaxis_masked, get_percentile_range(temps_avg_masked)


@log_cacheable
def mktrace_rhs_avg(*, log: TempStationDeviceLog, timeaxis: np.ndarray, hue: float = 210, line_dash: str = 'solid',
                    opacity: float = 1.0) \
        -> Tuple[BaseTraceType, np.ndarray, np.ndarray, Tuple[float, float]]:
    rhs_avg, rhs_avg_mask = make_np_array(log=log, field_name='humidity_average')
    timeaxis_masked: np.ndarray
    rhs_avg_masked: np.ndarray
    timeaxis_masked, rhs_avg_masked = make_masked_arrays(timeaxis, rhs_avg, mask=rhs_avg_mask, log=log)
    if rhs_avg_mask.size:
        trace = plotly_go.Scatter(
            x=timeaxis_masked, y=rhs_avg_masked, yaxis='y3',
            name=f'{log.device_id} rh',
            meta=dict(id=f'{log_device_full_id(log)}.rh_avg'),
            legendgroup=log_device_full_id(log),
            line=dict(shape='spline', width=2, color=f'hsla({hue:.0f}, 100%, 75%, 0.75)', dash=line_dash),
            opacity=opacity
        )
    else:
        trace = None
    return trace, rhs_avg_masked, timeaxis_masked, get_percentile_range(rhs_avg_masked)


@log_cacheable
def mktrace_houravg(*, log: TempsDeviceLog, values: np.ndarray, timeaxis: np.ndarray,
                    data_name: str, yaxis: str = 'y2', hue: float = 210, line_dash: str = 'solid',
                    opacity: float = 1.0) \
        -> Tuple[BaseTraceType, np.ndarray, Tuple[float, float]]:
    filter_size: int = min(max(0, math.ceil(values.size / 2) * 2 - 1), 59)
    values_houravg: np.ndarray = savgol_filter(values, filter_size, 4)
    trace = plotly_go.Scatter(
        x=timeaxis, y=values_houravg, yaxis=yaxis,
        name=f'{log.device_id} {data_name} hour',
        meta=dict(id=f'{log_device_full_id(log)}.{data_name}.hour_avg'),
        legendgroup=log_device_full_id(log), showlegend=False, hoverinfo='skip',
        line=dict(shape='spline', width=7, color=f'hsla({hue:.0f}, 100%, 50%, 0.5)', dash=line_dash),
        opacity=opacity
    )
    return trace, values_houravg, get_percentile_range(values_houravg)


@log_cacheable
def mktrace_peaks_valleys(*, log: TempsDeviceLog, values: np.ndarray, timeaxis: np.ndarray,
                          values_ref: Optional[np.ndarray] = None,
                          data_name: str, yaxis: str = 'y2', hue: float = 210, opacity: float = 1.0) \
        -> Tuple[Tuple[BaseTraceType, ...], Tuple[float, float]]:
    if values_ref is None:
        values_ref = values
    traces: List[BaseTraceType] = []
    peaks: np.ndarray = find_peaks(values, prominence=0.05, distance=15)[0]
    valleys: np.ndarray = find_peaks(-values, prominence=0.05, distance=15)[0]
    ranges: List[np.ndarray] = []
    if peaks.size:
        peaks_times: np.ndarray = timeaxis[peaks]
        peaks_values: np.ndarray = values[peaks]
        peaks_values = np.append(peaks_values, [values_ref[-1] + (peaks_values[-1] - values_ref[peaks[-1]])])
        peaks_times = np.append(peaks_times, [timeaxis[-1]])
        traces.append(plotly_go.Scatter(
            x=peaks_times, y=peaks_values, yaxis=yaxis,
            name=f'{log.device_id} {data_name} upper bound',
            meta=dict(id=f'{log_device_full_id(log)}.{data_name}.err_upper'),
            legendgroup=log_device_full_id(log), showlegend=False, hoverinfo='skip',
            mode='lines', line=dict(width=0, shape='spline'),
            opacity=opacity
        ))
        ranges.append(peaks_values)
    if valleys.size:
        valleys_times: np.ndarray = timeaxis[valleys]
        valleys_values: np.ndarray = values[valleys]
        valleys_values = np.append(valleys_values, [values_ref[-1] - (values_ref[peaks[-1]] - valleys_values[-1])])
        valleys_times = np.append(valleys_times, [timeaxis[-1]])
        traces.append(plotly_go.Scatter(
            x=valleys_times, y=valleys_values, yaxis=yaxis,
            name=f'{log.device_id} {data_name} lower bound',
            meta=dict(id=f'{log_device_full_id(log)}.{data_name}.err_lower'),
            legendgroup=log_device_full_id(log), showlegend=False, hoverinfo='skip',
            mode='lines', line=dict(width=0, shape='spline'),
            fillcolor=f'hsla({hue:.0f}, 75%, 50%, 0.33)', fill='tonexty',
            opacity=opacity
        ))
        ranges.append(valleys_values)
    return (tuple(traces) if traces else None), (get_percentile_range(np.concatenate(ranges)) if ranges else (None, None))


@log_cacheable
def mktrace_temps_set(*, log: ThermDeviceLog, timeaxis: np.ndarray, values: Optional[np.ndarray] = None,
                      name_postfix: Optional[str] = None,
                      hue: float = 210, showlegend: bool = True, line_dash: str = 'solid', opacity: float = 1.0) \
        -> Tuple[BaseTraceType, np.ndarray, np.ndarray, Tuple[float, float]]:
    if values is None:
        temps_set, temps_set_mask = make_np_array(log=log, field_name='temp_set')
    else:
        temps_set = values
        temps_set_mask = np.arange(temps_set.size)
    timeaxis_masked: np.ndarray
    temps_set_masked: np.ndarray
    timeaxis_masked, temps_set_masked = make_masked_arrays(timeaxis, temps_set, mask=temps_set_mask, log=log)
    # TODO: Simplify trace by removing contiguous values and use straight instead of splines (might be already)
    if temps_set_mask.size:
        trace = plotly_go.Scatter(
            x=timeaxis_masked, y=temps_set_masked, yaxis='y2',
            name=f'{log.device_id} target' + (' ' + name_postfix if name_postfix else ''),
            meta=dict(id=f'{log_device_full_id(log)}.temp_set{"." + name_postfix if name_postfix else ""}'),
            legendgroup=log_device_full_id(log), showlegend=showlegend,
            mode='lines', line=dict(color=f'hsla({hue:.0f}, 75%, 50%, {0.5 * opacity})', dash=line_dash),
            fillcolor=f'hsla({hue:.0f}, 75%, 50%, {0.1 * opacity})', fill='tozeroy',
            opacity=opacity
        )
    else:
        trace = None
    return trace, temps_set_masked, timeaxis_masked, get_percentile_range(temps_set_masked)


@log_cacheable
def mktrace_bat_voltage(*, log: DeviceLog, timeaxis: np.ndarray, hue: float = 210, line_dash: str = 'solid',
                        opacity: float = 1.0) \
        -> Tuple[BaseTraceType, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float]]:
    bat_voltage, bat_voltage_mask = make_np_array(log=log, field_name='battery_voltage')
    timeaxis_masked: np.ndarray
    bat_voltage_masked: np.ndarray
    timeaxis_masked, bat_voltage_masked = make_masked_arrays(timeaxis, bat_voltage, mask=bat_voltage_mask, log=log)
    bat_voltage_smooth: np.ndarray
    if bat_voltage_mask.size:
        bat_voltage_smooth = bat_voltage_masked
        bat_voltage_smooth = savgol_filter(bat_voltage_smooth, 9, 3, mode='mirror')
        bat_voltage_smooth = median_filter(bat_voltage_smooth, 19, mode='mirror')
        bat_voltage_smooth = gaussian_filter1d(bat_voltage_smooth, 10, mode='mirror')
        trace = plotly_go.Scatter(
            x=timeaxis_masked, y=bat_voltage_smooth, yaxis='y4',
            name=f'{log.device_id} vcc',
            meta=dict(id=f'{log_device_full_id(log)}.battery'),
            legendgroup=log_device_full_id(log),
            line=dict(color=f'hsla({hue:.0f}, 75%, 50%, 0.67)', dash=line_dash),
            fillcolor=f'hsla({hue:.0f}, 75%, 50%, 0.1)', fill='tozeroy',
            opacity=opacity
        )
    else:
        bat_voltage_smooth = np.array([], dtype='float')
        trace = None
    return trace, bat_voltage_masked, timeaxis_masked, bat_voltage_smooth, get_percentile_range(bat_voltage_masked)


@log_cacheable
def mktrace_switch_times(*, log: ThermDeviceLog, timeaxis: np.ndarray, hue: float = 210, opacity: float = 1.0) \
        -> Optional[BaseTraceType]:
    switch_np: np.ndarray = np.array([int(record.therm_state) for record in log])
    if not switch_np.size:
        return None
    switch_starts: List[int] = (np.where(switch_np[:-1] < switch_np[1:])[0] + 1).tolist()
    switch_ends: List[int] = (np.where(switch_np[:-1] > switch_np[1:])[0] + 1).tolist()
    if not switch_starts and not switch_ends:
        return None
    if not switch_starts or (switch_ends and switch_starts[0] > switch_ends[0]):
        switch_starts = [0] + switch_starts
    if not switch_ends or (switch_starts and switch_starts[-1] > switch_ends[-1]):
        switch_ends += [switch_np.size - 1]
    switch_segments: List[Tuple[int, int]] = list(zip(switch_starts, switch_ends))
    segments_ms: np.ndarray = np.array([s.total_seconds() * 1000 for s
                                        in timeaxis[switch_ends] - timeaxis[switch_starts]])
    segments_text: List[str] = [f'{log.device_id}: ON for '
                                f'{format_time((timeaxis[s[1]] - timeaxis[s[0]]).total_seconds(), with_seconds=False)}<br>'
                                f'(from {timeaxis[s[0]].strftime("%H:%M")} to {timeaxis[s[1]].strftime("%H:%M")})'
                                for s in switch_segments]
    trace = plotly_go.Bar(
        name=f'{log.device_id} switch',
        meta=dict(id=f'{log_device_full_id(log)}.switch'),
        legendgroup=log_device_full_id(log) + '.switch',
        x=timeaxis[switch_starts],
        width=segments_ms,
        offset=0,
        y=np.ones(len(switch_starts)),
        text=segments_text,
        textposition='inside',
        textfont=dict(color='rgba(255, 255, 255, 0.67)'),
        hoverinfo='skip',
        yaxis='y',
        marker_line=dict(width=0),
        marker_color=f'hsla({hue:.0f}, 75%, 50%, 0.25)',
        opacity=opacity
    )
    return trace


@log_cacheable
def mktrace_missing_data(*, log: DeviceLog, timeaxis: np.ndarray, interval_threshold: float = 2.0,
                         hue: float = 210, opacity: float = 1.0) \
        -> BaseTraceType:
    timestamps_intervals: np.ndarray = np.array([td.total_seconds() for td in (timeaxis[1:] - timeaxis[:-1])])
    missing_data_starts: np.ndarray = np.where(timestamps_intervals > interval_threshold)[0]
    texts: List[str] = [f'{log.device_id}: no data for {format_time(timestamps_intervals[s], with_seconds=False)}<br>'
                        f'(from {timeaxis[s].strftime("%H:%M")} to {timeaxis[s+1].strftime("%H:%M")})'
                        for s in missing_data_starts]
    trace = plotly_go.Bar(
        name=f'{log.device_id} missing',
        meta=dict(id=f'{log_device_full_id(log)}.missing'),
        legendgroup=log_device_full_id(log),
        showlegend=False,
        x=timeaxis[missing_data_starts],
        width=timestamps_intervals[missing_data_starts] * 1000,
        offset=0,
        y=-np.ones(missing_data_starts.size),
        base=np.ones(missing_data_starts.size),
        text=texts,
        textposition='inside',
        textfont=dict(color='rgba(255, 255, 255, 0.4)'),
        hoverinfo='skip',
        yaxis='y',
        marker_line=dict(width=0),
        marker_color=f'hsla({hue:.0f}, 10%, 50%, 0.1)',
        opacity=opacity
    )
    return trace


def add_timed_annotation(fig: plotly_go.Figure, value: float, timestamp: datetime, yaxis: str, row: int, col: int,
                         value_fmt: str = '.2f', postfix: str = '',
                         hue: float = 210, opacity: float = 1.0, **kwargs) -> None:
    color: str = f'hsla({hue:.0f}, 100%, 67%, {0.5 * opacity:.2f})'
    border_color: str = f'hsla({hue:.0f}, 100%, 10%, {0.75 * opacity:.2f})'
    font_color: str = f'hsla({hue:.0f}, 67%, 90%, {0.9 * opacity:.2f})'
    font_size: float = 10
    text: str = f'{value:{value_fmt}}{postfix}'
    fig.add_annotation(
        x=timestamp, y=value,
        yref=yaxis,
        text=text, font=dict(size=font_size, color=font_color),
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
        ax=11, ay=0,
        xanchor='left', yanchor='middle', standoff=1,
        arrowcolor=color, bgcolor=color, bordercolor=border_color, opacity=opacity,
        row=row, col=col,
        **kwargs
    )


def create_plot_figure(devices: List[Device], plot_mode: Optional[str] = None,
                       log_getter: Optional[Callable[[Device], DeviceLog]] = None) \
        -> Tuple[plotly_go.Figure, Optional[datetime], Optional[datetime]]:
    
    plot_figures_cache_cleanup()

    fig: plotly_go.Figure = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[220, 260, 140],
        specs=[[dict(secondary_y=True)],
               [dict(t=0.18)],
               [dict()]],
        vertical_spacing=0.04
    )

    xranges: List[Tuple[datetime, datetime]] = []
    y2ranges: List[Tuple[float, float]] = []
    y3ranges: List[Tuple[float, float]] = []
    y4ranges: List[Tuple[float, float]] = []

    def append_xrange(a: np.ndarray, minmax: bool = False):
        nonlocal xranges
        if not a.size:
            return
        bounds: Tuple[Any, Any]
        if minmax:
            bounds = (a.min(), a.max())
        else:
            bounds = (a[0], a[-1])
        xranges.append(bounds)

    plot_until: datetime = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=3)

    hue: float = 210.0
    for device in devices:
        log: DeviceLog = log_getter(device) if log_getter is not None else device.log_daily
        if not log:
            continue
        device_config: DeviceConfig = device.config
        device_update_interval = device_config.get('update_interval')
        
        timeaxis = make_timeaxis(log=log, non_roundable_interval=device_update_interval * 1.5)
    
        if isinstance(log, (ThermDeviceLog, TempStationDeviceLog)):
            temps_avg_trace, temps_avg, temps_avg_times, temps_avg_range = mktrace_temps_avg(log=log, timeaxis=timeaxis,
                                                                                             hue=hue)
            if temps_avg_trace is not None:
                fig.add_trace(temps_avg_trace, row=1, col=1, secondary_y=True)
                y2ranges.append(temps_avg_range)
                append_xrange(temps_avg_times)
                if plot_mode != 'prepend':
                    add_timed_annotation(fig, value=temps_avg[-1], timestamp=temps_avg_times[-1],
                                         postfix='°C', yaxis='y2', hue=hue, row=1, col=1, secondary_y=True)

                temps_houravg_trace, *_ = mktrace_houravg(log=log, values=temps_avg,
                                                          timeaxis=temps_avg_times,
                                                          data_name='temp', yaxis='y2', hue=hue)
                fig.add_trace(temps_houravg_trace, row=1, col=1, secondary_y=True)

            temps_cur, temps_cur_mask = make_np_array(log=log, field_name='temp_current')
            if temps_cur_mask.size:
                temps_cur_times: np.ndarray
                temps_cur_masked: np.ndarray
                temps_cur_times, temps_cur_masked = make_masked_arrays(timeaxis, temps_cur, mask=temps_cur_mask, log=log)
                error_traces, raw_range = mktrace_peaks_valleys(log=log, values=temps_cur_masked,
                                                                timeaxis=temps_cur_times,
                                                                values_ref=temps_avg,
                                                                data_name='temp', yaxis='y2', hue=hue)
                if error_traces:
                    for t in error_traces:
                        fig.add_trace(t, row=1, col=1, secondary_y=True)
                    y2ranges.append(raw_range)
                    append_xrange(temps_cur_times)

            set_hue: float = math.fmod(hue - 150, 360)
            temps_set_trace, temps_set, temps_set_times, temps_set_range = mktrace_temps_set(log=log, timeaxis=timeaxis,
                                                                                             hue=set_hue)
            if temps_set_trace is not None:
                fig.add_trace(temps_set_trace, row=1, col=1, secondary_y=True)
                y2ranges.append(temps_set_range)
                append_xrange(temps_set_times)

                if plot_mode != 'prepend':
                    # TODO: Differentiate predicted traces, as to be able to delete them through js
                    sched_temps: List[float] = log.get_sched_temps(device_config)
                    last_timeaxis_time: datetime = timeaxis[-1]
                    next_timeaxis_hour: datetime = (last_timeaxis_time.replace(minute=0, second=0, microsecond=0)
                                                    + timedelta(hours=1))
                    last_hour_minutes: int = int((next_timeaxis_hour - last_timeaxis_time).total_seconds()) // 60
                    predicted_until: datetime = plot_until
                    predicted_hours: int = int((predicted_until - next_timeaxis_hour).total_seconds()) // 3600
                    predicted_temps: np.ndarray = np.repeat(sched_temps[last_timeaxis_time.hour], last_hour_minutes + 1)
                    predicted_temps = np.concatenate((
                        predicted_temps,
                        np.repeat([sched_temps[(next_timeaxis_hour + timedelta(hours=hr)).hour]
                                   for hr in range(predicted_hours)], 60)
                    ))
                    predicted_timeaxis: np.ndarray = np.array([last_timeaxis_time]
                                                              + [next_timeaxis_hour + timedelta(minutes=m)
                                                                 for m in range(-last_hour_minutes, predicted_hours * 60)])
                    temps_set_predicted_trace, *_, temps_set_predicted_range = \
                        mktrace_temps_set(log=log, values=predicted_temps, timeaxis=predicted_timeaxis,
                                          name_postfix='predicted',
                                          hue=set_hue, showlegend=False, line_dash='dot', opacity=0.5)
                    fig.add_trace(temps_set_predicted_trace, row=1, col=1, secondary_y=True)
                    y2ranges.append(temps_set_predicted_range)

        if isinstance(log, TempStationDeviceLog):
            rh_hue: float = math.fmod(hue - 30, 360)

            rhs_avg_trace, rhs_avg, rhs_avg_times, rhs_avg_range = mktrace_rhs_avg(log=log, timeaxis=timeaxis,
                                                                                   hue=rh_hue, opacity=0.4)
            if rhs_avg_trace is not None:
                fig.add_trace(rhs_avg_trace, row=2, col=1)
                y3ranges.append(rhs_avg_range)
                append_xrange(rhs_avg_times)
                if plot_mode != 'prepend':
                    add_timed_annotation(fig, value=rhs_avg[-1], timestamp=rhs_avg_times[-1],
                                         postfix='%', yaxis='y3', hue=rh_hue, row=2, col=1)

                rhs_houravg_trace, *_ = mktrace_houravg(log=log, values=rhs_avg,
                                                        timeaxis=rhs_avg_times,
                                                        data_name='rh', yaxis='y3',
                                                        hue=rh_hue, opacity=0.4)
                fig.add_trace(rhs_houravg_trace, row=2, col=1)

            rhs_cur, rhs_cur_mask = make_np_array(log=log, field_name='humidity_current')
            if rhs_cur_mask.size:
                rhs_cur_times: np.ndarray
                rhs_cur_masked: np.ndarray
                rhs_cur_times, rhs_cur_masked = make_masked_arrays(timeaxis, rhs_cur, mask=rhs_cur_mask, log=log)
                error_traces, raw_range = mktrace_peaks_valleys(log=log, values=rhs_cur_masked,
                                                                timeaxis=rhs_cur_times,
                                                                values_ref=rhs_avg, data_name='rh', yaxis='y3',
                                                                hue=rh_hue, opacity=0.2)
                if error_traces:
                    for t in error_traces:
                        fig.add_trace(t, row=2, col=1)
                    y3ranges.append(raw_range)
                    append_xrange(rhs_cur_times)

        if isinstance(log, ThermDeviceLog):
            switch_hue: float = math.fmod(hue - 190, 360)
            switch_times_trace = mktrace_switch_times(log=log, timeaxis=timeaxis, hue=switch_hue)
            if switch_times_trace is not None:
                fig.add_trace(switch_times_trace, row=1, col=1)

        bat_hue: float = math.fmod(hue - 15, 360)
        bat_voltage_trace, bat_voltage, bat_voltage_times, bat_voltage_smooth, bat_voltage_range = \
            mktrace_bat_voltage(log=log, timeaxis=timeaxis, hue=bat_hue)
        if bat_voltage_trace is not None:
            fig.add_trace(bat_voltage_trace, row=3, col=1)
            y4ranges.append(bat_voltage_range)
            append_xrange(bat_voltage_times)
            if plot_mode != 'prepend':
                add_timed_annotation(fig, value=bat_voltage_smooth[-1], timestamp=bat_voltage_times[-1],
                                     postfix='V', yaxis='y4', hue=bat_hue, row=3, col=1)

            error_traces, raw_range = mktrace_peaks_valleys(log=log, values=bat_voltage,
                                                            timeaxis=bat_voltage_times,
                                                            values_ref=bat_voltage_smooth,
                                                            data_name='bat', yaxis='y4', hue=bat_hue)
            if error_traces:
                for t in error_traces:
                    fig.add_trace(t, row=3, col=1)
                y4ranges.append(raw_range)
    
        missing_data_trace = mktrace_missing_data(log=log, timeaxis=timeaxis, 
                                                  interval_threshold=(device_update_interval * 5.0), hue=hue)
        fig.add_trace(missing_data_trace, row=1, col=1)

        hue = math.fmod(hue + 100.0, 360.0)

    if plot_mode != 'prepend':
        fig.add_vline(x=datetime.now().timestamp() * 1000,
                      annotation_text=datetime.now().strftime("%d/%m/%Y %H:%M"),
                      annotation_textangle=0,
                      annotation_position='bottom left',
                      annotation_font=dict(color='rgba(255, 191, 255, 0.8)', size=9),
                      annotation_yshift=-3,
                      line_dash='dot',
                      line_color='magenta',
                      layer='below')

    RangedType = Union[float, datetime]
    RangeSpanType = Union[float, timedelta]

    def compute_axis_range(minmaxrange: List[Tuple[RangedType, RangedType]], pad: Optional[float] = 0.1,
                           limits: Optional[Tuple[Optional[RangedType], Optional[RangedType]]] = None) \
            -> Optional[Tuple[RangedType, RangedType]]:
        if not minmaxrange:
            return None
        range_mins: List[RangedType]
        range_maxs: List[RangedType]
        range_min: RangedType
        range_max: RangedType
        range_mins, range_maxs = zip(*minmaxrange)
        range_min, range_max = min(range_mins), max(range_maxs)
        range_span: RangeSpanType = range_max - range_min
        if pad:
            range_pad: RangeSpanType = range_span * pad
            range_min = range_min - range_pad
            range_max = range_max + range_pad
        if limits is not None:
            limit_min: Optional[RangedType]
            limit_max: Optional[RangedType]
            limit_min, limit_max = limits
            if limit_min is not None:
                range_min = max(limit_min, range_min)
            if limit_max is not None:
                range_max = min(limit_max, range_max)
        return range_min, range_max

    y2range: Optional[Tuple[RangedType]] = compute_axis_range(y2ranges, limits=(0.0, None))
    y3range: Optional[Tuple[RangedType]] = compute_axis_range(y3ranges, limits=(0.0, None))
    y4range: Optional[Tuple[RangedType]] = compute_axis_range(y4ranges, limits=(0.0, None))
    fig.update_layout(
        template='plotly_dark',
        height=560,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            range=[datetime.now() - timedelta(hours=12), plot_until],
            rangeslider=dict(visible=True, thickness=0.1),
            rangeselector=dict(
                visible=True,
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=2, label="2d", step="day", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                ]),
                bgcolor='rgba(127, 127, 127, 0.33)',
                activecolor='rgba(31, 63, 127, 0.66)',
                x=1.0,
                xanchor='right',
                y=1.05,
                yanchor='bottom'
            ),
            type='date'
        ),
        yaxis=dict(  # events
            visible=False,
            range=[0, 1],
            fixedrange=True
        ),
        yaxis2=dict(  # temps
            title_text='temperature [°C]',
            hoverformat='.2f',
            dtick=1,
            side='right',
            rangemode="nonnegative",
            fixedrange=True,
            range=list(y2range) if y2range else None,
        ),
        yaxis3=dict(  # rh
            title_text='humidity [%]',
            hoverformat='.2f',
            # dtick=10,
            side='right',
            anchor='y2',
            # position=1,
            rangemode="nonnegative",
            fixedrange=True,
            range=list(y3range) if y3range else None,
            # gridcolor='hsla(300, 40%, 40%, 0.3)',
            # tickfont=dict(color='hsla(300, 25%, 75%, 0.5)')
        ),
        yaxis4=dict(  # bat voltage
            title_text='battery [V]',
            hoverformat='.3f',
            side='right',
            # anchor='y3',
            # position=1,
            rangemode="nonnegative",
            fixedrange=True,
            range=list(y4range) if y4range else None,
            # gridcolor='hsla(300, 40%, 40%, 0.3)',
            # tickfont=dict(color='hsla(300, 25%, 75%, 0.5)')
        ),
        legend=dict(
            orientation='h',
            x=1.0,
            xanchor='right',
            y=1.12,
            yanchor='bottom',
            traceorder='normal',
        ),
        hoverlabel=dict(namelength=-1),
        hovermode='x unified',
        hoverdistance=1,
        dragmode='pan',
    )
    fig.update_xaxes(showgrid=True, dtick=3600000, row=1, col=1)
    fig.update_xaxes(showgrid=True, dtick=3600000, row=2, col=1)
    fig.update_xaxes(showgrid=True, dtick=3600000, row=3, col=1)
    xaxis_min: Optional[datetime] = None
    xaxis_max: Optional[datetime] = None
    if xranges:
        xaxis_min, xaxis_max = compute_axis_range(xranges, pad=0)
    return fig, xaxis_min, xaxis_max


def create_plot(plot_mode: Optional[str] = None, **log_filters):
    valid_filters: MutableMapping[str, Any] = {k: v for k, v in log_filters.items() if v is not None}

    def log_getter(d: Device):
        nonlocal valid_filters
        return d.log_daily if not valid_filters else d.get_log(**valid_filters)

    devices: List[Device] = list(Device.iterate_known_devices())
    fig: plotly_go.Figure
    data_dt_min: Optional[datetime]
    data_dt_max: Optional[datetime]
    fig, data_dt_min, data_dt_max = create_plot_figure(devices, plot_mode=plot_mode, log_getter=log_getter)
    # TODO: Replace more constants with config-specified values
    next_update_times: List[datetime]
    if not valid_filters:
        last_logs = [d.log_daily for d in devices]
        next_update_times = [ensure_tz(l[-1].timestamp).astimezone(tz=None).replace(tzinfo=None)
                             + timedelta(seconds=(l.average_sync_interval or (300 * (64 / 60) + 15)) + 10)
                             for l in last_logs if l]
        next_update_times = [t for t in next_update_times if t > datetime.now()]
    else:
        next_update_times = []
    expected_update_interval: Optional[float]
    if next_update_times:
        expected_update_time: datetime = min(next_update_times)
        expected_update_time = expected_update_time.astimezone(tz=None).replace(tzinfo=None)
        expected_update_interval = max(0.0, (expected_update_time - datetime.now()).total_seconds()) or None
    else:
        expected_update_interval = None
    data_ts_min: Optional[float] = data_dt_min.timestamp() * 1000 if data_dt_min else None
    data_ts_max: Optional[float] = data_dt_max.timestamp() * 1000 if data_dt_max else None
    data = {
        'plots': fig,
        'timestamps_range': [data_ts_min, data_ts_max],
        'status': 'SUCCESS'
    }
    if expected_update_interval:
        data['next_update_in'] = expected_update_interval
    json_data = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return json_data


@app.route('/plot')
@compress.compressed()
def plot_ajax():
    ts_from: Optional[str] = request.args.get('from')
    ts_to: Optional[str] = request.args.get('to')
    ts_exclusive: Optional[str] = request.args.get('exclusive')
    plot_mode: Optional[str] = request.args.get('mode')
    timestamp_min: Optional[datetime] = dateutil_parser(ts_from) if ts_from else None
    timestamp_max: Optional[datetime] = dateutil_parser(ts_to) if ts_to else None
    bound_exclusive: Optional[bool] = ts_exclusive.lower() in ('1', 'true', 'yes') if ts_exclusive else None
    json_data = create_plot(timestamp_min=timestamp_min, timestamp_max=timestamp_max, bound_exclusive=bound_exclusive,
                            plot_mode=plot_mode)
    return Response(json_data, mimetype=app.config["JSONIFY_MIMETYPE"])


@app.route('/')
@compress.compressed()
def homepage():
    if not check_session_valid():
        return redirect(url_for('auth_page'))
    return render_template('status.jinja2')


@app.route('/favicon.ico')
@compress.compressed()
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico',
                               mimetype='image/vnd.microsoft.icon')
