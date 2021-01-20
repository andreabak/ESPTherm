"""Module with server routes for web UI"""

import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import MutableMapping, Optional, Tuple, Any

from dateutil.parser import parse as dateutil_parser
from flask import request, url_for, jsonify, render_template, send_from_directory, redirect, Response

from . import app, compress
from .config import server_config
from .plot import create_plot


SESSION_COOKIE_NAME: str = 'flasksession'
MAIN_DEVICE: Tuple[str, str] = ('therm', 'main')


@dataclass
class UserSession:
    """Dataclass for user sessions"""
    valid: bool = True


authed_sessions: MutableMapping[str, UserSession] = {}


def get_session_cookie() -> Optional[str]:
    """Return the session cookie from the current Flask request, or None"""
    return request.cookies.get(SESSION_COOKIE_NAME)


def check_session_valid() -> bool:  # TODO: Create anon sessions too so we can rate-limit pin attempts
    """
    Checks whether the session cookie of the current Flask request
    and, if it does, verifies if it's valid.
    :return: True if the session is valid else False
    """
    global authed_sessions
    cookie_id: Optional[str] = get_session_cookie()
    if not cookie_id or cookie_id not in authed_sessions:
        return False
    return authed_sessions[cookie_id].valid


def create_session_cookie(response: Optional[Response] = None) -> str:
    """
    Creates an authenticated session cookie.
    :param response: if specifies, a Flask `Response` object to instruct to set the cookie header
    :return: the id string of the session
    """
    global authed_sessions
    session_id: str = uuid.uuid4().hex
    if response is not None:
        response.set_cookie(SESSION_COOKIE_NAME, session_id)
    authed_sessions[session_id] = UserSession()
    return session_id


@app.route('/auth', methods=['GET', 'POST'])
@compress.compressed()
def auth_page() -> Response:
    """Authentication page (pinpad) route"""
    if request.method == 'POST':  # Check auth
        response: MutableMapping[str, Any] = {}
        received_code: Optional[str] = request.form.get('secret_code') if request.form else None
        if received_code and received_code == str(server_config['ui_secret_code']):
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


@app.route('/plot')
@compress.compressed()
def plot_ajax():
    """Ajax route endpoint function for fetching plot data"""
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
    """Homepage route"""
    if not check_session_valid():  # TODO: Make this into a decorator and wrap routes
        return redirect(url_for('auth_page'))
    return render_template('status.jinja2')


@app.route('/favicon.ico')
@compress.compressed()
def favicon():
    """Url route endpoint function for the site favicon"""
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico',
                               mimetype='image/vnd.microsoft.icon')
