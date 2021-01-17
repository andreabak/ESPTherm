#!/usr/bin/env python3
"""Main runtime entry point module for ESPTherm server"""

# pylint: disable=unused-import, relative-beyond-top-level

import logging
from argparse import ArgumentParser, Namespace

from . import app
# Import routes
from . import sync
from . import ui
from .devices import Device


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address to bind to')
    parser.add_argument('--port', type=int, default=8367, help='Host port to bind to')
    args: Namespace = parser.parse_args()

    Device.load_known_devices()
    app.logger.setLevel(logging.INFO)  # pylint: disable=no-member
    app.run(host=args.host, port=args.port)
