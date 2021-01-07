import logging
from argparse import ArgumentParser, Namespace

from . import app
from . import sync  # Import sync routes
from . import ui  # Import ui routes
from .devices import Device


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address to bind to')
    parser.add_argument('--port', type=int, default=8367, help='Host port to bind to')
    args: Namespace = parser.parse_args()

    Device.load_known_devices()
    app.logger.setLevel(logging.INFO)
    app.run(host=args.host, port=args.port)
