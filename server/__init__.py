"""ESPTherm server package"""

from flask import Flask
from flask_compress import Compress

from .config import ServerConfig


__all__ = ["app", "compress"]


app: Flask = Flask(__name__)
app.config["COMPRESS_REGISTER"] = False  # disable default compression of all eligible requests
compress: Compress = Compress()
compress.init_app(app)
