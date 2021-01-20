"""Devices definitions"""

from . import tempsdevices, therm, tempstation

from .tempsdevices import *
from .therm import *
from .tempstation import *


__all__ = tempsdevices.__all__ + therm.__all__ + tempstation.__all__

# TODO: Dynamic import of device modules?
