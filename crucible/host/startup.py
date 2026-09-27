import sys

from ..platform import startup

sys.modules[__name__] = startup
