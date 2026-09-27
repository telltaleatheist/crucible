import sys

from ..platform import runner

sys.modules[__name__] = runner
