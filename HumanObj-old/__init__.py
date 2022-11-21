import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from . import _init_paths_

import sys
print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)
