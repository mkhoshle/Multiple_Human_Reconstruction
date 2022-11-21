import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import sys, os
lib_dir = os.path.dirname(__file__)
root_dir = os.path.join(lib_dir.replace(os.path.basename(lib_dir),''))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from .build import build_model
from . import smpl as smpl_model

