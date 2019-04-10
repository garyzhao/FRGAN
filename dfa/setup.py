from __future__ import absolute_import

import numpy as np
from distutils.core import setup, Extension


# define the extension module
libc = Extension('z_buffer', sources=['z_buffer.c'], include_dirs=[np.get_include()])

# run the setup
setup(ext_modules=[libc])
