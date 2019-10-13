from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

# linux server python3
ext_modules = [Extension("pyclasso", sources=["pyclasso.pyx", "classo.c"], include_dirs=[np.get_include()], 
    libraries=["openblas"])]
# macbook and linux server python2
#ext_modules = [Extension("pyclasso", sources=["pyclasso.pyx", "classo.c"], include_dirs=[np.get_include()])]

setup(
    name = "FOD",
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules
)