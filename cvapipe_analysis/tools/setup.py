# to compile run:
# python setup.py build_ext --inplace

import numpy as np
import distutils as du
from Cython.Build import cythonize

ext_modules = [
    du.extension.Extension(
        "bincorr",
        ["bincorrlib.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

du.core.setup(
    name='bincorr',
    ext_modules=cythonize(ext_modules)
)