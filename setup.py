from distutils.extension import Extension

import numpy
from Cython.Build import cythonize
from setuptools import find_packages, setup

extensions = [
    Extension(
        'sklearn_quantile.utils.weighted_quantile',
        ['sklearn_quantile/utils/weighted_quantile.pyx'],
    ),
    Extension('sklearn_quantile.ensemble.quantile', ['sklearn_quantile/ensemble/quantile.pyx']),
    Extension('sklearn_quantile.ensemble.maximum', ['sklearn_quantile/ensemble/maximum.pyx']),
]

setup(
    packages=find_packages(),
    ext_modules=cythonize(extensions, annotate=True),
    include_dirs=[numpy.get_include()],
)
