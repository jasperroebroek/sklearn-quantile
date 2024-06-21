from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


extensions = [
    Extension("sklearn_quantile.utils.weighted_quantile",
              ["sklearn_quantile/utils/weighted_quantile.pyx"]),
    Extension("sklearn_quantile.ensemble.quantile",
              ["sklearn_quantile/ensemble/quantile.pyx"]),
    Extension("sklearn_quantile.ensemble.maximum",
              ["sklearn_quantile/ensemble/maximum.pyx"]),
    # Extension("sklearn_quantile.utils.WQ",
    #           ["sklearn_quantile/utils/WQ.pyx"])
]

setup(
    packages=find_packages(),
    ext_modules=cythonize(extensions, annotate=True, language_level="3"),
    include_dirs=[numpy.get_include()]
)
