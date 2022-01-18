from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

extensions = [
    Extension("sklearn_quantile.utils.weighted_quantile",
              ["sklearn_quantile/utils/weighted_quantile.pyx"]),
    Extension("sklearn_quantile.ensemble.qrf",
              ["sklearn_quantile/ensemble/qrf.pyx"]),
    Extension("sklearn_quantile.ensemble.max_rf",
              ["sklearn_quantile/ensemble/max_rf.pyx"])
]

setup(
    name='sklearn_quantile',
    version='0.0.1',
    packages=find_packages(),
    url='',
    license='BSD 3 clause',
    author='Jasper Roebroek',
    author_email='roebroek.jasper@gmail.com',
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
    install_requires=['numpy', 'sklearn', 'joblib'],
    extras_require={
        'develop': ['cython', 'sphinx', 'sphinx_rtd_theme', 'numpydoc', 'jupyter', 'matplotlib']
    }
)
