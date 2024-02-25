#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: skip-file

from setuptools import find_packages, setup


def read_file(filename: str) -> str:
    with open(filename, encoding="utf-8") as f:
        return f.read().strip()


packages = find_packages(".", exclude=["tests"])
version = read_file("VERSION")

setup(
    name='fcest',
    version=version,
    author='Onno P. Kampman',
    author_email='onno.kampman@gmail.com',
    description='Methods for estimation of functional connectivity',
    license='Apache License 2.0',
    url='https://github.com/OnnoKampman/FCEst',
    packages=packages,
    install_requires=[
        'gpflow==2.6.1',
        'numpy',
        'pandas==1.5.3',
        'rpy2==3.4.5',
        'scipy',
        'statsmodels',
        'tensorflow>=2.10',
    ],
    python_requires='>=3.10',
    zip_safe=False
)
