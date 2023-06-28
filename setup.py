#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: skip-file

from setuptools import find_packages, setup

packages = find_packages(".", exclude=["tests"])

setup(
    name='fcest',
    version='0.0.1',
    author='Onno P. Kampman',
    author_email='onno.kampman@gmail.com',
    description='Methods for estimation of functional connectivity',
    license='Apache License 2.0',
    url='https://github.com/OnnoKampman/FCEst',
    packages=packages,
    install_requires=[
        'gpflow==2.5.2',
        'numpy',
        'pandas',
        'rpy2==3.4.5',
        'scipy',
        'tensorflow',
    ],
    python_requires='>=3.10',
    zip_safe=False
)
