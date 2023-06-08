#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='fcest',
    version='0.0.1',
    description='Methods for estimation of functional connectivity',
    url='http://github.com/OnnoKampman/FCEst',
    author='Onno P. Kampman',
    author_email='onno.kampman@gmail.com',
    license='Apache License 2.0',
    packages=['fcest'],
    install_requires=[
        'gpflow==2.5.2',
        'numpy',
        'pandas',
        'scipy',
        'tensorflow',
    ],
    python_requires='>=3.8',
    zip_safe=False
)
