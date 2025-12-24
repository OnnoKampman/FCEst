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
        'gpflow==2.10.0',
        'ipykernel',  # for running Jupyter Notebooks
        'matplotlib',  # for plotting test results
        'numpy<2',
        'pandas==1.5.3',
        'rpy2==3.4.5',
        'scipy',
        'statsmodels',
        'tensorflow==2.15',
        'tensorflow-probability==0.23',
        'tf-keras',
    ],
    python_requires='>=3.11',
    zip_safe=False
)
