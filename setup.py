from setuptools import setup

setup(
    name='FCEst',
    version='0.0.1',
    description='Methods for estimation of functional connectivity',
    url='http://github.com/OnnoKampman/FCEst',
    author='Onno P. Kampman',
    author_email='onno.kampman@gmail.com',
    license='Apache 2.0',
    packages=['FCEst'],
    install_requires=[
        'gpflow==2.5.2',
        'matplotlib',
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',
    ],
    zip_safe=False
)
