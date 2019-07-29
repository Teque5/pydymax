#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''
Geodetic Conversion from Lat/Lon to Dymaxion Fuller Projection

Where to go from here?
* `python3 setup.py develop`
* `python3 setup.py install`
* `python3 -m dymax.examples`
* `python3 setup.py test`
* `pytest --doctest-modules`
* `pylint dymax`
'''
from setuptools import setup
from dymax import __version__

setup(
    name='dymax',
    version=__version__,
    author='Teque5',
    maintainer='Teque5',
    url='https://github.com/Teque5/pydymax',
    packages=['dymax'],
    description='Dymaxion Fuller Projection Utilities',
    long_description=__doc__,
    license='Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)',
    install_requires=['numpy', 'matplotlib', 'Pillow', 'numba'],
    package_data={'dymax': ['data/*.xz', 'data/*.jpg']},
    test_suite='tests',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
)
