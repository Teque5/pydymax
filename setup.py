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
    url='http://teque5.com',
    packages=['dymax'],
    description='Dymaxion Fuller Projection Utilities',
    long_description=__doc__,
    license='Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)',
    install_requires=['numpy', 'matplotlib', 'Pillow', 'numba'],
    package_data={'dymax': ['data/*.dat', 'data/*.jpg']}
    )
