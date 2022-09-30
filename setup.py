#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''
Geodetic Conversion from Lat/Lon to Dymaxion Fuller Projection

Where to go from here?
* `pip install --editable .`
* `python3 -m dymax.examples`
* `python3 setup.py test`
* `pytest --doctest-modules`
* `pylint dymax`
'''
from setuptools import setup
import os, re

with open(os.path.join('dymax', '__init__.py')) as derp:
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', derp.read()).group(1)

setup(
    name='dymax',
    version=version,
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
