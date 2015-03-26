#!/usr/bin/env python

from distutils.core import setup
from dymax import __version__

setup(name='dymax.py',
      version=__version__,
      description='dymax.py: Lat/Lon to Dymaxion Fuller Projection',
      author='Teque5',
      author_email='',
      maintainer='Teque5',
      maintainer_email='',
      url=' http://teque5.com',
      packages=['dymax'],
      long_description='Geodetic Conversion from Lat/Lon to Dymaxion Fuller Projection',
      license='Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)',
      platforms=['any'],
     )