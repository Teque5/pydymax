#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''Routines to load shoreline data'''
import lzma
import numpy as np
import os
import pkg_resources
import struct
import time

from . import convert

PKG_DATA = pkg_resources.resource_filename('dymax', 'data') + os.path.sep

def load_gshhs_xz(filename):
    '''
    Convert GSHHS Binary File to Numpy Arrays

    Written for GSHHS v2.3.7 utilizing API v2.0
    For some insane reason in this version they didn't wrap the coastlines
    around the longitude discontinutiy correctly.

    Files were created by using:
    for res in ['c', 'l', 'i', 'h', 'f']:
        with open('gshhs_{}.b'.format(res), 'rb') as raw:
            with open('gshhs_{}.xz'.format(res), 'wb') as out:
                out.write(lzma.compress(raw.read()))

    Returns
    -------
    headers : ndarray of int32
        n : int
            number of points in this coastline
        flag : int
            (5 items) = level + version << 8 + greenwich << 16 + source << 24 + river << 25
            low byte:    level = flag & 255: Values: 1 land, 2 lake, 3 island_in_lake, 4 pond_in_island_in_lake
            2nd byte:    version = (flag >> 8) & 255: Values: Should be 12 for GSHHG release 12 (i.e., version 2.2)
            3rd byte:    greenwich = (flag >> 16) & 1: Values: Greenwich is 1 if Greenwich is crossed
            4th byte:    source = (flag >> 24) & 1: Values: 0 = CIA WDBII, 1 = WVS
            4th byte:    river = (flag >> 25) & 1: Values: 0 = not set, 1 = river-lake and level = 2
        area_full : int
            Area of original full-resolution polygon in 1/10 km^2.
        container: int
            Index of container polygon that encloses this polygon (-1 if none)
        ancestor : int
            Index of ancestor polygon in the full resolution set that was the source of this polygon (-1 if none)
    wvs : list of ndarray of int32
        lon : float
            WGS84 Longitude in degrees from (-180, 180)
        lat : float
            WGS84 Latitude in degrees from (-90, 90)
    '''
    with lzma.open(filename, 'rb') as handle:
        # world vector shoreline
        # lon, lat
        coasts = []
        # headers
        # id, n, flag, west, east, south, north, area, area_full, container, ancestor
        headers = []
        while True:
            try:
                header = struct.unpack(
                    '>11i',
                    handle.read(11*4))
                numbytes = header[1]
                coast = struct.unpack(
                    '>{}i'.format(numbytes*2),
                    handle.read(numbytes*2*4))
                coast = np.array(coast, dtype=np.int32).reshape((numbytes, 2))
                # dump
                header = np.array(header[1:3] + header[8:], dtype=np.int32)
                # convert microdegrees to degrees
                coast = coast.astype(np.float32) / 1e6
                # chang lon from (-180, 360) to (-180, 180)
                lon = np.radians(coast[:, 0])
                lon = np.degrees(np.arctan2(np.sin(lon), np.cos(lon)))
                coast[:, 0] = lon
                coasts += [coast]
                headers += [header]
            except struct.error:
                print('{} EOF'.format(filename))
                break
    headers = np.vstack(headers)
    return headers, coasts

def get_coastlines(resolution='c'):
    '''
    Return Dymax Lands

    Parameters
    ----------
    resolution : string
        Resolutions are valid in the following set:
            Crude Resolution (25 km) 'c'
            Low Resolution (5 km) 'l'
            Intermediate Resolution (1 km) 'i'
            High Resolution (0.2 km) 'h'
            Full Resolution (0.04 km) 'f'
    filter : list
        Filter for flag. See load_gshhs_xz for details.

    Returns
    -------
    coasts : list of 2-D ndarrays
        Each island in list contains an array of N points. Each point is a
        (lon, lat) pair of WGS84 coordinates.
    dymax_coasts : list of list of floats
        Each island in list contains an array of N points. Each point is a
        (x_pos, y_pos) pair of dymax coordinates.
    '''
    dymax_coasts = []
    headers, coasts = load_gshhs_xz(os.path.join(PKG_DATA,'gshhs_{}.xz'.format(resolution)))
    for hdx, header in enumerate(headers):
        dymax_coast = []
        # only consider land, not islands or lakesd
        for cdx in range(header[0]):
            lon, lat = coasts[hdx][cdx]
            dymax_coast += [convert.lonlat2dymax(lon, lat)]
        dymax_coasts += [dymax_coast]
    return coasts, dymax_coasts
