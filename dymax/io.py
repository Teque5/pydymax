#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''Routines to load shoreline data'''
import os
import numpy as np
import pkg_resources
import time

from . import convert

PKG_DATA = pkg_resources.resource_filename('dymax', 'data') + os.path.sep

def get_islands(resolution='c', verbose=True):
    '''
    Get coastlines from NOAA's GSHHS
    Global Self-consistent Hierarchical High-resolution Shorelines

    Parameters
    ----------
    resolution : string
        Resolutions are valid in the following set:
        c (crude), l (low), i (intermediate), h (high), f (full)
    verbose : bool

    Returns
    -------
    lonlat_islands : list of 2-D ndarrays
        Each island in list contains an array of N points. Each point is a
        (lon, lat) pair of WGS84 coordinates.
    dymax_islands : ndarray
        Each island in list contains an array of N points. Each point is a
        (x_pos, y_pos) pair of dymax coordinates.
    '''
    ### Load Coastlines
    binfile = open(PKG_DATA+'gshhs_'+resolution+'.dat', 'rb')
    data = np.fromfile(binfile, '<f4')
    data = data.reshape(len(data)//2, 2)

    start = time.time()
    dymaxdata = np.zeros_like(data)
    for idx, row in enumerate(data):
        dymaxdata[idx] = convert.lonlat2dymax(row[0], row[1])
    if verbose: print(':: mapped {:d} points to dymax projection @ {:.1f} pts/sec [{:.1f} secs total]'.format(len(dymaxdata), len(dymaxdata)/(time.time()-start), time.time()-start))

    ### Load Metadata
    with open(PKG_DATA+'gshhsmeta_'+resolution+'.dat', 'r') as derp:
        places = derp.read()
        places = places.split('\n')

    lonlat_islands = []
    dymax_islands = []
    #1, area, numpoints, limit_south, limit_north, startbyte, numbytes, id-(E/W crosses dateline east or west)
    for place in places:
        if len(place) < 2: continue
        column = place.split()
        #print(column)
        if float(column[1]) < 500 and resolution == 'c': continue # eliminate tiny area islands
        start_idx = int(column[5])//8
        stop_idx = start_idx + int(column[6])//8
        lonlat_islands += [data[start_idx:stop_idx]]
        dymax_islands += [dymaxdata[start_idx:stop_idx]]
    if verbose: print(':: computed', len(places), 'coastlines')
    return lonlat_islands, dymax_islands
