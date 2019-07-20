#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''Routines to load shoreline data'''
import os
import numpy as np
import pkg_resources
import struct
import time

from . import convert

PKG_DATA = pkg_resources.resource_filename('dymax', 'data') + os.path.sep

def gshhg2df(filename):
    '''
    Convert GSHHG Binary to Pandas DataFrame

    Written for GSHHG v2.3.7 utilizing API v2.0

    Returns
    -------
    headers : ndarray of int32
        int id;         /* Unique polygon id number, starting at 0 */
        int n;          /* Number of points in this polygon */
        int flag;       /* = level + version << 8 + greenwich << 16 + source << 24 + river << 25 */
        /* flag contains 5 items, as follows:
         * low byte:    level = flag & 255: Values: 1 land, 2 lake, 3 island_in_lake, 4 pond_in_island_in_lake
         * 2nd byte:    version = (flag >> 8) & 255: Values: Should be 12 for GSHHG release 12 (i.e., version 2.2)
         * 3rd byte:    greenwich = (flag >> 16) & 1: Values: Greenwich is 1 if Greenwich is crossed
         * 4th byte:    source = (flag >> 24) & 1: Values: 0 = CIA WDBII, 1 = WVS
         * 4th byte:    river = (flag >> 25) & 1: Values: 0 = not set, 1 = river-lake and level = 2
         */
        int west, east, south, north;   /* min/max extent in micro-degrees */
        int area;       /* Area of polygon in 1/10 km^2 */
        int area_full;  /* Area of original full-resolution polygon in 1/10 km^2 */
        int container;  /* Id of container polygon that encloses this polygon (-1 if none) */
        int ancestor;   /* Id of ancestor polygon in the full resolution set that was the source of this polygon (-1 if none) */
    wvs : list of ndarray of int32
        WGS84 (lon, lat) vertices in microdegrees.
    '''
    with open(filename, 'rb') as handle:
        # world vector shoreline
        # lon, lat
        coasts = []
        # headers
        # id, n, flag, west, east, south, north, area, area_full, container, ancestor
        headers = np.empty(shape=(0, 11), dtype=np.int32)
        while True:
            try:
                header = struct.unpack(
                    '>11i',
                    handle.read(11*4))
                coast = struct.unpack(
                    '>{}i'.format(header[1]*2),
                    handle.read(header[1]*2*4))
                coast = np.array(coast, dtype=np.int32).reshape((header[1], 2))
                # dump
                coasts += [coast]
                headers = np.vstack((headers, header))
            except struct.error:
                print('{} EOF'.format(filename))
                break
    return headers, coasts

    # wvs = pd.DataFrame(data={'lon':data[:, 0], 'lat':data[:, 1]})
    #
    # with open(PKG_DATA+'gshhsmeta_'+resolution+'.dat', 'r') as derp:
    #     places = derp.read()
    #     places = places.split('\n')
    #
    # meta = pd.read_csv(PKG_DATA+'gshhsmeta_'+resolution+'.dat')

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
    with open(PKG_DATA+'gshhs_'+resolution+'.dat', 'rb') as binfile:
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
