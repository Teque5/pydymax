#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''Routines to load shoreline data'''
import h5py
import lzma
import os
import numpy as np
import pkg_resources
import struct
import time

from . import convert

PKG_DATA = pkg_resources.resource_filename('dymax', 'data') + os.path.sep

def load_gshhs_xz(filename):
    '''
    Convert GSHHS Binary File to Numpy Arrays

    Written for GSHHS v2.3.7 utilizing API v2.0

    Files were created by using:
    for res in ['c', 'l', 'i', 'h', 'f']:
        with open('gshhs_{}.b'.format(res), 'rb') as raw:
            with open('gshhs_{}.xz'.format(res), 'wb') as out:
                out.write(lzma.compress(raw.read()))

    Returns
    -------
    headers : ndarray of int32
        cdx : int
            coastline starting index
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
            Id of container polygon that encloses this polygon (-1 if none)
        ancestor : int
            Id of ancestor polygon in the full resolution set that was the source of this polygon (-1 if none)
    wvs : list of ndarray of int32
        WGS84 (lon, lat) vertices in degrees.
    '''
    cdx = 0 # polygon ID
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
                header = np.array((cdx,) + header[1:3] + header[8:], dtype=np.int32)
                coasts += [coast]
                headers += [header]
                # increment coast line starting index
                cdx += len(coast)
                # print(cdx,end=' ')
            except struct.error:
                print('{} EOF'.format(filename))
                break
    headers = np.vstack(headers)
    coasts = np.vstack(coasts)
    # convert microdegrees to degrees
    coasts = coasts.astype(np.float32) / 1e6
    return headers, coasts

def gshhg2hdf5(gshhg_folder, filename='gshhg'):
    '''
    Convert GSHHG binaries to single HFD5 file.

    Crude Resolution (25 km) 'c'
    Low Resolution (5 km) 'l'
    Intermediate Resolution (1 km) 'i'
    High Resolution (0.2 km) 'h'
    Full Resolution (0.04 km) 'f'

    gshss.h5
      [resolution]/
        headers/
        coasts/[number]

    In HDF5 parlance, datasets and sub-objects of groups.
    '''
    with lzma.open('{}.h5.xz'.format(filename), 'wb') as lz_handle:
        with h5py.File(lz_handle, 'w', libver='latest') as h5_handle:
            for resolution in ['c','l','i','h','f']:
                source_file = os.path.join(gshhg_folder, 'gshhs_{}.b'.format(resolution))
                headers, coasts = load_gshhg_binary(source_file)
                group = h5_handle.create_group(resolution)
                group.create_dataset('headers', data=headers)#, compression='gzip', compression_opts=9)
                group.create_dataset('coasts', data=coasts)#, compression='gzip', compression_opts=9)

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
