#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''Dymaxion Projection Conversion Subroutines'''
import math
from functools import lru_cache
import numba
import numpy as np
import time

from . import constants

@numba.njit(fastmath=True)
def magnitude(vec):
    '''N-Dimentional Vector Magnitude'''
    acc = 0
    for val in vec:
        acc += val**2
    return math.sqrt(acc)

@numba.njit(fastmath=True)
def euclidean(vec_a, vec_b):
    '''N-Dimensional Vector L2 Norm; runs in 270ns'''
    acc = 0
    for idx in range(len(vec_a)):
        acc += (vec_a[idx] - vec_b[idx])**2
    return math.sqrt(acc)

### Dymax Conversion Main Routine
@lru_cache(maxsize=2**12)
def lonlat2dymax(lon, lat, getlcd=False):
    '''
    Lon Lat 2 Dymax XY

    This is the primary reason this whole package exists.

    Parameters
    ----------
    lon : float
        Longitude in radians.
    lat : float
        Latitude in radians.
    getlcd : bool, optional(default False)
        Set to return the LCD triangle index. You get it for free when computing
        position in triangle.

    Returns
    -------
    x_pos : float
        X position in dymaxion coordinates.
    y_pos : float
        Y position in dymaxion coordinates.
    lcd : int, optional
        Lowest-Common-Denominator sub-triangle index.

    Example
    -------
    >>> vert = lonlat2dymax(-77.0367, 38.8951)
    >>> 'x={:.8f}, y={:.8f}'.format(*vert)
    'x=3.30326834, y=1.53381487'
    '''
    # Convert the given(lon, lat) coordinate into spherical
    # polar coordinates(r, theta, phi) with radius=1.
    # Angles are given in radians, NOT degrees.
    theta, phi = lonlat2spherical(lon, lat)

    # convert the spherical polar coordinates into cartesian
    # (x, y, z) coordinates.
    XYZ = spherical2cartesian(theta, phi)
    # determine which of the 20 spherical icosahedron triangles
    # the given point is in and the LCD triangle.
    tri, lcd = fuller_triangle(XYZ)

    # Determine the corresponding Fuller map plane(x, y) point
    x_pos, y_pos = dymax_point(tri, lcd, XYZ)

    if getlcd: return x_pos, y_pos, lcd
    else:      return x_pos, y_pos

### Dymax Conversion Subroutines
def vert2dymax(vert, vertset, push=.9999):
    '''
    Convert Vertex Index to XY Position We need to 'nudge' the point a little
    bit into the triangle. Without the nudge, the vertices would be exactly
    between dymaxion faces and wouldn't make sense for plotting. Hence we do a
    weighted average with point idx having a massive weight

    Example
    -------
    >>> vert = vert2dymax(3, constants.vert_indices[1])
    >>> 'x={:.8f}, y={:.8f}'.format(*vert)
    'x=2.00000033, y=0.86617338'
    '''
    XYZ = np.zeros(3)
    for idx in range(3):
        if vertset[idx] == vert:
            XYZ += constants.vertices[vert] * push
        else:
            XYZ += constants.vertices[vertset[idx]] * (1-push)

    ### Determine the corresponding Fuller map plane(x, y) point
    tri, hlcd = fuller_triangle(XYZ)
    x_pos, y_pos = dymax_point(tri, hlcd, XYZ)
    return x_pos, y_pos

def face2dymax(face_idx, push=.9999, atomic=False):
    '''
    Convert Icosahedron Face to (4) XY Vertices

    Parameters
    ----------
    face_idx : int
        Dymaxion face index.
    push : float
        Multiplier distance from vertex to center. Has the effect of compressing
        the face toward the center for easy plotting usage.
    atomic will draw the LCD subtriangles

    Returns
    -------
    points : ndarray of float
        Vertices correspointing to the face index requested. Normally this is
        4 vertics, but if atomic will return 7 vertices.

    Example
    -------
    >>> verts = face2dymax(1, push=.75)
    >>> for vdx, vert in enumerate(verts):
    ...     print('v{} x={:.8f}, y={:.8f}'.format(vdx, vert[0], vert[1]))
    v0 x=2.35304556, y=1.64720662
    v1 x=1.64695413, y=1.64720662
    v2 x=2.00000025, y=1.03571383
    v3 x=2.35304556, y=1.64720662
    '''
    if atomic:
        points = np.zeros((6+1, 2))
        for jdx in range(6):
            if not jdx % 2: XYZ = constants.vertices[constants.vert_indices[face_idx, jdx//2]] # Normal Vertex
            else:
                up = constants.vertices[constants.vert_indices[face_idx, (jdx//2+1)%3]]
                down = constants.vertices[constants.vert_indices[face_idx, (jdx//2+2)%3]]
                XYZ = np.mean([up, down], axis=0)
            XYZ = XYZ * push + constants.XYZcenters[face_idx] * (1-push)
            tri, hlcd = fuller_triangle(XYZ)
            points[jdx] = dymax_point(tri, hlcd, XYZ)
    else:
        points = np.zeros((3+1, 2))
        for jdx in range(3):
            XYZ = constants.vertices[constants.vert_indices[face_idx, jdx]] * push + constants.XYZcenters[face_idx] * (1-push)
            tri, hlcd = fuller_triangle(XYZ)
            points[jdx] = dymax_point(tri, hlcd, XYZ)

    points[-1] = points[0] # Loop Back to Start
    return points

def lonlat2spherical(lon, lat):
    '''
    Convert (lon, lat) point into spherical polar coordinates with radius=1.
    Angles are given in radians.
    note: Not on WGS84 Ellipsoid

    >>> theta, phi = lonlat2spherical(179, 89)
    >>> 'theta={:.6f}, phi={:.6f}'.format(theta, phi)
    'theta=0.017453, phi=3.124139'
    '''
    h_theta = 90 - lat
    h_phi = lon
    if lon < 0: h_phi = lon + 360
    theta = math.radians(h_theta)
    phi = math.radians(h_phi)
    return theta, phi

def spherical2cartesian(theta, phi):
    '''
    Covert spherical polar coordinates to cartesian coordinates.
    Input angles in radians, output as unit vector.

    Note that numba doesn't speed this one up.

    >>> spherical2cartesian(math.pi/2, math.pi)
    (-1.0, 1.2246467991473532e-16, 6.123233995736766e-17)
    '''
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)
    return x, y, z

def cartesian2spherical(XYZ):
    '''
    Convert Cartesian to Spherical (Non-WGS84)
    Takes a [X,Y,Z] unit vector as input.
    (theta, phi) ~ (lon, lat)

    >>> cartesian2spherical([0.131, -0.84, 0.525])
    (-1.4160901241763815, 1.0180812136981134)
    '''
    phi = math.acos(XYZ[2])
    theta = math.atan2(XYZ[1], XYZ[0])
    return theta, phi

def fuller_triangle(XYZ):
    '''
    Determine which major icosahedron triangle
    and minor lowest common dinominator triangle
    the XYZ point is in. (6 LCDs per Triangle)

    Parameters
    ----------
    xyz : tuple of floats
        Cartesian coordinate position.

    Returns
    -------
    h_tri : int
        Dymaxion triangle index.
    h_lcd : int
        Dymaxion lowest-common-denominator triangle index.

    Example
    -------
    >>> fuller_triangle([-1, 0, 0])
    (10, 2)
    '''
    h_tri = -1
    h_dist1 = np.inf

    # Which triangle face center is the closest to the given point
    # is the triangle in which the given point is in.
    for idx in range(constants.facecount):
        h = constants.XYZcenters[idx] - XYZ
        h_dist2 = magnitude(h)
        if h_dist2 < h_dist1:
            h_tri = idx
            h_dist1 = h_dist2

    # Now the LCD triangle is determined.
    v1, v2, v3 = constants.vert_indices[h_tri]
    h_dist1 = euclidean(XYZ, constants.vertices[v1])
    h_dist2 = euclidean(XYZ, constants.vertices[v2])
    h_dist3 = euclidean(XYZ, constants.vertices[v3])

    if   h_dist1 <= h_dist2 <= h_dist3: h_lcd = 0
    elif h_dist1 <= h_dist3 <= h_dist2: h_lcd = 5
    elif h_dist2 <= h_dist1 <= h_dist3: h_lcd = 1
    elif h_dist2 <= h_dist3 <= h_dist1: h_lcd = 2
    elif h_dist3 <= h_dist1 <= h_dist2: h_lcd = 4
    elif h_dist3 <= h_dist2 <= h_dist1: h_lcd = 3
    return h_tri, h_lcd

def dymax_point(tri, lcd, XYZ):
    '''
    In order to rotate the given point into the template spherical
    triangle, we need the spherical polar coordinates of the center
    of the face and one of the face vertices. So set up which vertex
    to use.

    Parameters
    ----------
    tri : int
        Dymaxion face index where we want to be.
    lcd : int
        Dymaxion sub-triangle where we want to be.
    XYZ : tuple of floats
        Pseudo-ECEF coordinate that will be projected to dymaxion.

    Returns
    -------
    pointx : float
        X position for dymaxion projection.
    pointy : float
        Y position for dymaxion projection.

    Example
    -------
    >>> vert = dymax_point(10, 2, [-1.0, 0, 0])
    >>> 'x={:.8f}, y={:.8f}'.format(*vert)
    'x=3.50247081, y=0.09535516'
    '''
    v1 = constants.vert_indices[tri][0]

    h0XYZ = XYZ
    h1XYZ = constants.vertices[v1]

    theta, phi = cartesian2spherical(constants.XYZcenters[tri])

    axis = 2
    h0XYZ = rotate3d(axis, theta, h0XYZ)
    h1XYZ = rotate3d(axis, theta, h1XYZ)

    axis = 1
    h0XYZ = rotate3d(axis, phi, h0XYZ)
    h1XYZ = rotate3d(axis, phi, h1XYZ)

    theta, phi = cartesian2spherical(h1XYZ)
    theta = theta - np.pi/2

    axis = 2
    h0XYZ = rotate3d(axis, theta, h0XYZ)

    ### exact transformation equations
    gz = math.sqrt(1 - h0XYZ[0]**2 - h0XYZ[1]**2)
    gs = math.sqrt(5 + 2 * math.sqrt(5)) / (gz * math.sqrt(15))

    gxp = h0XYZ[0] * gs
    gyp = h0XYZ[1] * gs

    ga0p = 2 * gyp / math.sqrt(3) + (constants.gel / 3)
    ga1p = gxp - (gyp / math.sqrt(3)) +  (constants.gel / 3)
    ga2p = (constants.gel / 3) - gxp - (gyp / math.sqrt(3))

    ga0 = constants.gt + math.atan2(ga0p - 0.5 * constants.gel, constants.gdve)
    ga1 = constants.gt + math.atan2(ga1p - 0.5 * constants.gel, constants.gdve)
    ga2 = constants.gt + math.atan2(ga2p - 0.5 * constants.gel, constants.gdve)

    gx = 0.5 * (ga1 - ga2)
    gy = (2 * ga0 - ga1 - ga2) / (2 * math.sqrt(3))

    ### Re-scale so plane triangle edge length is 1
    pointx = gx / constants.garc
    pointy = gy / constants.garc

    ### Move and Rotate as Appropriate
    # You can disable the special translations for uniform triangles
    if   tri == 8  and lcd < 4:
        xtranslate, ytranslate, rotation = constants.dymax_translate08_special
    elif tri == 15 and lcd < 3:
        xtranslate, ytranslate, rotation = constants.dymax_translate15_special
    else:
        xtranslate, ytranslate, rotation = constants.dymax_translate[tri]

    pointx, pointy = rotate2d(rotation, pointx, pointy)
    pointx += xtranslate
    pointy += ytranslate
    return pointx, pointy

def rotate2d(angle, pointx, pointy):
    '''
    Rotate a point orientation in XY-plane around Z
    This function obeys the right hand rule.

    >>> rotate2d(90,.5,1)
    (-1.0, 0.5000000000000001)
    '''

    ha = math.radians(angle)
    hx = pointx
    hy = pointy
    pointx = hx * math.cos(ha) - hy * math.sin(ha)
    pointy = hx * math.sin(ha) + hy * math.cos(ha)

    return pointx, pointy

def rotate3d(axis, alpha, XYZ, reverse=True):
    '''
    Rotate a 3-D point about the specified axis by alpha radians
    For some horrible reason, we are doing left hand rotation.
    reverse == left hand rotation, set to False for normal

    >>> rotate3d(0, np.pi/4, [.3, .5, .4])
    (0.3, 0.6363961030678928, -0.07071067811865467)
    '''
    if reverse: alpha = -alpha

    if axis == 0:
        # Rotate around X
        XYZ = (XYZ[0],
               XYZ[1] * math.cos(alpha) - XYZ[2] * math.sin(alpha),
               XYZ[1] * math.sin(alpha) + XYZ[2] * math.cos(alpha))

    elif axis == 1:
        # Rotate around Y
        XYZ = (XYZ[0] * math.cos(alpha) + XYZ[2] * math.sin(alpha),
               XYZ[1],
               -XYZ[0] * math.sin(alpha) + XYZ[2] * math.cos(alpha))

    elif axis == 2:
        # Rotate around Z
        XYZ = (XYZ[0] * math.cos(alpha) - XYZ[1] * math.sin(alpha),
               XYZ[0] * math.sin(alpha) + XYZ[1] * math.cos(alpha),
               XYZ[2])

    return XYZ

@numba.jit
def raytrace(x_loc, y_loc, poly):
    '''
    Determine if position (x_loc, y_loc) is inside polygon.

    Examples
    --------
    >>> raytrace(2.01, .5, [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    False
    >>> raytrace(0.99, .5, [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    True
    '''
    psize = len(poly)
    inside = False
    p2x = 0.
    p2y = 0.
    xints = 0.
    p1x, p1y = poly[0]
    for pdx in range(psize + 1):
        p2x, p2y = poly[pdx % psize]
        if y_loc > min(p1y, p2y):
            if y_loc <= max(p1y, p2y):
                if x_loc <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y_loc-p1y) * (p2x-p1x) / (p2y-p1y) + p1x
                    if p1x == p2x or x_loc <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def benchmark(verbose=True):
    '''
    simple unique point benchmark

    on i7-8550U, points/sec
    v1.0.0: 13600
    v1.1.0: 28000
    v1.1.1: 18300
    '''
    lon_res = 1000
    lat_res = 100
    lons = np.linspace(-180, 180, lon_res)
    lats = np.linspace(-90, 90, lat_res)
    start = time.time()
    for lat in lats:
        for lon in lons:
            _ = lonlat2dymax(lon, lat)
    if verbose:
        print(':: mapped {:d} unique points to dymax projection @ {:.1f} pts/sec [{:.1f} secs total]'.format(
            lon_res * lat_res,
            (lon_res * lat_res) / (time.time()-start),
            time.time()-start))

### Determine (X,Y) Projection Coordinates for Dymaxion Triangle Centers
dymax_centers = np.zeros((constants.facecount, 2))
for fdx in range(constants.facecount):
    tri, hlcd = fuller_triangle(constants.XYZcenters[fdx])
    dymax_centers[fdx] = dymax_point(tri, hlcd, constants.XYZcenters[fdx])
