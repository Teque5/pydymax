#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''Constants for Dymaxion Projection Module'''
import math
import numpy as np

### Quick Vector Functions
magnitude = lambda vector: np.sqrt(np.dot(vector, vector))

### Icosahedron Time
facecount, vertexcount = 20, 12

### 20 Faces             [v0, v1, v2]
vert_indices = np.array([[ 0,  1,  2], [ 0,  2,  3], [ 0,  3,  4], [ 0,  4,  5],
                         [ 0,  1,  5], [ 1,  2,  7], [ 2,  7,  8], [ 2,  3,  8],
                         [ 3,  8,  9], [ 3,  4,  9], [ 4,  9, 10], [ 4,  5, 10],
                         [ 5,  6, 10], [ 1,  5,  6], [ 1,  6,  7], [ 7,  8, 11],
                         [ 8,  9, 11], [ 9, 10, 11], [ 6, 10, 11], [ 6,  7, 11]])

### 12 XYZ vertices  [        x,            y,         z]
vertices = np.array([[ 0.420152,     0.078145,  0.904083],
                     [ 0.995005,    -0.091348,  0.040147],
                     [ 0.518837,     0.835420,  0.181332],
                     [-0.414682,     0.655962,  0.630676],
                     [-0.515456,    -0.381717,  0.767201],
                     [ 0.355781,    -0.843580,  0.402234],
                     [ 0.414682,    -0.655962, -0.630676],
                     [ 0.515456,     0.381717, -0.767201],
                     [-0.355781,     0.843580, -0.402234],
                     [-0.995009,     0.091348, -0.040147],
                     [-0.518837,    -0.835420, -0.181332],
                     [-0.420152,    -0.078145, -0.904083]])

### 12 Lon, Lat vertices  [       lon,        lat]
lon_lat_verts = np.array([[10.53620  ,  64.7     ],
                          [  -5.24539,   2.300882],
                          [  58.15771,  10.447378],
                          [ 122.3    ,  39.1     ],
                          [-143.47849,  50.103201],
                          [ -67.13233,  23.717925],
                          [ -57.7    , -39.1     ],
                          [  36.5215 , -50.1032  ],
                          [  36.5215 , -50.1032  ],
                          [ 174.7546 ,  -2.3009  ],
                          [-121.84229, -10.447345],
                          [-169.4638 , -64.7     ]])

### Calculate Spherical Triangle Centers
XYZcenters = np.zeros((facecount, 3))
for idx in range(facecount):
    verts = np.array([vertices[jdx] for jdx in vert_indices[idx]])
    vertmean = np.mean(verts, axis=0)
    XYZcenters[idx] = vertmean/magnitude(vertmean)

# Translate                          [X,   Y,                 Rotation]
dymax_translate = np.array([         [2.0, 7 / (2 * math.sqrt(3)), 240],
                                     [2.0, 5 / (2 * math.sqrt(3)), 300],
                                     [2.5, 2 / math.sqrt(3),         0],
                                     [3.0, 5 / (2 * math.sqrt(3)),  60],
                                     [2.5, 4 * math.sqrt(3) / 3.0, 180],
                                     [1.5, 4 * math.sqrt(3) / 3.0, 300],
                                     [1.0, 5 / (2 * math.sqrt(3)), 300],
                                     [1.5, 2 / math.sqrt(3),         0],
                                     [2.0, 1 / (2 * math.sqrt(3)),   0],
                                     [2.5, 1 / math.sqrt(3),        60],
                                     [3.5, 1 / math.sqrt(3),        60],
                                     [3.5, 2 / math.sqrt(3),       120],
                                     [4.0, 5 / (2 * math.sqrt(3)),  60],
                                     [4.0, 7 / (2 * math.sqrt(3)),   0],
                                     [5.0, 7 / (2 * math.sqrt(3)),   0],
                                     [5.5, 2 / math.sqrt(3),         0],
                                     [1.0, 1 / (2 * math.sqrt(3)),   0],
                                     [4.0, 1 / (2 * math.sqrt(3)), 120],
                                     [4.5, 2 / math.sqrt(3),         0],
                                     [5.0, 5 / (2 * math.sqrt(3)),  60]])
dymax_translate08_special = np.array([1.5, 1 / math.sqrt(3),       300]) # if LCD < 4
dymax_translate15_special = np.array([0.5, 1 / math.sqrt(3),        60]) # if LCD < 3

### Optimizations
garc = 2 * math.asin(math.sqrt( (5 - math.sqrt(5)) / 10))
gt = garc / 2
gdve = math.sqrt((3 + math.sqrt(5)) / (5 + math.sqrt(5)))
gel = math.sqrt(8 / (5 + math.sqrt(5)))
