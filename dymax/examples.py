#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''Dymaxion Projection Examples'''
import os
import time
from sys import stdout
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from PIL import Image, ImageOps

from . import convert
from . import constants
from . import io

def plot_triangles(save=False, show=True, dpi=300, verbose=True):
    '''Draw Dymax Spherical Triangles'''
    plt.figure(figsize=(20, 12))
    for jdx in range(constants.facecount):
        if jdx in [8, 15]: continue
        points = convert.face2dymax(jdx, push=.95)
        xcenter, ycenter = convert.dymax_centers[jdx]
        plt.text(xcenter, ycenter, str(jdx), ha='center', va='center')
        plt.plot(points[:, 0], points[:, 1], lw=5, alpha=.5)
    plt.gca().set_xlim([0, 5.5])
    plt.gca().set_ylim([0, 2.6])
    plt.gca().set_aspect('equal')
    if verbose: print(':: plotted', constants.facecount, 'triangles')
    if save: plt.savefig('dymax_triangles.png', bbox_inches='tight', dpi=dpi, transparent=True, pad_inches=0)
    if show:
        plt.tight_layout()
        plt.show()
    else: plt.close()

def plot_triangles_meridians(resolution='c', save=False, show=True, dpi=300, verbose=True):
    '''Draw Dymax Triangles, All countries, and Meridians'''
    lonlat_islands, dymax_islands = io.get_coastlines(resolution)
    n = 1000
    plt.figure(figsize=(20, 12))
    plt.title('Dymaxion Map Projection')

    ### Dymaxion Latitude Meridians
    lons = np.linspace(-180, 180, n)
    latgrid = np.linspace(-85, 85, 35)
    points = []
    start = time.time()
    for lat in latgrid:
        for lon in lons:
            points += [convert.lonlat2dymax(lon, lat)]
    if verbose: print(':: mapped {:d} points to dymax projection @ {:.1f} pts/sec [{:.1f} secs total]'.format(len(points), len(points)/(time.time()-start), time.time()-start))
    points = np.array(points)
    plt.plot(points[:, 0], points[:, 1], ',', c='k', alpha=.3)#,'.',lw=0)#,c=range(n))

    ### Dymaxion Longitude Meridians
    lats = np.linspace(-85, 85, n)
    longrid = np.linspace(-180, 175, 72)
    points = []
    start = time.time()
    for lon in longrid:
        for lat in lats:
            points += [convert.lonlat2dymax(lon, lat)]
    if verbose: print(':: mapped {:d} points to dymax projection @ {:.1f} pts/sec [{:.1f} secs total]'.format(len(points), len(points)/(time.time()-start), time.time()-start))
    points = np.array(points)
    plt.plot(points[:, 0], points[:, 1], ',', c='k', alpha=.3)#,'.',lw=0)#,c=range(n))

    ### Dymaxion Face Tiles
    for jdx in range(constants.facecount):
        if jdx in [8, 15]: continue
        points = convert.face2dymax(jdx, push=.95)
        xcenter, ycenter = convert.dymax_centers[jdx]
        plt.text(
            xcenter, ycenter, str(jdx),
            size='x-large', ha='center', va='center')
        plt.plot(points[:, 0], points[:, 1], lw=5, alpha=.7)

    ### Draw Landmasses
    patches = []
    for island in dymax_islands:
        polygon = Polygon(np.array(island))#, closed=False, fill=False)
        patches.append(polygon)

    p = PatchCollection(patches, alpha=.3, linewidths=1, facecolors=None)
    colors = 100*np.random.random(len(patches))
    p.set_array(np.array(colors))
    plt.gca().add_collection(p)
    if verbose: print(':: plotted', len(patches), 'coastlines')
    plt.gca().set_xlim([0, 5.5])
    plt.gca().set_ylim([0, 2.6])
    plt.gca().set_aspect('equal')
    if save: plt.savefig('dymax_earthmeridianstriangles.png', bbox_inches='tight', dpi=dpi, transparent=True, pad_inches=0)
    if show:
        plt.tight_layout()
        plt.show()
    else: plt.close()

def plot_triangles_rectilinear(resolution='c', save=False, show=True, dpi=300, verbose=True):
    lonlat_islands, dymax_islands = io.get_coastlines(resolution)
    plt.figure(figsize=(20, 12))
    plt.title('The dymax face polygons look super-fucked on a rectilinear projection')
    patches = []
    faces = []
    for island in lonlat_islands:
        polygon = Polygon(np.array(island), closed=False, fill=True)
        patches.append(polygon)

    for face in range(constants.facecount):
        derp = np.zeros((3, 2))
        for vtex in range(3):
            derp[vtex] = constants.lon_lat_verts[constants.vert_indices[face, vtex]]
        polygon = Polygon(derp, closed=False, fill=True)
        faces.append(polygon)

    colors = 100*np.random.random(len(patches))
    p = PatchCollection(patches, cmap=plt.cm.viridis, alpha=0.7, linewidths=0)
    f = PatchCollection(faces, cmap=plt.cm.viridis, alpha=0.3, linewidths=1)
    p.set_array(np.array(colors))
    f.set_array(np.array(colors))
    plt.gca().add_collection(p)
    plt.gca().add_collection(f)
    if verbose: print(':: plotted', len(patches), 'coastlines')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    if save: plt.savefig('dymax_rectilineartriangles.png', bbox_inches='tight', dpi=dpi, transparent=True, pad_inches=0)
    if show:
        plt.tight_layout()
        plt.show()
    else: plt.close()

def plot_lcd_triangles(verbose=True, save=False, show=True, dpi=300, resolution='c'):
    '''Each Icosahedron Face has six sub-triangles that are splitting on.'''
    lonlat_islands, dymax_islands = io.get_coastlines(resolution)
    plt.figure(figsize=(20, 12))

    xs, ys = [], []
    lcds = []
    for _ in range(10000):
        lon = np.random.random()*360 - 180
        lat = np.random.random()*180 - 90
        x, y, lcd = convert.lonlat2dymax(lon, lat, getlcd=True)
        xs += [x]
        ys += [y]
        lcds += [lcd]

    ### Draw LCD Triangle Indices
    colorlist = 'rgcmyg'
    for val in range(10000):
        plt.text(
            xs[val], ys[val], str(lcds[val]),
            color=colorlist[lcds[val]], alpha=.5,
            ha='center', va='center', size='small')

    ### Draw Large Fuller Triangles
    for jdx in range(constants.facecount):
        if jdx in [8, 15]: continue # Edge Triangles
        points = convert.face2dymax(jdx, push=.95)
        xcenter, ycenter = convert.dymax_centers[jdx]
        plt.plot(points[:, 0], points[:, 1], lw=5, alpha=.5)

    ### Draw Fuller LCD Sub-Triangles
    for jdx in range(constants.facecount):
        if jdx in [8, 15]: continue # Edge Triangles
        points = convert.face2dymax(jdx, push=.999, atomic=True)
        xcenter, ycenter = convert.dymax_centers[jdx]
        plt.text(xcenter, ycenter, str(jdx), size='xx-large', ha='center', va='center')
        plt.plot(points[:, 0], points[:, 1], lw=1, alpha=1, color='k', ls='dotted')

    ### Draw Landmasses
    patches = []
    for island in dymax_islands:
        polygon = Polygon(np.array(island), closed=True, fill=True)
        patches.append(polygon)

    colors = 100*np.random.random(len(patches))
    p = PatchCollection(patches, cmap=plt.cm.viridis, alpha=.5, linewidths=0)
    p.set_array(np.array(colors))
    plt.gca().add_collection(p)
    if verbose: print(':: plotted', len(patches), 'coastlines')
    plt.gca().set_xlim([0, 5.5])
    plt.gca().set_ylim([0, 2.6])
    plt.gca().set_aspect('equal')

    if save: plt.savefig('dymax_earthsubtriangles.png', bbox_inches='tight', dpi=dpi, transparent=True, pad_inches=0)
    if show:
        plt.tight_layout()
        plt.show()
    else: plt.close()

def plot_grid(verbose=True, save=False, show=True, dpi=300):
    '''Show Dymaxion Grid'''
    plt.figure(figsize=(20, 12))
    patches = []
    for zdx, vertset in enumerate(constants.vert_indices):
        if zdx in [8, 15]: continue # Edge Triangles
        x, y = [], []
        for _, vert in enumerate(vertset):
            xt, yt = convert.vert2dymax(vert, vertset)
            #print(xt,yt)
            x += [xt]
            y += [yt]
            #print(xt,yt,i,vert)
        #plt.plot(x,y,'k',lw=.1)
        patches.append(Polygon(np.array([x, y]).T, closed=False, fill=True))

    colors = 100*np.random.random(len(patches))
    p = PatchCollection(patches, cmap=plt.cm.viridis, alpha=1, linewidths=0.)
    p.set_array(np.array(colors))
    plt.gca().add_collection(p)
    if verbose: print(':: plotted', len(patches), 'coastlines')
    plt.gca().set_aspect('equal')
    plt.gca().set_xlim([0, 5.5])
    plt.gca().set_ylim([0, 2.6])
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.gca().axis('off')

    if save: plt.savefig('dymax_grid.png', bbox_inches='tight', dpi=dpi, transparent=True, pad_inches=0)
    if show:
        plt.tight_layout()
        plt.show()
    else: plt.close()

def plot_coastline_vectors(verbose=True, save=False, show=True, dpi=300, resolution='c'):
    '''Draw Landmasses Only, no Background'''
    lonlat_islands, dymax_islands = io.get_coastlines(resolution)

    patches = []
    for island in dymax_islands:
        polygon = Polygon(np.array(island), closed=True, fill=True)
        patches.append(polygon)

    plt.figure(figsize=(20, 12), frameon=False)
    colors = 100 * np.random.random(len(patches))
    p = PatchCollection(patches, cmap=plt.cm.viridis, alpha=1, linewidths=0.)
    p.set_array(np.array(colors))
    plt.gca().add_collection(p)
    if verbose: print(':: plotted', len(patches), 'coastlines')
    plt.gca().set_aspect('equal')
    plt.gca().set_xlim([0, 5.5])
    plt.gca().set_ylim([0, 2.6])
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.gca().axis('off')
    if save: plt.savefig('dymax_landmasses.png', bbox_inches='tight', dpi=dpi, transparent=True, pad_inches=0)
    if show:
        plt.tight_layout()
        plt.show()
    else: plt.close()

def convert_rectimage_2_dymaximage(inFilename, outFilename, verbose=True, scale=300, speedup=1, save=False, show=True):
    '''
    Convert rectilinear image to dymax projection image.

    scale is number of pixels per dymax xy unit.
    How to calculate output scale:
        width = 160 # in cm
        resolution = 30 # in px/cm
        scale = (width * resolution) / 5.5
        __OR__
        final_size_in_pixels = (scale * 5.5, scale * 2.6)

    speedup gives a sparse preview of the output image and is specified as a
    time divisor.
    '''
    start = time.time()
    im = Image.open(inFilename) #Can be many different formats. #15 vertical and horizontal pixels per degree
    pix = im.load()
    if verbose: print(':: input image resolution =', im.size) # Get the width and hight of the image for iterating over

    ### LongLat2Dymax returns x = (0,5.5) and y=(0,2.6)
    dymax_xsize, dymax_ysize = int(5.5*scale), int(2.6*scale)
    dymaximg = Image.new('RGBA', (dymax_xsize, dymax_ysize), (255, 0, 0, 0)) # create a new transparent
    if verbose: print(':: output image resolution =', (dymax_xsize, dymax_ysize)) # Get the width and hight of the image for iterating over

    ### X and Y are indexed from topleft to bottom right
    if verbose: print(':: sweeping over Longitudes:')
    xsize, ysize = im.size
    for i, lon in enumerate(np.linspace(-180, 180, xsize/speedup, endpoint=True)):
        i *= speedup
        if i % 20 == 0:
            print('{:+07.2f} '.format(lon), end='')
            stdout.flush() # I would add flush=True to print, but thats only in python3.3+
        for j, lat in enumerate(np.linspace(90, -90, ysize/speedup, endpoint=True)):
            j *= speedup
            newx, newy = convert.lonlat2dymax(lon, lat)
            newx = int(newx*scale) - 1
            newy = int(newy*scale)
            try: dymaximg.putpixel((newx, newy), pix[i, j])
            # Sometimes a point won't map to an edge properly
            except IndexError: print('{{{:d}, {:d}}}'.format(newx, newy), end='')
    if verbose: print()
    dymaximg = ImageOps.flip(dymaximg) #it's upside down since putpixel flips too

    numpoints = im.size[0] * im.size[1] // speedup
    if verbose: print(':: mapped {:d} points to dymax projection @ {:.1f} pts/sec [{:.1f} secs total]'.format(numpoints, numpoints/(time.time()-start), time.time()-start))
    plt.figure(figsize=(20, 12), frameon=False)
    plt.gca().axis('off')
    if save: dymaximg.save(outFilename, format='PNG')
    if show:
        plt.tight_layout()
        plt.imshow(dymaximg)
        plt.show()
    else: plt.close()

def plot_face_hq(resolution='i', save=False, show=True, verbose=True, dpi=300):
    '''Draw Dymax Triangles, All countries, and Meridians'''
    lonlat_islands, dymax_islands = io.get_coastlines(resolution)
    # select a face
    fdx = np.random.randint(0, 20)
    verts = constants.vert_indices[fdx]
    # contants.vertices
    # for vert in verts

    plt.figure(figsize=(20, 20))
    plt.title('Dymaxion Face {}'.format(fdx))
    # Plot Edges
    edges = convert.face2dymax(fdx)
    plt.plot(edges[:, 0], edges[:, 1], lw=1, alpha=.7, color='r')

    n = 1000 # meridian resolution
    ### Dymaxion Latitude Meridians
    lons = np.linspace(-180, 180, n)
    latgrid = np.linspace(-85, 85, 35)
    points = []
    # print(edges)
    for lat in latgrid:
        for lon in lons:
            point = convert.lonlat2dymax(lon, lat)
            if convert.raytrace(point[0], point[1], edges):
                points += [point]
    points = np.array(points)
    plt.plot(points[:, 0], points[:, 1], ',', color='k', alpha=.5)#,'.',lw=0)#,c=range(n))

    ### Dymaxion Longitude Meridians
    lats = np.linspace(-85, 85, n)
    longrid = np.linspace(-180, 175, 72)
    points = []
    start = time.time()
    for lon in longrid:
        for lat in lats:
            point = convert.lonlat2dymax(lon, lat)
            if convert.raytrace(point[0], point[1], edges):
                points += [point]
    points = np.array(points)
    plt.plot(points[:, 0], points[:, 1], ',', color='k', alpha=.5)#,'.',lw=0)#,c=range(n))

    ### Draw Landmasses
    patches = []
    for island in dymax_islands:
        island_closed = []
        for vertex in island:
            if convert.raytrace(vertex[0], vertex[1], edges):
                island_closed += [vertex]
        if len(island_closed) > 0:
            polygon = Polygon(np.array(island_closed), closed=True)#, closed=False, fill=False)
            patches += [polygon]
    p = PatchCollection(patches, alpha=1, linewidths=.8, edgecolor='k', facecolor='none')
    plt.gca().add_collection(p)
    if verbose: print(':: plotted', len(patches), 'coastlines')

    # Draw final figure bits
    plt.axis('off')
    plt.gca().set_aspect('equal')
    if save: plt.savefig('dymax_earthmeridianstriangles.png', bbox_inches='tight', dpi=dpi, transparent=True, pad_inches=0)
    if show:
        plt.tight_layout()
        plt.show()
    else: plt.close()

def run_examples(resolution='c', save=False, show=True, verbose=True):
    '''
    Run all the examples in this file.
    The first part of this is really fast, the image conversion stuff
    '''
    if verbose: print('>> Running Dymax Projection Examples')
    plot_triangles(save=save, show=show)
    plot_triangles_meridians(resolution=resolution, save=save, show=show)
    plot_triangles_rectilinear(resolution=resolution, save=save, show=show)
    plot_lcd_triangles(resolution=resolution, save=save, show=show)
    plot_grid(save=save, show=show)
    plot_coastline_vectors(resolution=resolution, save=save, show=show)
    plot_face_hq(resolution=resolution, save=save, show=show)
    convert_rectimage_2_dymaximage(io.PKG_DATA+'bmng.jpg', 'dymax_bmng.png', save=save, show=show)
    convert_rectimage_2_dymaximage(io.PKG_DATA+'etopo1.jpg', 'dymax_etopo1.png', save=save, show=show)

if __name__ == '__main__':
    run_examples()
