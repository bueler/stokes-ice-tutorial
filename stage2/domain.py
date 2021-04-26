#!/usr/bin/env python3

import numpy as np

def profile(mx, R, H):
    '''Exact SIA solution with half-length (radius) R and maximum height H, on
    interval [0,L] = [0,2R], centered at x=R.  See van der Veen (2013)
    equation (5.50).'''
    n = 3.0                       # glen exponent
    p1 = n / (2.0 * n + 2.0)      # = 3/8
    q1 = 1.0 + 1.0 / n            # = 4/3
    Z = H / (n - 1.0)**p1         # outer constant
    x = np.linspace(0.0, 2.0 * R, mx+1)
    X = (x - R) / R               # rescaled coord
    Xin = abs(X[abs(X) < 1.0])    # rescaled distance from center
    Yin = 1.0 - Xin
    s = np.zeros(np.shape(x))     # correct outside ice
    s[abs(X) < 1.0] = Z * ( (n + 1.0) * Xin - 1.0 \
                            + n * Yin**q1 - n * Xin**q1 )**p1
    return x, s

def writegeometry(geo, xtop, ytop, top=41, base=42, offset=51):
    '''Write a .geo file which saves the profile geometry.  Boundary
    order starts with (0.0,0.0), then does the base, then does the top,
    so boundary is traversed clockwise.'''
    # points on top
    ntop = len(xtop)
    geo.write('Point(%d) = {%f,%f,0,lc};\n' % (offset, 0.0, 0.0))
    for j in range(ntop-1):
        geo.write('Point(%d) = {%f,%f,0,lc};\n' \
                  % (offset + j + 1,xtop[-j-1],ytop[-j-1]))
    # line along base
    linestart = offset + ntop
    geo.write('Line(%d) = {%d,%d};\n' % (linestart, offset, offset + 1))
    # lines along top boundary
    for j in range(ntop-2):
        geo.write('Line(%d) = {%d,%d};\n' \
                  % (linestart + 1 + j, offset + j + 1, offset + j + 2))
    geo.write('Line(%d) = {%d,%d};\n' \
              % (linestart + ntop - 1, offset + ntop - 1, offset))
    # full line loop
    lineloopstart = linestart + ntop
    geo.write('Line Loop(%d) = {' % lineloopstart)
    for j in range(ntop-1):
        geo.write('%d,' % (linestart + j))
    geo.write('%d};\n' % (linestart + ntop - 1))
    # surface allows defining a 2D mesh
    surfacestart = lineloopstart + 1
    geo.write('Plane Surface(%d) = {%d};\n' % (surfacestart, lineloopstart))
    # "Physical" for marking boundary conditions
    geo.write('// boundary id = %d is top\n' % top)
    geo.write('Physical Line(%d) = {' % top)
    for j in range(ntop-2):
        geo.write('%d,' % (linestart + j + 1))
    geo.write('%d};\n' % (linestart + ntop - 1))
    geo.write('// boundary id = %d is base\n' % base)
    geo.write('Physical Line(%d) = {%d};\n' % (base, linestart))
    # ensure all interior elements are written ... NEEDED!
    geo.write('Physical Surface(%d) = {%d};\n' % \
              (surfacestart + 1, surfacestart))

filename = 'dome.geo'
print('writing domain geometry to file %s ...' % filename)
geo = open(filename, 'w')
mx = 50
geo.write('// geometry-description written by domain.py for mx=%d\n' % mx)
R = 10000.0
H = 1000.0
lc = 0.5 * (2.0 * R) / mx         # "characteristic length" used by gmsh
geo.write('lc = %f;\n' % lc)
x, ytop = profile(mx, R, H)
writegeometry(geo, x, ytop)       # create the rest of the file
geo.close()
