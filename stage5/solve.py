#!/usr/bin/env python3

# TODO
#   * optional bumpy base
#   * get multigrid solver from stage4/

import sys
import numpy as np
import argparse
from firedrake import *

parser = argparse.ArgumentParser(description=
'''Solve the Glen-Stokes momentum equations for a 3D ice sheet using an
extruded mesh and a multigrid solver.''', add_help=False)
parser.add_argument('-baserefine', type=int, metavar='X', default=2,
    help='how many refinement levels in generating base mesh, a disk')
parser.add_argument('-eps', type=float, metavar='X', default=1.0e-4,
    help='regularization used in viscosity')
parser.add_argument('-refine', type=int, metavar='X', default=2,
    help='number of 3D mesh refinements (for multigrid)')
parser.add_argument('-solvehelp', action='store_true', default=False,
    help='print help for solve.py options and stop')
parser.add_argument('-zlayers', type=int, metavar='X', default=4,
    help='number of levels in extruding mesh')
args, unknown = parser.parse_known_args()
if args.solvehelp:
    parser.print_help()
    sys.exit(0)

R = 10000.0
H = 1000.0

def profile(x, y, R, H):
    '''Exact SIA solution with radius R and maximum height H, on
    disk of radius R centered at (x,y)=(0,0).  See Bueler et al 2005,
    and look at "Test D".  This is the H_s shape.'''
    n = 3.0                       # glen exponent
    p1 = n / (2.0 * n + 2.0)      # = 3/8
    q1 = 1.0 + 1.0 / n            # = 4/3
    r = np.sqrt(x * x + y * y)
    xi = r / R
    lamhat = q1 * xi - (1.0 / n) + (1.0 - xi)**q1 - xi**q1
    s = (H / (1.0 - 1.0 / n)**p1) * lamhat**p1
    return s

printpar = PETSc.Sys.Print        # print once even in parallel
printpar('generating disk mesh and extruding to 3D ...')
basemesh = UnitDiskMesh(refinement_level=args.baserefine)
basemesh.coordinates.dat.data[:] *= R * (1.0 - 1.0e-10) # avoid degeneracy
belements, bnodes = basemesh.num_cells(), basemesh.num_vertices()
printpar('    (base mesh is disk with %d triangle elements and %d vertices%s)' \
         % (belements, bnodes, ' on rank 0' if basemesh.comm.size > 1 else ''))
# here SpatialCoordinate() returns things which give error
#   "AttributeError: 'Indexed' object has no attribute 'dat'"
xbase = basemesh.coordinates.dat.data_ro[:,0]
ybase = basemesh.coordinates.dat.data[:,1]
# set temporary height of 1
mesh = ExtrudedMesh(basemesh, layers=args.zlayers,
                    layer_height=1.0/args.zlayers)
P1base = FunctionSpace(basemesh,'P',1)
sbase = Function(P1base)
sbase.dat.data[:] = profile(xbase, ybase, R, H)
# extend sbase, defined on the base mesh, to the extruded mesh using the
#   'R' constant-in-the-vertical space
Q1R = FunctionSpace(mesh, 'P', 1, vfamily='R', vdegree=0)
s = Function(Q1R)
s.dat.data[:] = sbase.dat.data_ro[:]
Vcoord = mesh.coordinates.function_space()
x, y, z = SpatialCoordinate(mesh)
XYZ = Function(Vcoord).interpolate(as_vector([x, y, s * z]))
mesh.coordinates.assign(XYZ)
printpar('    (extruded %d-layer 3D mesh has %d prism elements and %d vertices%s)' \
         % (args.zlayers, belements * args.zlayers, bnodes * (args.zlayers + 1),
            ' on rank 0' if basemesh.comm.size > 1 else ''))

# now proceed as in stage4/ ... next section is nearly dimension-independent
V = VectorFunctionSpace(mesh, 'Lagrange', 2)
W = FunctionSpace(mesh, 'Lagrange', 1)
Z = V * W
up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)
n_u, n_p = V.dim(), W.dim()
printpar('    (sizes: n_u = %d, n_p = %d, N = %d)' % (n_u,n_p,n_u+n_p))

def D(w):               # strain-rate tensor
    return 0.5 * (grad(w) + grad(w).T)

secpera = 31556926.0    # seconds per year
g = 9.81                # m s-2
rho = 910.0             # kg m-3
n = 3.0
A3 = 3.1689e-24         # Pa-3 s-1;  EISMINT I value of ice softness
B3 = A3**(-1.0/3.0)     # Pa s(1/3);  ice hardness
eps = 0.0001
Dtyp = 1.0 / secpera    # s-1

fbody = Constant((0.0, 0.0, - rho * g))  # 3D version
Du2 = 0.5 * inner(D(u), D(u)) + (eps * Dtyp)**2.0
r = 1.0 / n - 1.0
F = inner(B3 * Du2**(r/2.0) * D(u), D(v)) * dx \
    - p * div(v) * dx \
    - div(u) * q * dx \
    - inner(fbody, v) * dx

# 3D version
bcs = [ DirichletBC(Z.sub(0), Constant((0.0, 0.0, 0.0)), 'bottom'),
        DirichletBC(Z.sub(0), Constant((0.0, 0.0, 0.0)), (1,)) ]

printpar('solving ...')
par = {'snes_linesearch_type': 'bt',
       'mat_type': 'aij',
       'ksp_type': 'preonly',
       'pc_type': 'lu',
       'pc_factor_shift_type': 'inblocks'}
solve(F == 0, up, bcs=bcs, options_prefix='s', solver_parameters=par)

# show average and maximum velocity
P1 = FunctionSpace(mesh, 'CG', 1)
one = Constant(1.0, domain=mesh)
area = assemble(dot(one,one) * dx)
umagav = assemble(sqrt(dot(u, u)) * dx) / area
umag = interpolate(sqrt(dot(u, u)), P1)
with umag.dat.vec_ro as vumag:
    umagmax = vumag.max()[1]
printpar('  ice speed (m a-1): av = %.3f, max = %.3f' \
         % (umagav * secpera, umagmax * secpera))

printpar('saving to dome.pvd ...')
u, p = up.split()
u.rename('velocity')
p.rename('pressure')
# integer-valued element-wise process rank
rank = Function(FunctionSpace(mesh,'DG',0))
rank.dat.data[:] = mesh.comm.rank
rank.rename('rank')
File('dome.pvd').write(u, p, rank)
