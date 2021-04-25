#!/usr/bin/env python3

import sys
import argparse
import numpy as np
from firedrake import *

parser = argparse.ArgumentParser(description=
'''Solve the Glen-Stokes momentum equations for a 2D ice sheet using an
extruded mesh.''', add_help=False)
parser.add_argument('-mx', type=int, metavar='MX', default=50,
    help='number of subintervals')
parser.add_argument('-mz', type=int, metavar='MZ', default=8,
    help='number of subintervals')
parser.add_argument('-solvehelp', action='store_true', default=False,
    help='print help for solve.py options and stop')
args, unknown = parser.parse_known_args()
if args.solvehelp:
    parser.print_help()
    sys.exit(0)

def profile(mx, R, H):
    '''Exact solution with half-length (radius) R and maximum height H, on
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

def extend(mesh, f):
    '''On an extruded mesh extend a function f(x,z), already defined on the
    base mesh, to the mesh using the 'R' constant-in-the-vertical space.'''
    Q1R = FunctionSpace(mesh, 'P', 1, vfamily='R', vdegree=0)
    fextend = Function(Q1R)
    fextend.dat.data[:] = f.dat.data_ro[:]
    return fextend

print('generating %d x %d mesh by extrusion ...' % (args.mx, args.mz))
R = 10000.0
H = 1000.0
# extrude a mesh with a temporary height of 1.0
base_mesh = IntervalMesh(args.mx, length_or_left=0.0, right=2.0*R)
mesh = ExtrudedMesh(base_mesh, layers=args.mz, layer_height=1.0/args.mz)
x, z = SpatialCoordinate(mesh)
# correct the height to equal the profile
xbase = base_mesh.coordinates.dat.data_ro
P1base = FunctionSpace(base_mesh,'P',1)
zz = Function(P1base)
_, zz.dat.data[:] = profile(args.mx, R, H)
Vcoord = mesh.coordinates.function_space()
XZ = Function(Vcoord).interpolate(as_vector([x, extend(mesh, zz) * z]))
mesh.coordinates.assign(XZ)
# now proceed as in stage2/ ...

V = VectorFunctionSpace(mesh, 'Lagrange', 2)
W = FunctionSpace(mesh, 'Lagrange', 1)
Z = V * W
up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)

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

fbody = Constant((0.0, - rho * g))
Du2 = 0.5 * inner(D(u), D(u)) + (eps * Dtyp)**2.0
r = 1.0 / n - 1.0
F = inner(B3 * Du2**(r/2.0) * D(u), D(v)) * dx \
    - p * div(v) * dx \
    - div(u) * q * dx \
    - inner(fbody, v) * dx

# different boundary conditions relative to stage2/:
#   base label is 'bottom', and we add noslip condition on degenerate ends
bcs = [ DirichletBC(Z.sub(0), Constant((0.0, 0.0)), 'bottom'),
        DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (1,2)) ]

par = {'snes_linesearch_type': 'bt',
       'ksp_type': 'gmres',
       'pc_type': 'fieldsplit',
       'pc_fieldsplit_type': 'schur',
       'pc_fieldsplit_schur_factorization_type': 'full',
       'pc_fieldsplit_schur_precondition': 'a11',
       'fieldsplit_0_ksp_type': 'preonly',
       'fieldsplit_0_pc_type': 'lu',
       'fieldsplit_1_ksp_rtol': 1.0e-3,
       'fieldsplit_1_ksp_type': 'gmres',
       'fieldsplit_1_pc_type': 'none'}

print('solving ...')
solve(F == 0, up, bcs=bcs, options_prefix='s', solver_parameters=par)

print('saving to dome.pvd ...')
u, p = up.split()
u.rename('velocity')
p.rename('pressure')
File('dome.pvd').write(u, p)
