#!/usr/bin/env python3

import sys
import argparse
import numpy as np
from firedrake import *
from firedrake.petsc import PETSc

parser = argparse.ArgumentParser(description=
'''stage3/  Solve the Glen-Stokes momentum equations for a 2D ice sheet using an
extruded mesh.''', add_help=False)
parser.add_argument('-eps', type=float, metavar='X', default=1.0e-4,
    help='regularization used in viscosity')
parser.add_argument('-mx', type=int, metavar='MX', default=50,
    help='number of subintervals')
parser.add_argument('-mz', type=int, metavar='MZ', default=8,
    help='number of vertical layers')
parser.add_argument('-solvehelp', action='store_true', default=False,
    help='print help for solve.py options and stop')
args, unknown = parser.parse_known_args()
if args.solvehelp:
    parser.print_help()
    sys.exit(0)

def profile(x, R, H):
    '''Exact SIA solution for surface elevation, with half-length (radius) R
    and maximum height H, on interval [0,L] = [0,2R], centered at x=R.
    See van der Veen (2013) equation (5.50).'''
    n = 3.0                       # glen exponent
    p1 = n / (2.0 * n + 2.0)      # = 3/8
    q1 = 1.0 + 1.0 / n            # = 4/3
    Z = H / (n - 1.0)**p1         # outer constant
    X = (x - R) / R               # rescaled coord
    Xin = abs(X[abs(X) < 1.0])    # rescaled distance from center
    Yin = 1.0 - Xin
    s = np.zeros(np.shape(x))
    s[abs(X) < 1.0] = Z * ( (n + 1.0) * Xin - 1.0 \
                            + n * Yin**q1 - n * Xin**q1 )**p1
    return s

printpar = PETSc.Sys.Print        # print once even in parallel
printpar('generating %d x %d mesh by extrusion ...' % (args.mx, args.mz))
R = 10000.0
H = 1000.0
# extrude a mesh with a temporary height of 1.0
base_mesh = IntervalMesh(args.mx, length_or_left=0.0, right=2.0*R)
mesh = ExtrudedMesh(base_mesh, layers=args.mz, layer_height=1.0/args.mz)
# compute the profile height
xbase = base_mesh.coordinates.dat.data_ro
P1base = FunctionSpace(base_mesh,'P',1)
sbase = Function(P1base)
sbase.dat.data[:] = profile(xbase, R, H)
# extend sbase, defined on the base mesh, to the extruded mesh using the
#   'R' constant-in-the-vertical space
Q1R = FunctionSpace(mesh, 'P', 1, vfamily='R', vdegree=0)
s = Function(Q1R)
s.dat.data[:] = sbase.dat.data_ro[:]
Vcoord = mesh.coordinates.function_space()
x, z = SpatialCoordinate(mesh)
XZ = Function(Vcoord).interpolate(as_vector([x, s * z]))
mesh.coordinates.assign(XZ)

# now proceed basically as in stage2/
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
Dtyp = 1.0 / secpera    # s-1

fbody = Constant((0.0, - rho * g))
Du2 = 0.5 * inner(D(u), D(u)) + (args.eps * Dtyp)**2.0
nu = 0.5 * B3 * Du2**((1.0 / n - 1.0)/2.0)
F = ( inner(2.0 * nu * D(u), D(v)) \
      - p * div(v) - q * div(u) - inner(fbody, v) ) * dx

# different boundary conditions relative to stage2/, for extruded case:
#   base label is 'bottom', and we add noslip condition on degenerate ends
bcs = [ DirichletBC(Z.sub(0), Constant((0.0, 0.0)), 'bottom'),
        DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (1,2)) ]

printpar('solving ...')
par = {'snes_linesearch_type': 'bt',
       'ksp_type': 'preonly',
       'pc_type': 'lu',
       'pc_factor_shift_type': 'inblocks'}
solve(F == 0, up, bcs=bcs, options_prefix='s', solver_parameters=par)

# print average and maximum velocity
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
u = up.subfunctions[0]
p = up.subfunctions[1]
u *= secpera
p /= 1.0e5
u.rename('velocity (m/a)')
p.rename('pressure (bar)')
File('dome.pvd').write(u, p)
