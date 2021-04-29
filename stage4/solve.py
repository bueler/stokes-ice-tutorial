#!/usr/bin/env python3

import sys
import argparse
import numpy as np
from firedrake import *

# recover stage3/:
#     ./solve.py -refine 0 -mz 8 -marginheight 0.0

# performance demo (1 min run time on my thelio)
#     tmpg -n 12 ./solve.py -s_snes_converged_reason -mx 4000 -refine 2 -s_snes_monitor -s_snes_atol 1.0e-2

parser = argparse.ArgumentParser(description=
'''stage4/  Solve the Glen-Stokes momentum equations for a 2D ice sheet using
an extruded mesh, rescaled equations, vertical grid sequencing, and physical
diagnostics.''', add_help=False)
parser.add_argument('-eps', type=float, metavar='X', default=1.0e-4,
    help='regularization used in viscosity (default=10^{-4})')
parser.add_argument('-marginheight', type=float, metavar='X', default=1.0,
    help='height of degeneration point at margin (default=1 m)')
parser.add_argument('-mx', type=int, metavar='MX', default=50,
    help='subintervals in coarse mesh (default=50)')
parser.add_argument('-mz', type=int, metavar='MZ', default=2,
    help='vertical layers in coarse mesh (default=2)')
parser.add_argument('-o', metavar='FILE.pvd', type=str, default='dome.pvd',
    help='output filename (default=dome.pvd)')
parser.add_argument('-refine', type=int, metavar='X', default=1,
    help='refinements when generating mesh hierarchy (default=1)')
parser.add_argument('-refinefactor', type=int, metavar='X', default=4,
    help='refinement factor when generating mesh hierarchy (default=4)')
parser.add_argument('-single', action='store_true', default=False,
    help='solve only on the finest level, without grid sequencing')
parser.add_argument('-solvehelp', action='store_true', default=False,
    help='print help for solve.py options and stop')
args, unknown = parser.parse_known_args()
if args.solvehelp:
    parser.print_help()
    sys.exit(0)

def profile(x, R, H):
    '''Exact SIA solution with half-length (radius) R and maximum height H, on
    interval [0,L] = [0,2R], centered at x=R.  See van der Veen (2013)
    equation (5.50).'''
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
    s[s < 1.0] = args.marginheight   # needed so that prolong() can find nodes
    return s

# level-independent information
secpera = 31556926.0    # seconds per year
g = 9.81                # m s-2
rho = 910.0             # kg m-3
n = 3.0
A3 = 3.1689e-24         # Pa-3 s-1;  EISMINT I value of ice softness
B3 = A3**(-1.0/3.0)     # Pa s(1/3);  ice hardness
Dtyp = 1.0 / secpera    # s-1
sc = 1.0e-7             # velocity scale constant for symmetric equation scaling
fbody = Constant((0.0, - rho * g))
par = {'snes_linesearch_type': 'bt', 'ksp_type': 'preonly',
       'pc_type': 'lu', 'pc_factor_shift_type': 'inblocks'}
printpar = PETSc.Sys.Print        # print once even in parallel

def D(w):               # strain-rate tensor
    return 0.5 * (grad(w) + grad(w).T)

printpar('generating %d-level mesh hierarchy ...' % (args.refine + 1))
R = 10000.0
H = 1000.0
basemesh = IntervalMesh(args.mx, length_or_left=0.0, right=2.0*R)
xbase = basemesh.coordinates.dat.data_ro
P1base = FunctionSpace(basemesh,'P',1)
sbase = Function(P1base)
sbase.dat.data[:] = profile(xbase, R, H)

hierarchy = SemiCoarsenedExtrudedHierarchy( \
                basemesh, 1.0, base_layer=args.mz,
                refinement_ratio=args.refinefactor, nref=args.refine)
for j in range(args.refine + 1):
    Q1R = FunctionSpace(hierarchy[j], 'P', 1, vfamily='R', vdegree=0)
    s = Function(Q1R)
    s.dat.data[:] = sbase.dat.data_ro[:]
    Vcoord = hierarchy[j].coordinates.function_space()
    x, z = SpatialCoordinate(hierarchy[j])
    XZ = Function(Vcoord).interpolate(as_vector([x, s * z]))
    hierarchy[j].coordinates.assign(XZ)

# solve the problem for each level in the hierarchy
upcoarse = None
levels = args.refine + 1
jrange = [levels - 1,] if args.single else range(levels)
for j in jrange:
    mesh = hierarchy[j]
    V = VectorFunctionSpace(mesh, 'Lagrange', 2)
    W = FunctionSpace(mesh, 'Lagrange', 1)
    Z = V * W
    up = Function(Z)
    scu, p = split(up)             # scaled velocity, unscaled pressure
    v, q = TestFunctions(Z)

    # use a more generous eps except when we get to the finest level
    if args.single or j == levels - 1:
        eps = args.eps
    else:
        eps = 100.0 * args.eps

    # symmetrically rescale the equations for better conditioning
    Du2 = 0.5 * inner(D(scu * sc), D(scu * sc)) + (eps * Dtyp)**2.0
    nu = 0.5 * B3 * Du2**((1.0 / n - 1.0)/2.0)
    F = ( sc*sc * inner(2.0 * nu * D(scu), D(v)) \
          - sc * p * div(v) - sc * q * div(scu) \
          - sc * inner(fbody, v) ) * dx

    # different boundary conditions relative to stage2/:
    #   base label is 'bottom', and we add noslip condition on degenerate ends
    bcs = [ DirichletBC(Z.sub(0), Constant((0.0, 0.0)), 'bottom'),
            DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (1,2)) ]

    # get initial condition by coarsening previous level
    if upcoarse is not None:
        prolong(upcoarse, up)

    printpar('solving on level %d (%d x %d mesh) ...' \
             % (j, args.mx, args.mz * (args.refinefactor)**j))
    n_u, n_p = V.dim(), W.dim()
    printpar('  sizes: n_u = %d, n_p = %d' % (n_u,n_p))
    solve(F == 0, up, bcs=bcs, options_prefix='s', solver_parameters=par)
    if upcoarse is None:
        upcoarse = up.copy()

    # print average and maximum velocity
    scu, _ = up.split()
    u = scu * sc
    P1 = FunctionSpace(mesh, 'CG', 1)
    one = Constant(1.0, domain=mesh)
    area = assemble(dot(one,one) * dx)
    umagav = assemble(sqrt(dot(u, u)) * dx) / area
    umag = interpolate(sqrt(dot(u, u)), P1)
    with umag.dat.vec_ro as vumag:
        umagmax = vumag.max()[1]
    printpar('  ice speed (m a-1): av = %.3f, max = %.3f' \
             % (umagav * secpera, umagmax * secpera))

# generate tensor-valued deviatoric stress tau, and effective viscosity nu,
#   from the velocity solution
def stresses(mesh, u):
    Du2 = 0.5 * inner(D(u), D(u)) + (args.eps * Dtyp)**2.0
    Q1 = FunctionSpace(mesh,'Q',1)
    TQ1 = TensorFunctionSpace(mesh, 'Q', 1)
    nu = Function(Q1).interpolate(0.5 * B3 * Du2**((1.0 / n - 1.0)/2.0))
    nu.rename('effective viscosity (Pa s)')
    tau = Function(TQ1).interpolate(2.0 * nu * D(u))
    tau /= 1.0e5
    tau.rename('tau (bar)')
    return tau, nu

printpar('saving u,p,tau,nu,rank to %s ...' % args.o)
u, p = up.split()
u *= sc
tau, nu = stresses(hierarchy[-1], u)
u *= secpera
p /= 1.0e5
u.rename('velocity (m/a)')
p.rename('pressure (bar)')
# integer-valued element-wise process rank
rank = Function(FunctionSpace(mesh,'DG',0))
rank.dat.data[:] = mesh.comm.rank
rank.rename('rank')
File(args.o).write(scu, p, tau, nu, rank)
