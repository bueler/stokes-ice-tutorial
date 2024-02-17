#!/usr/bin/env python3

import sys
import numpy as np
import argparse
from firedrake import *
from firedrake.petsc import PETSc

parser = argparse.ArgumentParser(description=
'''stage5/  Solve the Glen-Stokes momentum equations for a 3D ice sheet using
an extruded mesh and an optional bumpy bed.''', add_help=False)
parser.add_argument('-b0', type=float, metavar='X', default=500.0,
    help='scale of bed bumpiness (default=500 m)')
parser.add_argument('-baserefine', type=int, metavar='X', default=2,
    help='refinement levels in generating disk base mesh (default=2)')
parser.add_argument('-eps', type=float, metavar='X', default=1.0e-4,
    help='regularization used in viscosity (default=10^{-4})')
parser.add_argument('-marginheight', type=float, metavar='X', default=1.0,
    help='height of degeneration point at margin (default=1 m)')
parser.add_argument('-mz', type=int, metavar='MZ', default=2,
    help='vertical layers in coarse mesh (default=2)')
parser.add_argument('-o', metavar='FILE.pvd', type=str, default='dome.pvd',
    help='output filename (default=dome.pvd)')
parser.add_argument('-refine', type=int, metavar='X', default=1,
    help='vertical refinements for 3D mesh (default=1)')
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

def bed(x, y, R, b0):
    '''A smooth, bumpy bed of magnitude b0.'''
    rr = (x * x + y * y) / (R * R)
    return b0 * np.sin(4.0 * np.pi * (x + y) / R) * (1.0 - rr)

# level-independent information
secpera = 31556926.0    # seconds per year
g = 9.81                # m s-2
rho = 910.0             # kg m-3
n = 3.0
A3 = 3.1689e-24         # Pa-3 s-1;  EISMINT I value of ice softness
B3 = A3**(-1.0/3.0)     # Pa s(1/3);  ice hardness
Dtyp = 1.0 / secpera    # s-1
sc = 1.0e-7             # velocity scale constant for symmetric equation scaling
fbody = Constant((0.0, 0.0, - rho * g))
par = {'snes_linesearch_type': 'bt',
       'ksp_type': 'preonly',
       'pc_type': 'lu',
       'pc_factor_shift_type': 'inblocks',
       'pc_factor_mat_solver_type': 'mumps'}
printpar = PETSc.Sys.Print        # print once even in parallel

def D(w):               # strain-rate tensor
    return 0.5 * (grad(w) + grad(w).T)

printpar('generating disk mesh in base (map-plane) by %d refinements ...' \
         % args.baserefine)
basemesh = UnitDiskMesh(refinement_level=args.baserefine)
basemesh.coordinates.dat.data[:] *= R * (1.0 - 1.0e-10) # avoid degeneracy
belements, bnodes = basemesh.num_cells(), basemesh.num_vertices()
printpar('    (%s2D base mesh is disk with %d triangle elements and %d nodes)' \
         % ('on rank 0: ' if basemesh.comm.size > 1 else '',
            belements, bnodes))

printpar('generating %d-level mesh hierarchy ...' % (args.refine + 1))
hierarchy = SemiCoarsenedExtrudedHierarchy( \
                basemesh, 1.0, base_layer=args.mz,
                refinement_ratio=args.refinefactor, nref=args.refine)
xbase = basemesh.coordinates.dat.data_ro[:,0]
ybase = basemesh.coordinates.dat.data_ro[:,1]
P1base = FunctionSpace(basemesh,'P',1)
sbase = Function(P1base)
sbase.dat.data[:] = profile(xbase, ybase, R, H)
bbase = Function(P1base)               # initialized to zero
bbase.dat.data[:] = bed(xbase, ybase, R, args.b0)
for j in range(args.refine + 1):
    Q1R = FunctionSpace(hierarchy[j], 'P', 1, vfamily='R', vdegree=0)
    s = Function(Q1R)
    s.dat.data[:] = sbase.dat.data_ro[:]
    b = Function(Q1R)
    b.dat.data[:] = bbase.dat.data_ro[:]
    Vcoord = hierarchy[j].coordinates.function_space()
    x, y, z = SpatialCoordinate(hierarchy[j])
    XYZ = Function(Vcoord).interpolate(as_vector([x, y, (s - b) * z + b]))
    hierarchy[j].coordinates.assign(XYZ)
fmz = args.mz * args.refinefactor**args.refine
printpar('    (%sfine-level 3D mesh has %d prism elements and %d nodes)' \
         % ('on rank 0: ' if basemesh.comm.size > 1 else '',
            belements * fmz, bnodes * (fmz + 1)))

# solve the problem for each level in the hierarchy
# (essentially dimension-independent, and nearly same as stage4/)
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
          - sc * inner(fbody, v) ) * dx(degree=3)

    # different boundary conditions relative to stage2/:
    #   base label is 'bottom', and we add noslip condition on degenerate ends
    bcs = [ DirichletBC(Z.sub(0), Constant((0.0, 0.0, 0.0)), 'bottom'),
            DirichletBC(Z.sub(0), Constant((0.0, 0.0, 0.0)), (1,)) ]

    # get initial condition by coarsening previous level
    if upcoarse is not None:
        prolong(upcoarse, up)

    printpar('solving on level %d with %d vertical layers ...' \
             % (j, args.mz * args.refinefactor**j))
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
u = up.subfunctions[0]
p = up.subfunctions[1]
u *= sc
tau, nu = stresses(hierarchy[-1], u)
u *= secpera
p /= 1.0e5
u.rename('velocity (m/a)')
p.rename('pressure (bar)')
rank = Function(FunctionSpace(mesh,'DG',0))
rank.dat.data[:] = mesh.comm.rank
rank.rename('rank')
File(args.o).write(scu, p, tau, nu, rank)
