#!/usr/bin/env python3

import sys
import argparse
import numpy as np
from firedrake import *

# recover stage3/:
#     ./solve.py -refine 0 -marginheight 0.0 -direct

# KEY HINT ABOUT PERFORMANCE:  all of these 2-layer, eps=10^-2 runs in stage3/,
# with modest snes_rtol, are super cheap, and give at most 11 iterations and
# time 17 seconds on finest, and O(N) times
#     for MX in 500 1000 2000 4000 8000 16000; do tmpg -n 1 ./solve.py -s_snes_converged_reason -mx $MX -mz 2 -s_snes_rtol 1.0e-3 -s_snes_max_it 200 -eps 1.0e-2; done

# SO HERE IS A WAY FORWARD:
#    * use SemiCoarsenedExtrudedHierarchy()
#    * grid sequence with eps=1.0e-2,1.0e-3,1.0e-4,1.0e-4,...

# good settings:
#     ./solve.py -mx 20 -mz 2 -s_snes_converged_reason -s_ksp_converged_reason -s_snes_ksp_ew -s_snes_rtol 1.0e-2 -refine K

parser = argparse.ArgumentParser(description=
'''stage4/  Solve the Glen-Stokes momentum equations for a 2D ice sheet using
an extruded mesh and a Schur-multigrid solver.''', add_help=False)
parser.add_argument('-direct', action='store_true', default=False,
    help='use direct LU solver for each Newton step')
parser.add_argument('-eps', type=float, metavar='X', default=1.0e-4,
    help='regularization used in viscosity')
parser.add_argument('-marginheight', type=float, metavar='X', default=1.0,
    help='height of degeneration point at margin (default 1 m)')
parser.add_argument('-mx', type=int, metavar='MX', default=50,
    help='number of subintervals in coarse mesh')
parser.add_argument('-mz', type=int, metavar='MZ', default=8,
    help='number of subintervals in coarse mesh')
parser.add_argument('-refine', type=int, metavar='J', default=2,
    help='number of multigrid refinements when generating hierarchy')
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

printpar = PETSc.Sys.Print        # print once even in parallel
fmz = args.mz * 2**args.refine
printpar('generating %d-level mesh hierarchy ...' % (args.refine + 1))
R = 10000.0
H = 1000.0
base = IntervalMesh(args.mx, length_or_left=0.0, right=2.0*R)
xbase = base.coordinates.dat.data_ro
P1base = FunctionSpace(base,'P',1)
sbase = Function(P1base)
sbase.dat.data[:] = profile(xbase, R, H)

hierarchy = SemiCoarsenedExtrudedHierarchy( \
                base, 1.0, base_layer=args.mz,
                refinement_ratio=2, nref=args.refine)
for j in range(args.refine + 1):
    Q1R = FunctionSpace(hierarchy[j], 'P', 1, vfamily='R', vdegree=0)
    s = Function(Q1R)
    s.dat.data[:] = sbase.dat.data_ro[:]
    Vcoord = hierarchy[j].coordinates.function_space()
    x, z = SpatialCoordinate(hierarchy[j])
    XZ = Function(Vcoord).interpolate(as_vector([x, s * z]))
    hierarchy[j].coordinates.assign(XZ)

parcoarse = {'snes_linesearch_type': 'bt', 'ksp_type': 'preonly',
             'pc_type': 'lu', 'pc_factor_shift_type': 'inblocks'}
if args.direct:
    par = parcoarse
else:
    par = {'snes_linesearch_type': 'bt',
        'ksp_type': 'fgmres',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'pc_fieldsplit_schur_fact_type': 'lower',
        'pc_fieldsplit_schur_precondition': 'selfp',
        'fieldsplit_0_ksp_type': 'preonly',
        #'fieldsplit_0_pc_type': 'lu',
        'fieldsplit_0_pc_type': 'mg',
        #'fieldsplit_0_mg_levels_ksp_type': 'gmres',
        #'fieldsplit_0_mg_levels_ksp_max_it': 2,
        'fieldsplit_0_mg_levels_ksp_type': 'richardson',
        'fieldsplit_0_mg_levels_pc_type': 'bjacobi',
        'fieldsplit_0_mg_levels_sub_pc_type': 'ilu',
        #'fieldsplit_0_mg_levels_pc_type': 'lu',
        'fieldsplit_0_mg_coarse_ksp_type': 'preonly',
        'fieldsplit_0_mg_coarse_pc_type': 'lu',
        #'fieldsplit_1_ksp_rtol': 1.0e-3,
        #'fieldsplit_1_ksp_max_it': 5,
        'fieldsplit_1_ksp_type': 'gmres',
        #'fieldsplit_1_ksp_type': 'preonly',
        'fieldsplit_1_pc_type': 'jacobi'}

        # 'pc_fieldsplit_schur_factorization_type': 'lower',
        # 'pc_fieldsplit_schur_precondition': 'selfp',  # BAD FOR ACCURACY?
        # 'fieldsplit_0_ksp_type': 'preonly',
        # 'fieldsplit_0_pc_type': 'mg',
        # 'fieldsplit_0_mg_levels_ksp_type': 'richardson',
        # 'fieldsplit_0_mg_levels_pc_type': 'asm',
        # 'fieldsplit_0_mg_levels_sub_pc_type': 'ilu',           # sor?
        # #'fieldsplit_1_ksp_rtol': 1.0e-2, # see iterations with -s_fieldsplit_1_ksp_converged_reason
        # 'fieldsplit_1_ksp_max_it': 3,
        # 'fieldsplit_1_ksp_type': 'gmres',
        # 'fieldsplit_1_pc_type': 'jacobi',
        # 'fieldsplit_1_pc_jacobi_type': 'diagonal'}

secpera = 31556926.0    # seconds per year
g = 9.81                # m s-2
rho = 910.0             # kg m-3
n = 3.0
A3 = 3.1689e-24         # Pa-3 s-1;  EISMINT I value of ice softness
B3 = A3**(-1.0/3.0)     # Pa s(1/3);  ice hardness
Dtyp = 1.0 / secpera    # s-1

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
    su, p = split(up)
    v, q = TestFunctions(Z)

    def D(w):               # strain-rate tensor
        return 0.5 * (grad(w) + grad(w).T)

    fbody = Constant((0.0, - rho * g))
    lam = 1.0e7
    # FIXME adjust eps on coarse meshes
    Du2 = 0.5 * inner(D(su/lam), D(su/lam)) + (args.eps * Dtyp)**2.0
    nu = 0.5 * B3 * Du2**((1.0 / n - 1.0)/2.0)
    F = ( lam**(-2) * inner(2.0 * nu * D(su), D(v)) \
          - lam**(-1) * p * div(v) - lam**(-1) * q * div(su) \
          - lam**(-1) * inner(fbody, v) ) * dx

    # different boundary conditions relative to stage2/:
    #   base label is 'bottom', and we add noslip condition on degenerate ends
    bcs = [ DirichletBC(Z.sub(0), Constant((0.0, 0.0)), 'bottom'),
            DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (1,2)) ]

    printpar('solving on level %d (%d x %d mesh) ...' \
             % (j, args.mx, args.mz * 2**j))
    if upcoarse is not None:
        prolong(upcoarse, up)
    solve(F == 0, up, bcs=bcs, options_prefix='s',
          solver_parameters=parcoarse if j == 0 else par)
    if upcoarse is None:
        upcoarse = up.copy()

    # print average and maximum velocity
    su, _ = up.split()
    u = su / lam
    P1 = FunctionSpace(mesh, 'CG', 1)
    one = Constant(1.0, domain=mesh)
    area = assemble(dot(one,one) * dx)
    umagav = assemble(sqrt(dot(u, u)) * dx) / area
    umag = interpolate(sqrt(dot(u, u)), P1)
    with umag.dat.vec_ro as vumag:
        umagmax = vumag.max()[1]
    printpar('  ice speed (m a-1): av = %.3f, max = %.3f' \
             % (umagav * secpera, umagmax * secpera))

# generate regularized effective viscosity from the solution:
#   nu = (1/2) B_n X^((1/n)-1)
# where X = sqrt(|Du|^2 + eps^2 Dtyp^2)
def effectiveviscosity(mesh, u):
    P1 = FunctionSpace(mesh, 'Lagrange', 1)
    Du2 = 0.5 * inner(D(u), D(u)) + (args.eps * Dtyp)**2.0
    nu = interpolate(0.5 * B3 * Du2**((1.0 / n - 1.0)/2.0), P1)
    nu.rename('effective viscosity')
    return nu

printpar('saving to dome.pvd ...')
su, p = up.split()
su.rename('velocity')
su.dat.data[:] /= lam
p.rename('pressure')
File('dome.pvd').write(su, p, effectiveviscosity(hierarchy[-1],su))
