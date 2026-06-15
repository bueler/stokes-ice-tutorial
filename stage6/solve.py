#!/usr/bin/env python3

import argparse
import sys

parser = argparse.ArgumentParser(
    description="""stage6/  Solve the coupled surface kinematical equation and Glen-Stokes
momentum equations for a 2D ice sheet using an extruded mesh and rescaled
equations.  Does explicit (FIXME but Swedish stabilized) time-stepping from initial
dome shape.""",
    add_help=False,
)
parser.add_argument(
    "-dt",
    type=float,
    metavar="DT",
    default=1.0,
    help="time step in years (default=1.0)",
)
parser.add_argument(
    "-eps",
    type=float,
    metavar="X",
    default=1.0e-4,
    help="regularization used in viscosity (default=10^{-4})",
)
parser.add_argument(
    "-marginheight",
    type=float,
    metavar="X",
    default=1.0,
    help="height of degeneration point at margin (default=1 m)",
)
parser.add_argument(
    "-mx",
    type=int,
    metavar="MX",
    default=50,
    help="subintervals in coarse mesh (default=50)",
)
parser.add_argument(
    "-mz",
    type=int,
    metavar="MZ",
    default=2,
    help="vertical layers in coarse mesh (default=2)",
)
parser.add_argument(
    "-N", type=int, metavar="N", default=5, help="number of time steps (default=50)"
)
parser.add_argument(
    "-o",
    metavar="FILE.pvd",
    type=str,
    default="dome.pvd",
    help="output filename (default=dome.pvd)",
)
parser.add_argument(
    "-refine",
    type=int,
    metavar="X",
    default=1,
    help="vertical refinements when generating mesh hierarchy (default=1)",
)
parser.add_argument(
    "-refinefactor",
    type=int,
    metavar="X",
    default=4,
    help="vertical refinement factor when generating mesh hierarchy (default=4)",
)
parser.add_argument(
    "-solvehelp",
    action="store_true",
    default=False,
    help="print help for solve.py options and stop",
)
args, passthroughoptions = parser.parse_known_args()
if args.solvehelp:
    parser.print_help()
    sys.exit(0)

import petsc4py

petsc4py.init(passthroughoptions)
import numpy as np
from firedrake import *
from firedrake.petsc import PETSc


def profile(x, R, H):
    """Exact SIA solution for surface elevation, with half-length (radius) R
    and maximum height H, on interval [0,L] = [0,2R], centered at x=R.
    See van der Veen (2013) equation (5.50)."""
    n = 3.0  # glen exponent
    p1 = n / (2.0 * n + 2.0)  # = 3/8
    q1 = 1.0 + 1.0 / n  # = 4/3
    Z = H / (n - 1.0) ** p1  # outer constant
    X = (x - R) / R  # rescaled coord
    Xin = abs(X[abs(X) < 1.0])  # rescaled distance from center
    Yin = 1.0 - Xin
    s = np.zeros(np.shape(x))
    s[abs(X) < 1.0] = Z * ((n + 1.0) * Xin - 1.0 + n * Yin ** q1 - n * Xin ** q1) ** p1
    s[s < 1.0] = args.marginheight  # needed so that prolong() can find nodes
    return s


# level-independent information
secpera = 31556926.0  # seconds per year
g = 9.81  # m s-2
rho = 910.0  # kg m-3
n = 3.0
A3 = 3.1689e-24  # Pa-3 s-1;  EISMINT I value of ice softness
B3 = A3 ** (-1.0 / 3.0)  # Pa s(1/3);  ice hardness
Dtyp = 1.0 / secpera  # s-1
fbody = Constant((0.0, -rho * g))
par = {
    "snes_converged_reason": None,
    "snes_monitor": None,
    "snes_linesearch_type": "bt",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_shift_type": "inblocks",
    "pc_factor_mat_solver_type": "mumps",
}
printpar = PETSc.Sys.Print  # print once even in parallel


def D(w):  # strain-rate tensor
    return 0.5 * (grad(w) + grad(w).T)


printpar("generating %d-level mesh hierarchy ..." % (args.refine + 1))
R = 10000.0
H = 1000.0
basemesh = IntervalMesh(args.mx, length_or_left=0.0, right=2.0 * R)
xbase = basemesh.coordinates.dat.data_ro
P1base = FunctionSpace(basemesh, "P", 1)

s = Function(P1base)
s.dat.data[:] = profile(xbase, R, H)

hierarchy = SemiCoarsenedExtrudedHierarchy(
    basemesh,
    1.0,
    base_layer=args.mz,
    refinement_ratio=args.refinefactor,
    nref=args.refine,
)


def setmeshgeometry(mesh, s, xzorig=None):
    Q1R = FunctionSpace(mesh, "P", 1, vfamily="R", vdegree=0)
    sR = Function(Q1R)
    sR.dat.data[:] = s.dat.data_ro[:]
    Vcoord = mesh.coordinates.function_space()
    if xzorig is None:
        xz = SpatialCoordinate(mesh)
        xzorig = Function(Vcoord).interpolate(xz)
    else:
        xz = Function(Vcoord).interpolate(xzorig)
    XZ = Function(Vcoord).interpolate(as_vector([xz[0], sR * xz[1]]))
    mesh.coordinates.assign(XZ)
    return xzorig


# set geometry on all levels; xflat,zflat will have finest level
for j in range(args.refine + 1):
    xzflat = setmeshgeometry(hierarchy[j], s)

# use finest mesh in hierarchy
mesh = hierarchy[-1]

# weak form for Stokes problem (on given geometry)
V = VectorFunctionSpace(mesh, "Lagrange", 2)
W = FunctionSpace(mesh, "Lagrange", 1)
Z = V * W
up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)
eps = args.eps
Du2 = 0.5 * inner(D(u), D(u)) + (eps * Dtyp) ** 2.0
nu = 0.5 * B3 * Du2 ** ((1.0 / n - 1.0) / 2.0)
F = (inner(2.0 * nu * D(u), D(v)) - p * div(v) - q * div(u) - inner(fbody, v)) * dx(
    degree=3
)

# boundary conditions: noslip on 'bottom' and degenerate ends
bcs = [
    DirichletBC(Z.sub(0), Constant((0.0, 0.0)), "bottom"),
    DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (1, 2)),
]

# weak form for surface kinematical equation
snew = Function(P1base).interpolate(s)
omega = TestFunction(P1base)
a = Function(P1base).interpolate((1.0 / secpera))  # FIXME
dsdt = (snew - s) / (args.dt * secpera)
# FIXME: surface trace of velocity; implicit edge stabilization
Fske = (dsdt - a) * w * dx
bcske = [ DirichletBC(P1base, Constant(0.0), (1, 2)), ]

# time stepping loop
printpar(
    "solving %d steps (dt=%.3f) on %d x %d mesh ..."
    % (args.N, args.dt, args.mx, args.mz * (args.refinefactor) ** j)
)
n_u, n_p = V.dim(), W.dim()
printpar("  sizes: n_u = %d, n_p = %d" % (n_u, n_p))
t = 0.0
for k in range(args.N):
    # solve Stokes on current geometry
    solve(F == 0, up, bcs=bcs, options_prefix="s", solver_parameters=par)
    u, p = up.subfunctions

    # integrate 1 to get area of domain
    R = FunctionSpace(mesh, "R", 0)
    one = Function(R).assign(1.0)
    area = assemble(dot(one, one) * dx)

    # print geometry measure
    with s.dat.vec_ro as height:
        smax = height.max()[1]
    printpar("  maximum surface height at t=%.3f a: %.3f" % (t / secpera, smax))

    # print average and maximum velocity
    P1 = FunctionSpace(mesh, "CG", 1)
    umagav = assemble(sqrt(dot(u, u)) * dx) / area
    umag = Function(P1).interpolate(sqrt(dot(u, u)))
    with umag.dat.vec_ro as vumag:
        umagmax = vumag.max()[1]
    printpar(
        "  ice speed (m a-1) at t=%.3f a: av = %.3f, max = %.3f"
        % (t / secpera, umagav * secpera, umagmax * secpera)
    )

    # solve SKE for one semi-implicit Euler time-step
    solve(Fske == 0, snew, bcs=bcske, options_prefix="ske", solver_parameters=par)
    s.dat.data[:] = snew.dat.data_ro[:]
    setmeshgeometry(mesh, s, xzorig=xzflat)

    t += args.dt * secpera

# print geometry measure at final time
with s.dat.vec_ro as height:
    smax = height.max()[1]
printpar("  maximum surface height at t=%.3f a: %.3f" % (t / secpera, smax))

# generate tensor-valued deviatoric stress tau, and effective viscosity nu,
#   from the velocity solution
def stresses(mesh, u):
    Du2 = 0.5 * inner(D(u), D(u)) + (args.eps * Dtyp) ** 2.0
    Q1 = FunctionSpace(mesh, "Q", 1)
    TQ1 = TensorFunctionSpace(mesh, "Q", 1)
    nu = Function(Q1).interpolate(0.5 * B3 * Du2 ** ((1.0 / n - 1.0) / 2.0))
    nu.rename("effective viscosity (Pa s)")
    tau = Function(TQ1).interpolate(2.0 * nu * D(u))
    tau /= 1.0e5
    tau.rename("tau (bar)")
    return tau, nu


printpar("saving u,p,tau,nu,rank to %s ..." % args.o)
u, p = up.subfunctions
tau, nu = stresses(hierarchy[-1], u)
u *= secpera
p /= 1.0e5
u.rename("velocity (m/a)")
p.rename("pressure (bar)")
# integer-valued element-wise process rank
rank = Function(FunctionSpace(mesh, "DG", 0))
rank.dat.data[:] = mesh.comm.rank
rank.rename("rank")
VTKFile(args.o).write(u, p, tau, nu, rank)
