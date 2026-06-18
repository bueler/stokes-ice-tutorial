#!/usr/bin/env python3

import argparse
import sys

parser = argparse.ArgumentParser(
    description="""stage6/  Solve the coupled surface kinematical equation and Glen-Stokes momentum equations for a 2D ice sheet using an extruded mesh.  Does explicit (FIXME but Swedish stabilized) time-stepping from initial Halfar dome shape.""",
    add_help=False,
)
hs = "time step in years (default=1.0)"
parser.add_argument("-dt", type=float, metavar="DT", default=1.0, help=hs)
hs = "regularization used in viscosity (default=10^{-4})"
parser.add_argument("-eps", type=float, metavar="X", default=1.0e-4, help=hs)
hs = "subintervals in coarse mesh (default=50)"
parser.add_argument("-mx", type=int, metavar="MX", default=50, help=hs)
hs = "vertical layers in coarse mesh (default=2)"
parser.add_argument("-mz", type=int, metavar="MZ", default=2, help=hs)
hs = "number of time steps (default=50)"
parser.add_argument("-N", type=int, metavar="N", default=5, help=hs)
hs = "output filename (default=dome.pvd)"
parser.add_argument("-o", metavar="FILE.pvd", default="dome.pvd", help=hs)
hs = "print help for solve.py options and stop"
parser.add_argument("-solvehelp", action="store_true", default=False, help=hs)
args, passthroughoptions = parser.parse_known_args()
if args.solvehelp:
    parser.print_help()
    sys.exit(0)

import petsc4py

petsc4py.init(passthroughoptions)
import numpy as np
from functools import cached_property
from firedrake import *
from firedrake.dmhooks import get_appctx, pop_appctx, push_appctx
from firedrake.petsc import PETSc

printpar = PETSc.Sys.Print  # print once even in parallel

# physical constants
secpera = 31556926.0  # seconds per year
g = 9.81  # m s-2
rho = 910.0  # kg m-3
n = 3.0
A3 = 3.1689e-24  # Pa-3 s-1;  EISMINT I value of ice softness
B3 = A3 ** (-1.0 / 3.0)  # Pa s(1/3);  ice hardness
Dtyp = 1.0 / secpera  # s-1;  strain rate scale

# geometry
R0 = 10000.0
H0 = 1000.0
L = 15000.0  # R0 < L


def set_profile(x, s):
    # set surface geometry from t = t0 Halfar time-dependent SIA geometry solution,
    # a dome with zero SMB; reference:
    #   * P. Halfar (1981), On the dynamics of the ice sheets, J. Geophys. Res. 86 (C11), 11065--11072
    pp = 1.0 + 1.0 / n
    rr = n / (2.0 * n + 1.0)
    xi = conditional(abs(x) < R0, 1.0 - abs(x / R0) ** pp, 0.0)
    s.interpolate(H0 * xi ** rr)

# create 1D base mesh and 2D mesh
basemesh = IntervalMesh(args.mx, -L, L)
xbase = SpatialCoordinate(basemesh)
P1base = FunctionSpace(basemesh, "P", 1)
mesh = ExtrudedMesh(basemesh, layers=args.mz, layer_height=1.0/args.mz)


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
    # push application context onto the DM attached to the coordinate function space
    actx = {"sR": sR}
    dm = mesh.coordinates.function_space().dm
    _ = pop_appctx(dm)
    push_appctx(dm, actx)
    return xzorig


# set initial geometry, but keep flat coordinates too
s = Function(P1base)
set_profile(xbase[0], s)
mesh = ExtrudedMesh(basemesh, layers=args.mz, layer_height=1.0/args.mz)
xzflat = setmeshgeometry(mesh, s)


class PinchColumnPressure(DirichletBC):

    def __init__(self, V, g, sub_domain, htol=1.0):
        self.htol = htol
        super().__init__(V, Constant(0.0), None)

    @cached_property
    def nodes(self):
        V = self.function_space()
        mesh = V.mesh()
        # get application ctx from coordinates DM
        actx = get_appctx(mesh.coordinates.function_space().dm)
        assert actx is not None, f"got None for appctx from {mesh} coordinates DM"
        # return P1 nodes in columns with surface elevation less than htol
        s = Function(V).interpolate(actx["sR"])
        return np.where(s.dat.data_ro_with_halos < self.htol)[0]


class PinchColumnVelocity(DirichletBC):
    """This 'pinched column' is for vector-valued velocity.  Compare PinchColumnPressure."""

    def __init__(self, V, g, sub_domain, htol=1.0):
        self.htol = htol
        super().__init__(V, as_vector([0.0, 0.0]), None)

    @cached_property
    def nodes(self):
        V = self.function_space()
        mesh = V.mesh()
        # get application ctx from coordinates DM
        actx = get_appctx(mesh.coordinates.function_space().dm)
        assert actx is not None, f"got None for appctx from {mesh} coordinates DM"
        # return vector P2 nodes in columns with height (thickness) less than htol
        # warning: assumes velocity space is P2
        P2scalar = FunctionSpace(V.mesh(), "CG", 2)
        s = Function(P2scalar).interpolate(actx["sR"])
        ss = Function(V).interpolate(as_vector([s, s]))
        return np.where(ss.dat.data_ro_with_halos < self.htol)[0]


# mixed spaces for Stokes
V = VectorFunctionSpace(mesh, "Lagrange", 2)
W = FunctionSpace(mesh, "Lagrange", 1)
Z = V * W
up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)

def D(w):  # strain-rate tensor
    return 0.5 * (grad(w) + grad(w).T)

# weak form for Stokes problem (on given geometry)
eps = args.eps
fbody = Constant((0.0, -rho * g))
Du2 = 0.5 * inner(D(u), D(u)) + (eps * Dtyp) ** 2.0
nu = 0.5 * B3 * Du2 ** ((1.0 / n - 1.0) / 2.0)
F = (inner(2.0 * nu * D(u), D(v)) - p * div(v) - q * div(u) - inner(fbody, v)) * dx(
    degree=3
)

# boundary conditions: noslip on 'bottom' and degenerate ends, pinch zero-height columns
bcs = [
    DirichletBC(Z.sub(0), Constant((0.0, 0.0)), "bottom"),
    DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (1, 2)),
    PinchColumnPressure(Z.sub(1), None, None),
    PinchColumnVelocity(Z.sub(0), None, None),
]


# set up for weak form and solver
par = {
    "snes_converged_reason": None,
    #"snes_monitor": None,
    "snes_linesearch_type": "bt",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_shift_type": "inblocks",
    "pc_factor_mat_solver_type": "mumps",
}


# this is from https://github.com/bueler/stokes-extrude
def trace_scalar_to_p1(basemesh, mesh, f, nointerpolate=False):
    """On an extruded mesh, compute the trace of any scalar function f
    along the top surface boundary at the P1 nodes.
    Set nointerpolate=True if f is already P1.  Returns a P1 function on
    basemesh."""
    P1 = FunctionSpace(mesh, "CG", 1)
    if nointerpolate:
        fP1 = f
    else:
        fP1 = Function(P1).interpolate(f)
    bc = DirichletBC(P1, 0.0, "top")
    P1basemesh = FunctionSpace(basemesh, "CG", 1)
    fbm = Function(P1basemesh)
    fbm.dat.data_with_halos[:] = fP1.dat.data_with_halos[bc.nodes]
    return fbm


# weak form for surface kinematical equation; FIXME should be VI
snew = Function(P1base).interpolate(s)
omega = TestFunction(P1base)
a = Function(P1base).interpolate(0.0)  # Halfar solution corresponds to a=0
usurf = Function(P1base)
wsurf = Function(P1base)
dsdt = (snew - s) / (args.dt * secpera)
Fske = (dsdt + usurf * s.dx(0) - wsurf - a) * omega * dx
# FIXME: implicit edge stabilization
bcske = [ DirichletBC(P1base, Constant(0.0), (1, 2)), ]

# time stepping loop
printpar(
    "solving %d steps (dt=%.3f) on %d x %d mesh ..."
    % (args.N, args.dt, args.mx, args.mz)
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

    # solve SKE for one semi-implicit Euler time-step  FIXME
    usurf.interpolate(trace_scalar_to_p1(basemesh, mesh, u[0]))
    wsurf.interpolate(trace_scalar_to_p1(basemesh, mesh, u[1]))
    solve(Fske == 0, snew, bcs=bcske, options_prefix="ske", solver_parameters=par)
    s.dat.data[:] = np.maximum(0.0, snew.dat.data_ro[:])
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
tau, nu = stresses(mesh, u)
u *= secpera
p /= 1.0e5
u.rename("velocity (m/a)")
p.rename("pressure (bar)")
# integer-valued element-wise process rank
rank = Function(FunctionSpace(mesh, "DG", 0))
rank.dat.data[:] = mesh.comm.rank
rank.rename("rank")
VTKFile(args.o).write(u, p, tau, nu, rank)
