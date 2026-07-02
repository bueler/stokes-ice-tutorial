#!/usr/bin/env python3

# FIXME for doc
# stable default run, *with* mass conservation:
#   python3 solve.py -walls

# FIXME what does Tominec et al say about CFL?  seems to me one can exceed CFL=1 *if* all stabilizations are on

# FIXME for doc
# demonstrates need for CFL:
#   python3 solve.py -nocfl -mx 100 -omovie movie.pvd   # with CFL turned off (and dt=1 a)
# easily destabilizes without load stabilization; visualize slosh
#   python3 solve.py -walls -noload -maxN 1         # mild slosh is already wrong
#   python3 solve.py -walls -noload -maxN 1 -nocfl  # severe slosh without CFL
# harder to catch wiggles which happen without edge stabilization:
#   python3 solve.py -mx 400 -T 1          # reasonable dome shape
#   python3 solve.py -mx 400 -T 1 -noedge  # not reasonable; wiggles at edge

import argparse
import sys

parser = argparse.ArgumentParser(
    description="""stage5/  Solve the coupled surface kinematical (free-surface) equation and Glen-Stokes momentum equations for a 2D ice sheet using an extruded mesh.  Uses first-order mostly-explicit time-stepping based on the two Swedish stabilizations.  Initial shape is from the Halfar solution.""",
    add_help=False,
)
hs = "coefficient to use in CFL scheme for time-stepping (default=0.5)"
parser.add_argument("-cfl", type=float, metavar="CFL", default=0.5, help=hs)
hs = "regularization used in viscosity (default=10^{-4})"
parser.add_argument("-eps", type=float, metavar="EPS", default=1.0e-4, help=hs)
hs = "initial ice thickness in center (default=1000 m)"
parser.add_argument("-H0", type=float, metavar="H0", default=1000.0, help=hs)
hs = "maximum number of time steps (default=10000)"
parser.add_argument("-maxN", type=int, metavar="N", default=10000, help=hs)
hs = "maximum time step in years (default=1.0)"
parser.add_argument("-maxdt", type=float, metavar="DT", default=1.0, help=hs)
hs = "subintervals in coarse mesh (default=50)"
parser.add_argument("-mx", type=int, metavar="MX", default=50, help=hs)
hs = "vertical layers in coarse mesh (default=4)"
parser.add_argument("-mz", type=int, metavar="MZ", default=4, help=hs)
hs = "turn off CFL determination of time step (instead use options -maxdt, -T for time axis)"
parser.add_argument("-nocfl", action="store_true", default=False, help=hs)
hs = "turn off edge stabilization"
parser.add_argument("-noedge", action="store_true", default=False, help=hs)
hs = "turn off extrapolated-load (FSSA-type) stabilization"
parser.add_argument("-noload", action="store_true", default=False, help=hs)
hs = "output filename (default=dome.pvd)"
parser.add_argument("-o", metavar="FILE.pvd", default="dome.pvd", help=hs)
hs = "output filename for movie; setting this name turns it on"
parser.add_argument("-omovie", metavar="FILE.pvd", default=None, help=hs)
hs = "print help for solve.py options and stop"
parser.add_argument("-solvehelp", action="store_true", default=False, help=hs)
hs = "end of run time in years (default=5.0)"
parser.add_argument("-T", type=float, metavar="T", default=5.0, help=hs)
hs = "put in no-tangential-traction walls so s>b everywhere"
parser.add_argument("-walls", action="store_true", default=False, help=hs)
args, passthroughoptions = parser.parse_known_args()
if args.solvehelp:
    parser.print_help()
    sys.exit(0)

import petsc4py

petsc4py.init(passthroughoptions)

import numpy as np
from firedrake import *
from firedrake.petsc import PETSc
from geometry import (
    PinchColumnPressure,
    PinchColumnVelocity,
    set_mesh_geometry,
    trace_to_vector_p2base,
    extend_from_p1base,
)

printpar = PETSc.Sys.Print  # print once even in parallel

# physical constants
secpera = 31556926.0  # seconds per year
g = 9.81  # m s-2
rho = 910.0  # kg m-3
n = 3.0
A3 = 3.1689e-24  # Pa-3 s-1;  EISMINT I value of ice softness
B3 = A3 ** (-1.0 / 3.0)  # Pa s(1/3);  ice hardness
Dtyp = 1.0 / secpera  # s-1;  strain rate scale


def set_halfar_profile(x, s, R0=10000.0, H0=args.H0):
    # Set surface geometry from t = t0 Halfar time-dependent SIA geometry solution,
    # a dome with zero SMB.  The surface is zero outside of (-R0,R0).  Reference:
    #   * P. Halfar (1981), On the dynamics of the ice sheets, J. Geophys. Res. 86 (C11), 11065--11072
    pp = 1.0 + 1.0 / n
    rr = n / (2.0 * n + 1.0)
    xi = conditional(abs(x) < R0, 1.0 - abs(x / R0) ** pp, 0.0)
    s.interpolate(H0 * xi ** rr)


# create 1D base mesh
if args.walls:
    L = 8000.0  # here s>b initially (and later too) because L < R0
else:
    L = 15000.0  # initial margin locations at +-R0 in Halfar profile
basemesh = IntervalMesh(args.mx, -L, L)
xbase = SpatialCoordinate(basemesh)
P1base = FunctionSpace(basemesh, "Lagrange", 1)

# set initial 2D mesh geometry, but store flat coordinates
s = Function(P1base)
set_halfar_profile(xbase[0], s)
mesh = ExtrudedMesh(basemesh, layers=args.mz, layer_height=1.0 / args.mz)
xzflat = set_mesh_geometry(mesh, s)
if not args.noload:
    sR = extend_from_p1base(mesh, s)

# surface mass balance function; a=0 in Halfar solution
a = Function(P1base).interpolate(0.0)
if not args.noload:
    aR = extend_from_p1base(mesh, a)

# mixed spaces for Stokes
V = VectorFunctionSpace(mesh, "Lagrange", 2)
W = FunctionSpace(mesh, "Lagrange", 1)
Z = V * W
up = Function(Z)
v, q = TestFunctions(Z)

# parameters for Stokes solver
stokespar = {
    "snes_converged_reason": None,
    # "snes_monitor": None,
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_shift_type": "inblocks",
    "pc_factor_mat_solver_type": "mumps",
}

# parameters for SKE VI solver
vipar = {
    "snes_converged_reason": None,
    # "snes_monitor": None,
    "snes_stol": 0.0,
    "snes_type": "vinewtonrsls",
    "snes_linesearch_type": "basic",
    "snes_vi_zero_tolerance": 1.0e-8,
    "snes_max_it": 200,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# strain-rate tensor
def D(w):
    return 0.5 * (grad(w) + grad(w).T)


def form_stokes(mesh, up, loadstab=False, dt=None):
    """Weak form for the Stokes problem on the geometry stored in the current
    mesh.  This must be called every time the geometry is re-set."""
    u, p = split(up)
    Du2 = 0.5 * inner(D(u), D(u)) + (args.eps * Dtyp) ** 2.0
    nu = 0.5 * B3 * Du2 ** ((1.0 / n - 1.0) / 2.0)
    F = (inner(2.0 * nu * D(u), D(v)) - p * div(v) - q * div(u)) * dx(degree=3)
    # body force source term
    F -= inner(as_vector([0.0, -rho * g]), v) * dx
    if loadstab:
        # apply load (FSSA-type) coupling stabilization, (4.28) in Tominec et al 2026
        nn = FacetNormal(mesh)
        nsR = as_vector([-sR.dx(0), Constant(1.0)])
        F += rho * g * dt * (0.5 * inner(u, nsR) + aR) * inner(v, nn) * ds_t
    return F


def bcs_stokes(Z):
    """Boundary conditions in all cases, both -walls and free-boundary:
      Dirichlet: <u,w>=0 on 'bottom', u=0 on lateral ends (1,2)
      Neumann: stress free on 'top', zero tangential stress on (1,2)
    Added in default free-boundar (VI) case:
      pinch zero-height columns = trivialize all u,p dofs if s<stol"""
    bcs = [
        DirichletBC(Z.sub(0), Constant((0.0, 0.0)), "bottom"),
        DirichletBC(Z.sub(0).sub(0), Constant(0.0), (1, 2)),
    ]
    if not args.walls:
        bcs += [
            PinchColumnPressure(Z.sub(1), None, None),
            PinchColumnVelocity(Z.sub(0), None, None),
        ]
    return bcs


# functions for surface kinematical equation
snew = Function(P1base)
omega = TestFunction(P1base)
VP2base = VectorFunctionSpace(basemesh, "CG", 2, dim=2)
uwsurf = Function(VP2base)
lb = Function(P1base).interpolate(Constant(0.0))
ub = Function(P1base).interpolate(Constant(PETSc.INFINITY))

def form_ske(s, snew, dt=None):
    """weak form for surface kinematical equation"""
    ns = as_vector([-s.dx(0), Constant(1.0)])
    Fske = ((snew - s) / dt - dot(uwsurf, ns) - a) * omega * dx
    if not args.noedge:
        # implicit edge stabilization, formula (4.3) in Tominec et al 2026
        hbase = CellDiameter(basemesh)
        h = (hbase("+") + hbase("-")) / 2
        nbase = FacetNormal(basemesh)
        velmag = sqrt(dot(uwsurf, uwsurf))
        gamma = 0.5 * h ** 2 * velmag
        Fske += gamma * jump(grad(snew), nbase) * jump(grad(omega), nbase) * dS
    return Fske


def get_dt(t, dx, umagmax):
    """Determine dt from CFL and options."""
    if not args.nocfl:
        dtcfl = args.cfl * dx / umagmax
    else:
        dtcfl = PETSc.INFINITY
    return np.array([dtcfl, args.maxdt * secpera, args.T * secpera - t]).min()


def report_shape(t, s, stol=1.0):
    with s.dat.vec_ro as ss:
        smax = ss.max()[1]
    # following not parallel
    xx = basemesh.coordinates.dat.data_ro
    width = xx[s.dat.data_ro > stol].max() - xx[s.dat.data_ro > stol].min()
    area = assemble(s * dx)
    printpar(
        f"  width = {width / 1e3:.3f} km, max(s) = {smax:.3f} m, area = {area / 1e6:.3f} km^2"
    )


# time stepping loop
deltax = 2.0 * L / args.mx
printpar(f"solving on {args.mx} x {args.mz} mesh (dx = {deltax:.3f} m) ...")
n_u, n_p = V.dim(), W.dim()
printpar(f"  sizes: n_u = {n_u}, n_p = {n_p}")
if args.omovie is not None:
    printpar(f"opening {args.omovie} for writing u,p at each time step ...")
    movie = VTKFile(args.omovie)
t = 0.0
dtsec = None
for k in range(args.maxN):
    if t >= args.T * secpera:
        break
    printpar(f"t={t / secpera:.3f} a (k={k}):")
    report_shape(t, s)

    # solve Stokes on current geometry (in current mesh), which requires
    # resetting form F and boundary conditions bcs
    if not args.noload:
        # update s, a in R space (for stabilization)
        sR.dat.data[:] = s.dat.data_ro[:]
        aR.dat.data[:] = a.dat.data_ro[:]
    # FIXME at first step, the -noload stabilization needs dtsec to be set,
    #       presumably by CFL after a Stokes solve w/o the -noload mechanism
    if dtsec is None:
        F = form_stokes(mesh, up, loadstab=False)
    else:
        F = form_stokes(mesh, up, loadstab=(not args.noload), dt=dtsec)
    bcs = bcs_stokes(Z)  # why has to be re-set? (must use current s immediately?)
    solve(F == 0, up, bcs=bcs, options_prefix="s", solver_parameters=stokespar)
    u, p = up.subfunctions

    if args.omovie is not None:
        u.rename("velocity (m/s)")
        p.rename("pressure (Pa)")
        movie.write(u, p, time=t)

    # print average and maximum velocity
    R = FunctionSpace(mesh, "R", 0)
    one = Function(R).assign(1.0)
    area = assemble(dot(one, one) * dx)  # integrate 1 to get area of domain
    P1 = FunctionSpace(mesh, "CG", 1)
    umagav = assemble(sqrt(dot(u, u)) * dx) / area
    umag = Function(P1).interpolate(sqrt(dot(u, u)))
    with umag.dat.vec_ro as vumag:
        umagmax = vumag.max()[1]
    printpar(
        f"  ice speed (m a-1) at t={t / secpera:.3f} a: av = {umagav * secpera:.3f}, max = {umagmax * secpera:.3f}"
    )

    dtsec = get_dt(t, deltax, umagmax)
    printpar(f"  dt = {dtsec / secpera:.3f} a")

    # solve SKE for one semi-implicit Euler time-step, a variational inequality
    trace_to_vector_p2base(basemesh, mesh, u, uwsurf)
    snew.interpolate(s)  # initial condition
    Fske = form_ske(s, snew, dt=dtsec)
    probske = NonlinearVariationalProblem(Fske, snew, [])  # bcs=[] .. no flux at (1,2)
    solverske = NonlinearVariationalSolver(
        probske, solver_parameters=vipar, options_prefix="ske"
    )
    solverske.solve(bounds=(lb, ub))
    s.interpolate(snew)  # update surface elevation

    # update mesh geometry (rescaling original geometry) and time
    set_mesh_geometry(mesh, s, xzorig=xzflat)
    t += dtsec

printpar(f"done ... t={t / secpera:.3f} a")
report_shape(t, s)
if args.omovie is not None:
    printpar(f"done with file {args.omovie}")

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


printpar(f"saving u,p,tau,nu,rank to {args.o} ...")
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
