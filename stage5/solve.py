#!/usr/bin/env python3

# FIXME for doc
"""# boundary conditions in default (VI version):
#   * <u,w>=0 on 'bottom'
#   * <u,w>=0 on lateral ends
#   * pinch zero-height columns (trivialize all u,p d.o.f.s if s<stol)
# boundary conditions for -walls (assumes s>b):
#   * <u,w>=0 on 'bottom'
#   * u=0 on lateral ends (and no tangential stress)
"""

import argparse
import sys

parser = argparse.ArgumentParser(
    description="""stage6/  Solve the coupled surface kinematical (free-surface) equation and Glen-Stokes momentum equations for a 2D ice sheet using an extruded mesh.  Uses first-order explicit time-stepping and the Swedish stabilization.  Initial shape is from the Halfar solution.""",
    add_help=False,
)
hs = "time step in years (default=1.0)"
parser.add_argument("-dt", type=float, metavar="DT", default=1.0, help=hs)
hs = "regularization used in viscosity (default=10^{-4})"
parser.add_argument("-eps", type=float, metavar="X", default=1.0e-4, help=hs)
hs = "subintervals in coarse mesh (default=50)"
parser.add_argument("-mx", type=int, metavar="MX", default=50, help=hs)
hs = "vertical layers in coarse mesh (default=4)"
parser.add_argument("-mz", type=int, metavar="MZ", default=4, help=hs)
hs = "number of time steps (default=5)"
parser.add_argument("-N", type=int, metavar="N", default=5, help=hs)
hs = "turn off stabilizations"
parser.add_argument("-noswede", action="store_true", default=False, help=hs)
hs = "output filename (default=dome.pvd)"
parser.add_argument("-o", metavar="FILE.pvd", default="dome.pvd", help=hs)
hs = "print help for solve.py options and stop"
parser.add_argument("-solvehelp", action="store_true", default=False, help=hs)
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
dtsec = args.dt * secpera


def set_halfar_profile(x, s, R0=10000.0, H0=1000.0):
    # Set surface geometry from t = t0 Halfar time-dependent SIA geometry solution,
    # a dome with zero SMB.  The surface is zero outside of (-R0,R0).  Reference:
    #   * P. Halfar (1981), On the dynamics of the ice sheets, J. Geophys. Res. 86 (C11), 11065--11072
    pp = 1.0 + 1.0 / n
    rr = n / (2.0 * n + 1.0)
    xi = conditional(abs(x) < R0, 1.0 - abs(x / R0) ** pp, 0.0)
    s.interpolate(H0 * xi**rr)


# create 1D base mesh
if args.walls:
    L = 8000.0
else:
    L = 15000.0
basemesh = IntervalMesh(args.mx, -L, L)
xbase = SpatialCoordinate(basemesh)
P1base = FunctionSpace(basemesh, "Lagrange", 1)

# set initial 2D mesh geometry, but keep flat coordinates too
s = Function(P1base)
set_halfar_profile(xbase[0], s)
mesh = ExtrudedMesh(basemesh, layers=args.mz, layer_height=1.0 / args.mz)
xzflat = set_mesh_geometry(mesh, s)

# surface mass balance function; a=0 in Halfar solution
a = Function(P1base).interpolate(0.0)

# copies of these fields on the extruded mesh (R space) are used in stabilization
if not args.noswede:
    sR = extend_from_p1base(mesh, s)
    aR = extend_from_p1base(mesh, a)

# mixed spaces for Stokes
V = VectorFunctionSpace(mesh, "Lagrange", 2)
W = FunctionSpace(mesh, "Lagrange", 1)
# alternatives for local conservation in Stokes:  W = FunctionSpace(mesh, "DQ", 0|1)
Z = V * W
up = Function(Z)
v, q = TestFunctions(Z)

# strain-rate tensor
def D(w):
    return 0.5 * (grad(w) + grad(w).T)

def form_and_bcs_stokes(up):
    """Weak form for the Stokes problem on the geometry stored in the current
    mesh.  This must be called every time the geometry is re-set."""
    u, p = split(up)
    Du2 = 0.5 * inner(D(u), D(u)) + (args.eps * Dtyp) ** 2.0
    nu = 0.5 * B3 * Du2 ** ((1.0 / n - 1.0) / 2.0)
    F = (inner(2.0 * nu * D(u), D(v)) - p * div(v) - q * div(u)) * dx(degree=3)
    if not args.noswede:  # formula (4.28) in Tominec et al 2026
        nsnorm = sqrt(sR.dx(0) ** 2 + 1.0)
        nvec = FacetNormal(mesh)
        F += (rho * g * dtsec / 2.0) * nsnorm * inner(u, nvec) * inner(v, nvec) * ds_t
        F += (rho * g * dtsec) * aR * inner(v, nvec) * ds_t
    f_body = as_vector([0.0, -rho * g])
    F -= inner(f_body, v) * dx  # source term
    if args.walls:
        bcs = [
            DirichletBC(Z.sub(0), Constant((0.0, 0.0)), "bottom"),
            DirichletBC(Z.sub(0).sub(0), Constant(0.0), (1, 2)),
        ]
    else:
        bcs = [
            DirichletBC(Z.sub(0), Constant((0.0, 0.0)), "bottom"),
            DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (1, 2)),
            PinchColumnPressure(Z.sub(1), None, None),
            PinchColumnVelocity(Z.sub(0), None, None),
        ]
    return F, bcs

# parameters for Stokes solver
stokespar = {
    "snes_converged_reason": None,
    # "snes_monitor": None,
    "snes_linesearch_type": "bt",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_shift_type": "inblocks",
    "pc_factor_mat_solver_type": "mumps",
}

# parameters for SKE VI solver
vipar = {  # "snes_monitor": None,
    "snes_converged_reason": None,
    # "snes_rtol": 1.0e-6,
    # "snes_atol": 1.0e-14,
    "snes_stol": 0.0,
    "snes_type": "vinewtonrsls",
    "snes_vi_zero_tolerance": 1.0e-8,
    "snes_linesearch_type": "basic",
    "snes_max_it": 200,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# weak form for surface kinematical equation
snew = Function(P1base)
omega = TestFunction(P1base)
VP2base = VectorFunctionSpace(basemesh, "CG", 2, dim=2)
uwsurf = Function(VP2base)
# basic surface equation
dsdt = (snew - s) / (args.dt * secpera)
ns = as_vector([-s.dx(0), Constant(1.0)])
Fske = (dsdt - dot(uwsurf, ns) - a) * omega * dx
if not args.noswede:
    # implicit edge stabilization, formula (4.3) in Tominec et al 2026
    hbase = CellDiameter(basemesh)
    nbase = FacetNormal(basemesh)
    h = (hbase("+") + hbase("-")) / 2
    velmag = sqrt(dot(uwsurf, uwsurf))
    gamma = 0.5 * h**2 * velmag
    Fske += gamma * jump(grad(snew), nbase) * jump(grad(omega), nbase) * dS

# set up surface kinematical equation solver, a variational inequality
probske = NonlinearVariationalProblem(Fske, snew, [])  # bcs=[] .. no flux at (1,2)
solverske = NonlinearVariationalSolver(
    probske, solver_parameters=vipar, options_prefix="ske"
)
lb = Function(P1base).interpolate(Constant(0.0))
ub = Function(P1base).interpolate(Constant(PETSc.INFINITY))


# ALTERNATE weak form for surface kinematical equation, for snewDG in DG0base
#   this is interesting because the characteristic function of a basemesh
#   element *is* in DG0base
DG0base = FunctionSpace(basemesh, "DG", 0)
snewDG = Function(DG0base)
omegaDG = TestFunction(DG0base)
# basic surface equation
dsdt = (snewDG - s) / (args.dt * secpera)
FskeDG = (dsdt - dot(uwsurf, ns) - a) * omegaDG * dx
# set up problem and solver (a variational inequality)
probskeDG = NonlinearVariationalProblem(FskeDG, snewDG, [])  # bcs=[] .. no flux at (1,2)
solverskeDG = NonlinearVariationalSolver(
    probskeDG, solver_parameters=vipar, options_prefix="skeDG"
)
lbDG = Function(DG0base).interpolate(Constant(0.0))
ubDG = Function(DG0base).interpolate(Constant(PETSc.INFINITY))


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
printpar(f"solving {args.N} steps (dt={args.dt:.3f}) on {args.mx} x {args.mz} mesh ...")
n_u, n_p = V.dim(), W.dim()
printpar(f"  sizes: n_u = {n_u}, n_p = {n_p}")
t = 0.0
for k in range(args.N):
    printpar(f"t={t / secpera:.3f} a (k={k}):")
    report_shape(t, s)

    # solve Stokes on current geometry (in current mesh), which requires
    # resetting form F and boundary conditions bcs
    if not args.noswede:
        # update s, a in R space (for stabilization)
        sR.dat.data[:] = s.dat.data_ro[:]
        aR.dat.data[:] = a.dat.data_ro[:]
    F, bcs = form_and_bcs_stokes(up)
    solve(F == 0, up, bcs=bcs, options_prefix="s", solver_parameters=stokespar)
    u, p = up.subfunctions

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

    # solve SKE for one semi-implicit Euler time-step
    trace_to_vector_p2base(basemesh, mesh, u, uwsurf)
    snew.interpolate(s)  # initial condition
    solverske.solve(bounds=(lb, ub))
    s.interpolate(snew)  # update surface elevation

    # also solve ALTERNATE SKE for explicit in DG
    snewDG.interpolate(snew)  # initial condition
    solverskeDG.solve(bounds=(lbDG, ubDG))

    # update mesh geometry (rescaling original geometry) and time
    set_mesh_geometry(mesh, s, xzorig=xzflat)
    t += dtsec

printpar(f"t={t / secpera:.3f} a (done):")
report_shape(t, s)


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

flatname = "flat-" + args.o
printpar(f"saving u,p,sR to flat file {flatname} (vert * 10^4) ...")
sR = extend_from_p1base(mesh, s)
sR.rename("snew (in R space)")
DG0R = FunctionSpace(mesh, "DG", 0, vfamily="R", vdegree=0)
sRDG = Function(DG0R, name="snewDG (in R space)")
sRDG.dat.data[:] = snewDG.dat.data_ro[:]
DG1R = FunctionSpace(mesh, "DG", 1, vfamily="R", vdegree=0)
diff_sR = Function(DG1R, name="snew - snewDG (in R space)").interpolate(sR - sRDG)
Vcoord = mesh.coordinates.function_space()
XZ = Function(Vcoord).interpolate(as_vector([xzflat[0], 1.0e4 * xzflat[1]]))
mesh.coordinates.assign(XZ)
VTKFile(flatname).write(u, p, sR, diff_sR)
