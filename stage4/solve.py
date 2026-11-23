# get options help:  python3 solve.py -solvehelp

import argparse
import sys

parser = argparse.ArgumentParser(
    description="""stage4/  Solve the coupled free-surface (kinematical) equation and power-law Stokes momentum equations for a 2D ice sheet.  Initial dome shape is from the Halfar solution.  Uses an extruded mesh of quadrilaterals, and defaults to Q2 x DQ1 for (u,p) in the Stokes equations.  Uses first-order, mostly-explicit, adaptive time-stepping based on the Swedish stabilizations (see slides), supplemented by a CFL condition motivated by margin-advance considerations.  The free-surface update is by a variational inequality (free-boundary) method.""",
    add_help=False,
)
hs = "-hdiv only: interior penalty constant (default=10)"
parser.add_argument("-alpha", type=float, metavar="A", default=10.0, help=hs)
hs = "coefficient to use in CFL scheme for time-stepping (default=0.25)"
parser.add_argument("-cfl", type=float, metavar="CFL", default=0.25, help=hs)
hs = "regularization used in viscosity (default=1.0e-4)"
parser.add_argument("-eps", type=float, metavar="EPS", default=1.0e-4, help=hs)
hs = "initial ice thickness in center of dome (default=1000 m)"
parser.add_argument("-H0", type=float, metavar="H0", default=1000.0, help=hs)
hs = "half-width of computational domain (default=15000 m)"
parser.add_argument("-L", type=float, metavar="L", default=15000.0, help=hs)
hs = "use H(div) velocity (RTCF2) and DQ1 pressure, instead of Q2 x (-pressure)"
parser.add_argument("-hdiv", action="store_true", default=False, help=hs)
hs = "maximum number of time steps (default=10000)"
parser.add_argument("-maxN", type=int, metavar="N", default=10000, help=hs)
hs = "maximum time step in years (set DT > 0 to engage)"
parser.add_argument("-maxdt", type=float, metavar="DT", default=-1.0, help=hs)
hs = "subintervals in coarse mesh (default=50)"
parser.add_argument("-mx", type=int, metavar="MX", default=50, help=hs)
hs = "vertical layers in coarse mesh (default=4)"
parser.add_argument("-mz", type=int, metavar="MZ", default=4, help=hs)
hs = "turn off CFL determination of time step; instead -maxdt, -T determine time axis"
parser.add_argument("-nocfl", action="store_true", default=False, help=hs)
hs = "turn off edge stabilization"
parser.add_argument("-noedge", action="store_true", default=False, help=hs)
hs = "turn off extrapolated-load (FSSA-type) stabilization"
parser.add_argument("-noload", action="store_true", default=False, help=hs)
hs = "output filename (default=dome.pvd)"
parser.add_argument("-o", metavar="FILE.pvd", default="dome.pvd", help=hs)
hs = "output filename for movie; provide .pvd name to turn movie on"
parser.add_argument("-omovie", metavar="FILE.pvd", default=None, help=hs)
hs = "output filename for ice area time series; provide name to turn on"
parser.add_argument("-ots", metavar="FILE.txt", default=None, help=hs)
hs = "element type to use for pressure: DQ1|DG0|Q1 (default=DQ1)"
pcho = ["DQ1", "DG0", "Q1"]
parser.add_argument("-pressure", metavar="X", choices=pcho, default="DQ1", help=hs)
hs = "initial half-width of dome (default=10000 m)"
parser.add_argument("-R0", type=float, metavar="R0", default=10000.0, help=hs)
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
from firedrake.exceptions import ConvergenceError

printpar = PETSc.Sys.Print  # print once even in parallel

from halfar import (
    secpera,
    g,
    rho,
    n,
    A3,
    B3,
    get_halfar_characteristic_time,
    get_halfar_dimensions_from_time,
    set_halfar_from_time,
    set_halfar_from_lengths,
)
from geometry import (
    PinchColumnPressure,
    PinchColumnVelocity,
    PinchColumnHdiv,
    set_mesh_geometry,
    trace_top_to_basemesh,
    extend_from_p1base,
    evaluate_shape,
)

# create 1D base mesh
if args.walls:
    assert args.L < args.R0, "for -walls, initial shape must extend to locations |x|=L"
basemesh = IntervalMesh(args.mx, -args.L, args.L)
xbase = SpatialCoordinate(basemesh)
P1base = FunctionSpace(basemesh, "CG", 1)

# set initial 2D mesh geometry, but store flat coordinates
t0 = get_halfar_characteristic_time(R0=args.R0, H0=args.H0)
printpar(
    f"initializing from Halfar profile (t0 = {t0 / secpera:.3f} a, R0 = {args.R0 / 1000.0:.0f} km, H0 = {args.H0:.0f} m) ..."
)
s = Function(P1base)
set_halfar_from_lengths(xbase[0], s, R0=args.R0, H0=args.H0)
mesh = ExtrudedMesh(basemesh, layers=args.mz, layer_height=1.0 / args.mz)
xzflat = set_mesh_geometry(mesh, s)
if not args.noload:
    sR = extend_from_p1base(mesh, s)

# surface mass balance function; a=0 in Halfar solution
a = Function(P1base).interpolate(0.0)
if not args.noload:
    aR = extend_from_p1base(mesh, a)

# mixed spaces for Stokes
#   note we prefer pressure spaces with interior DOFs to avoid accidental
#   Dirichlet condition on pressure from PinchColumnPressure; also relates
#   to element-wise conservation
if args.hdiv:
    # H(div) velocity as the sum of two tensor-product (base x vertical) pieces:
    #   x-component in CG2 x DG1:  continuous in x, so normal-continuous across
    #                              vertical facets, discontinuous in z
    #   z-component in DG1 x CG2:  continuous in z, so normal-continuous across
    #                              horizontal facets, discontinuous in x
    # this is RTCF2 in Firedrake numbering, = RT_1 in Cockburn et al (2007)
    assert args.pressure == "DQ1", "-hdiv requires -pressure DQ1"
    hcg = FiniteElement("CG", basemesh.ufl_cell(), 2)
    hdg = FiniteElement("DG", basemesh.ufl_cell(), 1)
    vcg = FiniteElement("CG", interval, 2)
    vdg = FiniteElement("DG", interval, 1)
    Vel = HDiv(TensorProductElement(hcg, vdg)) + HDiv(TensorProductElement(hdg, vcg))
    V = FunctionSpace(mesh, Vel)
else:
    V = VectorFunctionSpace(mesh, "CG", 2)
if args.pressure == "DQ1":  # with -hdiv: div V = W exactly
    W = FunctionSpace(mesh, "DQ", 1, variant="spectral")
elif args.pressure == "DG0":  # not preferred: lower accuracy u,p
    W = FunctionSpace(mesh, "DG", 0)
elif args.pressure == "Q1":  # not preferred: has dof at margin element boundary
    W = FunctionSpace(mesh, "CG", 1)
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
if args.hdiv:  # extra workspace; -hdiv fails w/o (FACTOR_OUTMEMORY)
    stokespar.update({"mat_mumps_icntl_14": 200})

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

Dtyp = 1.0 / secpera  # s-1;  strain rate scale
dt_loadstab = Constant(0.0)
dt_ske = Constant(0.0)

# various helper functions


def D(w):
    """UFL expression for strain-rate tensor."""
    return 0.5 * (grad(w) + grad(w).T)


def form_stokes(loadstab=False):
    """Weak form for the power-law Stokes problem on the geometry stored in the current mesh.  This must be called every time the geometry is re-set."""
    u, p = split(up)
    Du2 = 0.5 * inner(D(u), D(u)) + (args.eps * Dtyp) ** 2.0
    nu = 0.5 * B3 * Du2 ** ((1.0 / n - 1.0) / 2.0)
    F = (2.0 * nu * inner(D(u), D(v)) - p * div(v) - q * div(u)) * dx(degree=3)
    # body force source term
    F -= inner(as_vector([0.0, -rho * g]), v) * dx
    if loadstab:
        # apply load (FSSA-type) coupling stabilization, (4.28) in Tominec et al 2026
        nn = FacetNormal(mesh)
        nsR = as_vector([-sR.dx(0), Constant(1.0)])
        F += rho * g * dt_loadstab * (0.5 * inner(u, nsR) + aR) * inner(v, nn) * ds_t
    if args.hdiv:
        F += form_hdiv_sip(u, v, nu)
    return F


def form_hdiv_sip(u, v, nu):
    """Symmetric interior penalty (SIP) terms, needed when u,v are in H(div) but not H^1; see (2.11) in Cockburn, Kanschat, Schotzau (2007), but with 2 nu D(u) in place of nu grad(u).  Because normal components are continuous, only tangential jumps are nonzero.  The same terms on ds_b (ice base) weakly impose tangential no-slip (Nitsche); the normal component is strong via DirichletBC.  There are no terms on the walls (1,2) because the tangential stress is zero there.  The penalty is kappa = alpha / h with h = |K|/|F| on a facet F with area |F| and where |K| is the average of neighboring cell volumes.  Conditionals skip zero-height cells and zero-length facets, where the Piola map divides by det J = 0."""
    nn = FacetNormal(mesh)
    vol, fa = CellVolume(mesh), FacetArea(mesh)

    def jmp(w):  # tensor jump [[w (x) n]] = w+ (x) n+ + w- (x) n-
        return outer(w("+"), nn("+")) + outer(w("-"), nn("-"))

    # degree for facet quadrature
    #    Because of nu(Du), no fixed-degree choice makes the integrals exact.
    #    Consistency terms: {D(u)}:[[v]] is roughly linear times quadratic, so
    #    degree 3 is enough there.  Geometry factors: the cells are mapped
    #    bilinearly, and the Piola map adds non-polynomial factors.
    degds = 4

    # interior facets: SIP weak form
    kappa = args.alpha * fa("+") / avg(vol)  # FIXME find Houston & Suli ref?
    sip = (
        -inner(avg(2.0 * nu * D(u)), jmp(v))
        - inner(avg(2.0 * nu * D(v)), jmp(u))
        + kappa * avg(2.0 * nu) * inner(jmp(u), jmp(v))
    )
    # exclude facets which are singular in SIP integral
    ok = And(gt(avg(vol), 0.0), gt(fa("+"), 0.0))
    # integrate horizontal (dS_h) and vertical (dS_v) facet penalties
    F = conditional(ok, sip, 0.0) * (dS_h(degree=degds) + dS_v(degree=degds))

    # on ice bed: tangential no-slip by Nitsche
    #   (similar to one-sided version of SIP, but bed *facets* do not degenerate)
    kappab = args.alpha * fa / vol
    sipb = (
        -inner(2.0 * nu * D(u), outer(v, nn))
        - inner(2.0 * nu * D(v), outer(u, nn))
        + kappab * 2.0 * nu * inner(u, v)
    )
    F += conditional(gt(vol, 0.0), sipb, 0.0) * ds_b(degree=degds)
    return F


def bcs_stokes(Z):
    """Boundary conditions in all cases, both -walls and free-boundary:
      Dirichlet: u=0 on 'bottom', u_x=0 on lateral ends (1,2)
      Neumann: stress free on 'top', zero tangential stress on (1,2)
    Added in default free-boundary (VI) case:
      pinch zero-height columns = trivialize all u,p dofs if s<stol
    With -hdiv:
      DirichletBCs on Z.sub(0) set only the normal-flux dofs, thus
      u.n=0 on 'bottom' and u.n=u_x=0 on (1,2).  For 'bottom' tangential
      part see Nitsche in form_hdiv_sip().  For walls (1,2),  the
      tangential part is free, as in the Q2 case."""
    if args.hdiv:
        # g must be a Function in V:  a Constant would be projected onto V
        #   (H(div) has no point interpolation), which fails on zero-height cells
        zero = Function(Z.sub(0))
        bcs = [
            DirichletBC(Z.sub(0), zero, "bottom"),
            DirichletBC(Z.sub(0), zero, (1, 2)),
        ]
    else:
        bcs = [
            DirichletBC(Z.sub(0), Constant((0.0, 0.0)), "bottom"),
            DirichletBC(Z.sub(0).sub(0), Constant(0.0), (1, 2)),
        ]
    if not args.walls:
        bcs += [PinchColumnPressure(Z.sub(1), None, None)]
        if args.hdiv:
            bcs += [PinchColumnHdiv(Z.sub(0), None, None)]
        else:
            bcs += [PinchColumnVelocity(Z.sub(0), None, None)]
    return bcs


def evaluate_speed(mesh, u):
    """Find average and maximum speed of ice."""
    R = FunctionSpace(mesh, "R", 0)
    one = Function(R).assign(1.0)
    area = assemble(dot(one, one) * dx)  # integrate 1 to get area of domain
    P1 = FunctionSpace(mesh, "CG", 1)
    # pointwise guard: with -hdiv, the Piola map divides by det J, which is
    #   zero in zero-height cells and at zero-thickness corners of margin cells
    detJ = JacobianDeterminant(mesh)
    speed = conditional(gt(abs(detJ), 0.0), sqrt(dot(u, u)), 0.0)
    umagav = assemble(speed * dx) / area
    umag = Function(P1).interpolate(speed)
    with umag.dat.vec_ro as vumag:
        umagmax = vumag.max()[1]
    return umagav, umagmax


def form_ske():
    """weak form for surface kinematical equation"""
    # qsurf = u.ns is the surface flux, held exactly in DG2 for both Q2 and
    #   -hdiv; testing with omega=1 gives global mass conservation, except
    #   where the VI bound s >= 0 is active
    Fske = ((snew - s) / dt_ske - qsurf - a) * omega * dx
    if not args.noedge:
        # implicit edge stabilization, formula (4.3) in Tominec et al 2026
        hbase = CellDiameter(basemesh)
        h = (hbase("+") + hbase("-")) / 2
        nbase = FacetNormal(basemesh)
        velmag = avg(speedsurf)  # speedsurf is discontinuous with -hdiv
        gamma = 0.5 * h ** 2 * velmag
        Fske += gamma * jump(grad(snew), nbase) * jump(grad(omega), nbase) * dS
    return Fske


def stresses(mesh, u):
    """Generate tensor-valued deviatoric stress tau, and scalar effective viscosity nu, from the velocity solution.  Used for output diagnostics."""
    Du2 = 0.5 * inner(D(u), D(u)) + (args.eps * Dtyp) ** 2.0
    nuexpr = 0.5 * B3 * Du2 ** ((1.0 / n - 1.0) / 2.0)
    if args.hdiv:
        # with H(div) u, evaluating at vertices of a cell with a zero-thickness
        #   corner gives NaN (Piola map divides by det J = 0); DG0 evaluates
        #   at cell midpoints, and the pointwise guard skips zero-height cells
        S = FunctionSpace(mesh, "DG", 0)
        TS = TensorFunctionSpace(mesh, "DG", 0)
        ok = gt(abs(JacobianDeterminant(mesh)), 0.0)
        nu = Function(S).interpolate(conditional(ok, nuexpr, 0.0))
        tau = Function(TS).interpolate(conditional(ok, 2.0 * nu * D(u), zero((2, 2))))
    else:
        S = FunctionSpace(mesh, "Q", 1)
        TS = TensorFunctionSpace(mesh, "Q", 1)
        nu = Function(S).interpolate(nuexpr)
        tau = Function(TS).interpolate(2.0 * nu * D(u))
    nu.rename("effective viscosity (Pa s)")
    tau /= 1.0e5
    tau.rename("tau (bar)")
    return tau, nu


def get_dt(t, dx, umagmax):
    """Determine dt from CFL and user options."""
    dtcfl = PETSc.INFINITY if args.nocfl else args.cfl * dx / umagmax
    # A practical/realistic glacier simulation should additionally determine
    # -maxdt based on desired frequency of SMB coupling.)"""
    dtmax = PETSc.INFINITY if args.maxdt < 0.0 else args.maxdt * secpera
    dtvals = np.array([dtcfl, dtmax, args.T * secpera - t])
    dtreasons = ["CFL", "MAX", "END"]
    return float(dtvals.min()), dtreasons[int(dtvals.argmin())]


def report_shape(s):
    lm, rm, smax, iarea = evaluate_shape(basemesh, s)
    printpar(
        f"  width = {(rm - lm) / 1e3:.3f} km, max(s) = {smax:.3f} m, area = {iarea / 1e6:.3f} km^2"
    )
    return iarea


# functions for surface kinematical equation
snew = Function(P1base)
omega = TestFunction(P1base)
# surface quantities for SKE, on the base mesh:  qsurf = u.ns is the surface
#   flux, and speedsurf = |u| is only used in the edge stabilization; DG2 holds
#   the flux exactly in both cases:  quadratic per base mesh cell for Q2, and
#   linear but discontinuous at base mesh vertices for RTCF2 (-hdiv)
DG2base = FunctionSpace(basemesh, "DG", 2)
qsurf = Function(DG2base)
speedsurf = Function(DG2base)
lb = Function(P1base).interpolate(Constant(0.0))
ub = Function(P1base).interpolate(Constant(PETSc.INFINITY))

# set up SKE solver
Fske = form_ske()
probske = NonlinearVariationalProblem(Fske, snew, [])  # bcs=[] .. no flux at (1,2)
solverske = NonlinearVariationalSolver(
    probske, solver_parameters=vipar, options_prefix="surf"
)

# open files if they are requested
if args.omovie is not None:
    printpar(f"opening {args.omovie} for writing u,p at each time step ...")
    movie = VTKFile(args.omovie)
    # can't interpolate into RTCF2 (no dual basis for the sum of elements),
    #   so with -hdiv write velocity in vector DQ2
    Vmovie = VectorFunctionSpace(mesh, "DQ", 2) if args.hdiv else V
if args.ots is not None:
    printpar(f"opening {args.ots} for writing ice area at each time step ...")
    tsfile = open(args.ots, "w")
    tsfile.write("# (TIME IN YEARS) (ICE AREA IN KM^2)\n")

# main time stepping loop
deltax = 2.0 * args.L / args.mx
Lkm = args.L / 1000.0
printpar(
    f"solving on {args.mx} x {args.mz} mesh ([{-Lkm:.0f}, {Lkm:.0f}] km, dx = {deltax:.3f} m) ..."
)
printpar(f"  space dimensions: n_u = {V.dim()}, n_p = {W.dim()}")
t = 0.0
for k in range(args.maxN):
    if t >= args.T * secpera:
        break

    # report on current geometry
    printpar(f"t={t / secpera:.3f} a (k={k}):")
    iarea = report_shape(s)
    if args.ots is not None:
        tsfile.write(f"{t / secpera:.6f} {iarea / 1.0e6:.6f}\n")

    # update s, a in R space, for load stabilization
    if not args.noload:
        sR.dat.data[:] = s.dat.data_ro[:]
        aR.dat.data[:] = a.dat.data_ro[:]

    # at first step, the -noload stabilization needs dtsec to be set,
    # so do initial Stokes solve w/o the -noload mechanism
    if k == 0:
        F = form_stokes(loadstab=False)
        bcs = bcs_stokes(Z)
        solve(F == 0, up, bcs=bcs, options_prefix="stokes", solver_parameters=stokespar)
        u, _ = up.subfunctions
        _, umagmax = evaluate_speed(mesh, u)
        dt, _ = get_dt(t, deltax, umagmax)

    # solving Stokes on current geometry (in current mesh) requires
    # resetting form and boundary conditions
    dt_loadstab.assign(dt)
    F = form_stokes(loadstab=(not args.noload))
    bcs = bcs_stokes(Z)  # has to be re-set because PinchColumnX uses current s
    try:
        solve(F == 0, up, bcs=bcs, options_prefix="stokes", solver_parameters=stokespar)
    except ConvergenceError:
        # previous solution can be a poor initial guess where the margin has
        #   advanced (seen with -hdiv); retry once from zero
        printpar("  Stokes solve failed from previous solution; restarting from zero ...")
        up.assign(0.0)
        solve(F == 0, up, bcs=bcs, options_prefix="stokes", solver_parameters=stokespar)
    u, p = up.subfunctions
    if args.omovie is not None:
        upera = Function(Vmovie).interpolate(secpera * u)
        upera.rename("velocity (m/a)")
        p.rename("pressure (Pa)")
        movie.write(upera, p, time=t)

    # find ice speed, and use CFL to determine time step
    umagav, umagmax = evaluate_speed(mesh, u)
    dt, reason = get_dt(t, deltax, umagmax)
    printpar(
        f"  speed (m a-1): av = {umagav * secpera:.3f}, max = {umagmax * secpera:.3f} --> {reason} dt = {dt / secpera:.3f} a"
    )

    # solve SKE for one semi-implicit Euler time-step, a variational inequality
    sRtr = extend_from_p1base(mesh, s)  # s is still the Stokes geometry
    nsR = as_vector([-sRtr.dx(0), Constant(1.0)])
    trace_top_to_basemesh(basemesh, mesh, dot(u, nsR), qsurf)
    trace_top_to_basemesh(basemesh, mesh, sqrt(dot(u, u)), speedsurf)
    snew.interpolate(s)  # initial condition
    dt_ske.assign(dt)
    solverske.solve(bounds=(lb, ub))
    s.interpolate(snew)  # update surface elevation

    # update mesh geometry (rescaling original geometry) and time
    set_mesh_geometry(mesh, s, xzorig=xzflat)
    t += dt

# report on final quantities and close files
printpar(f"t={t / secpera:.3f} a (done):")
iarea = report_shape(s)
if args.ots is not None:
    tsfile.write(f"{t / secpera:.6f} {iarea / 1.0e6:.6f}\n")
    tsfile.close()
    printpar(f"done with file {args.ots}")
if args.omovie is not None:
    printpar(f"done with file {args.omovie}")
Rh, Hh = get_halfar_dimensions_from_time(t + t0, R0=args.R0, H0=args.H0)
printpar(
    f"  Halfar values (exact for SIA): width = {2 * Rh / 1000.0:.3f} km, max(s) = {Hh:.3f} m"
)

# write .pvd with results
printpar(f"saving u,p,tau,nu,rank to {args.o} ...")
u, p = up.subfunctions
tau, nu = stresses(mesh, u)
u *= secpera
p /= 1.0e5
u.rename("velocity (m/a)")
p.rename("pressure (bar)")
fields = [u, p, tau, nu]
if mesh.comm.size > 1:
    # integer-valued element-wise process rank
    rank = Function(FunctionSpace(mesh, "DG", 0))
    rank.dat.data[:] = mesh.comm.rank
    rank.rename("rank")
    fields.append(rank)
VTKFile(args.o).write(*fields)
