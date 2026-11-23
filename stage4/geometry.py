# Geometry-management tools for an evolving 2D glacier model which solves
# the Stokes equations and the surface kinematical (free-surface)
# equation on an extruded mesh.
#
# Components:
#   * PinchColumnPressure is a class for Dirichlet-type conditions for
#     trivializing the pressure in zero-height columns
#   * PinchColumnVelocity is the same, but for velocity
#   * PinchColumnVelocityHdiv is the same, but for H(div) velocity
#   * PinchColumnPressureHdiv is the same, but for pressure with H(div) velocity
#   * set_mesh_geometry() sets the extruded mesh geometry from a
#     surface elevation on the base mesh
#   * extend_from_p1base() extends a scalar P1 function on the base mesh
#     to a Q1 function in the R space on the extruded mesh
#   * trace_top_to_basemesh() transfers the top trace of a scalar
#     expression on the extruded mesh to a DG function on the base mesh
#   * set_thick_columns() sets an indicator of thick (not margin) columns
#   * margin_inflow() computes the flux from thick into margin columns
#   * evaluate_shape() computes the left and right margin position,
#     maximum surface elevation, and area of a planar glacier shape
#
# Known limitation:  These tools currently assume the bed elevation is
# identically zero.  This can easily be fixed.
#
# More complete geometry-management tools, suitable for certain
# multilevel hierarchies, are in https://github.com/bueler/stokes-extrude

import numpy as np
from functools import cached_property
from pyop2.mpi import MPI
import firedrake as fd
from firedrake.dmhooks import get_appctx, pop_appctx, push_appctx
from firedrake.petsc import PETSc


class PinchColumnPressure(fd.DirichletBC):
    """A 'pinched column' is one where the degrees of freedom are trivialized (pinned) according to the column height being below a tolerance.  The column height is determined from R-space elevation stored in an application context."""

    def __init__(self, V, g, sub_domain, htol=0.01):
        self.htol = htol
        super().__init__(V, fd.Constant(0.0), None)

    @cached_property
    def nodes(self):
        V = self.function_space()
        # get application ctx from coordinates DM
        ctx = get_appctx(V.mesh().coordinates.function_space().dm)
        assert ctx is not None, f"got None for appctx from {V.mesh()} coordinates DM"
        assert "sR" in ctx, f"key 'sR' not in context dictionary returned by DM"
        # return P1 nodes in columns with surface elevation less than htol
        s = fd.Function(V).interpolate(ctx["sR"])  # also need bR if b is not zero
        # print(np.where(s.dat.data_ro_with_halos < self.htol)[0])
        return np.where(s.dat.data_ro_with_halos < self.htol)[0]


class PinchColumnVelocity(fd.DirichletBC):
    """For vector-valued velocity.  Compare PinchColumnPressure."""

    def __init__(self, V, g, sub_domain, htol=0.01):
        self.htol = htol
        super().__init__(V, fd.as_vector([0.0, 0.0]), None)

    @cached_property
    def nodes(self):
        V = self.function_space()
        # get application ctx from coordinates DM
        ctx = get_appctx(V.mesh().coordinates.function_space().dm)
        assert ctx is not None, f"got None for appctx from {V.mesh()} coordinates DM"
        assert "sR" in ctx, f"key 'sR' not in context dictionary returned by DM"
        # return vector P2 nodes in columns with height (thickness) less than htol
        ss = fd.Function(V).interpolate(fd.as_vector([ctx["sR"], ctx["sR"]]))  # also need bR if b is not zero
        return np.where(ss.dat.data_ro_with_halos < self.htol)[0]


def margin_columns(svert, htol, theta):
    """Given svert, the surface elevation at the base mesh vertices of each column (one row per column), return a boolean array which is True for margin columns.  A column is a margin column if min(s) < max(htol, theta * max(s)) over its vertices, and otherwise a thick column.  Thus thick columns are not too thin (htol) and not too tapered (theta), so their cells are far from degenerate."""
    smin, smax = svert.min(axis=1), svert.max(axis=1)
    return smin < np.maximum(htol, theta * smax)


class PinchColumnVelocityHdiv(fd.DirichletBC):
    """For H(div) velocity on an extruded mesh over an interval base mesh.  Compare PinchColumnVelocity.  H(div) dofs are not point values, so the criterion is applied by mesh entity.  Margin and thick columns are as in margin_columns(); this excludes degenerate cells, with a zero-thickness corner, from thick columns.  Dofs interior to a margin column (horizontal facets and cell interiors) are pinned.  Dofs on a vertical facet (normal flux of u_x) are pinned if s<htol at that base mesh vertex, or if both adjacent columns are margin columns.  Thus a facet between a thick and a margin column keeps its dofs; it is a boundary of the thick column, with hydrostatic normal stress (see form_hdiv_margin_stress() in solve.py)."""

    def __init__(self, V, g, sub_domain, htol=0.01, theta=0.3):
        self.htol = htol
        self.theta = theta
        # zero Function in V, so no projection (compare bcs_stokes() in solve.py)
        super().__init__(V, fd.Function(V), None)

    @cached_property
    def nodes(self):
        V = self.function_space()
        ctx = get_appctx(V.mesh().coordinates.function_space().dm)
        assert ctx is not None, f"got None for appctx from {V.mesh()} coordinates DM"
        assert "sR" in ctx, f"key 'sR' not in context dictionary returned by DM"
        sR = ctx["sR"]  # one value per base mesh vertex; also need bR if b is not zero
        svert = sR.dat.data_ro_with_halos[sR.function_space().cell_node_list]
        # local dofs of one cell, by entity: (0, *) = on vertical facets, at
        #   base mesh vertex i;  (1, *) = interior to the column
        edofs = V.finat_element.entity_dofs()
        ncelllayers = V.mesh().layers - 1
        pinned = set()
        facetcand, facetkeep = set(), set()  # facet dofs of margin/thick columns
        colpin = margin_columns(svert, self.htol, self.theta)
        for c, cnodes in enumerate(V.cell_node_list):
            for (dimh, dimv), ents in edofs.items():
                for i, ldofs in ents.items():
                    if len(ldofs) == 0:
                        continue
                    if dimh == 0:
                        # (0,1) entities are vertical edges, numbered by base
                        #   mesh vertex; (0,0) are cell vertices, 2 per base
                        #   mesh vertex (none have dofs for RTCF2)
                        ib = i if dimv == 1 else i // 2
                        thin = svert[c, ib] < self.htol
                    else:
                        thin = colpin[c]
                    for ld in ldofs:
                        for l in range(ncelllayers):
                            node = cnodes[ld] + l * V.offset[ld]
                            if thin:
                                pinned.add(node)
                            elif dimh == 0:
                                (facetcand if colpin[c] else facetkeep).add(node)
        pinned = pinned.union(facetcand - facetkeep)
        return np.array(sorted(pinned), dtype=np.int32)


class PinchColumnPressureHdiv(fd.DirichletBC):
    """For discontinuous pressure with H(div) velocity:  pin all pressure dofs in margin columns, as in margin_columns().  Compare PinchColumnVelocityHdiv."""

    def __init__(self, W, g, sub_domain, htol=0.01, theta=0.3):
        self.htol = htol
        self.theta = theta
        super().__init__(W, fd.Constant(0.0), None)

    @cached_property
    def nodes(self):
        W = self.function_space()
        sR = get_appctx(W.mesh().coordinates.function_space().dm)["sR"]
        svert = sR.dat.data_ro_with_halos[sR.function_space().cell_node_list]
        colpin = margin_columns(svert, self.htol, self.theta)
        nl = W.mesh().layers - 1
        pinned = [cn + l * W.offset for c, cn in enumerate(W.cell_node_list) if colpin[c] for l in range(nl)]
        if len(pinned) == 0:
            return np.array([], dtype=np.int32)
        return np.unique(np.concatenate(pinned)).astype(np.int32)


def set_mesh_geometry(mesh, s, xzorig=None):
    """Set the geometry for an extruded mesh from a P1 scalar field s, that is, a surface elevation.  In default usage (xzorig not set), the original coordinate field is returned.  Pushes a copy of s, in the R space, into an application context attached to the DM, which is itself attached to the coordinate function space.  The PinchColumnX classes use this field."""
    sR = fd.Function(fd.FunctionSpace(mesh, "CG", 1, vfamily="R", vdegree=0))
    sR.dat.data[:] = s.dat.data_ro[:]
    Vcoord = mesh.coordinates.function_space()
    if xzorig is None:
        xz = fd.SpatialCoordinate(mesh)
        xzorig = fd.Function(Vcoord).interpolate(xz)
    else:
        xz = fd.Function(Vcoord).interpolate(xzorig)
    XZ = fd.Function(Vcoord).interpolate(fd.as_vector([xz[0], sR * xz[1]]))
    mesh.coordinates.assign(XZ)
    _ = pop_appctx(mesh.coordinates.function_space().dm)
    push_appctx(mesh.coordinates.function_space().dm, {"sR": sR})
    return xzorig


def extend_from_p1base(mesh, f):
    """On an extruded mesh, extend a P1 function f(x), defined for x in basemesh, to the extruded (x,z) mesh.  Returns a function on mesh in the 'R' constant-in-the-vertical space."""
    Q1R = fd.FunctionSpace(mesh, "CG", 1, vfamily="R", vdegree=0)
    fextend = fd.Function(Q1R)
    fextend.dat.data[:] = f.dat.data_ro[:]
    return fextend


def trace_top_to_basemesh(basemesh, mesh, expr, fbm):
    """Put the top-surface trace of a scalar UFL expression on the extruded mesh into fbm, a DG function of degree k on basemesh.  The intermediate space Q is DGk horizontally, so each column keeps its own values (no averaging at base mesh vertices), and CG1 vertically, so Q has dofs on the top surface.  Assumes the top dofs of Q are ordered like the dofs of DGk on basemesh."""
    k = fbm.function_space().ufl_element().degree()
    Q = fd.FunctionSpace(mesh, "DG", k, vfamily="CG", vdegree=1)
    # pointwise guard: for H(div) u the Piola map divides by det J, which is
    #   zero in zero-height columns and at zero-thickness corners of margin
    #   cells; set the trace to zero there
    detJ = fd.JacobianDeterminant(mesh)
    f = fd.Function(Q).interpolate(fd.conditional(fd.gt(abs(detJ), 0.0), expr, 0.0))
    bc = fd.DirichletBC(Q, 0.0, "top")
    fbm.dat.data_with_halos[:] = f.dat.data_with_halos[bc.nodes]
    return None


def set_thick_columns(s, chibm, chiR, tauR, SR, htol=0.01, theta=0.3):
    """Set indicators chi = 1 for thick columns and chi = 0 for margin columns, as in margin_columns().  chibm is DG0 on the base mesh, and chiR is DG0 x R on the extruded mesh (one dof per column, ordered like DG0 on the base mesh).  Also set the taper tau = min(s)/avg(s) over the vertices of each column into tauR (DG0 x R); tau >= theta in thick columns.  Also set the shear factor S = mz (max(s) - min(s)) / min(s) into SR (DG0 x R), the slope of the layers times the cell width, divided by the thinnest cell height in the column; S = 0 in margin columns.  s is P1 on the base mesh."""
    svert = s.dat.data_ro_with_halos[s.function_space().cell_node_list]
    thick = np.logical_not(margin_columns(svert, htol, theta)).astype(float)
    savg = svert.mean(axis=1)
    tau = np.where(savg > 0.0, svert.min(axis=1) / np.where(savg > 0.0, savg, 1.0), 1.0)
    chibm.dat.data_with_halos[chibm.function_space().cell_node_list[:, 0]] = thick
    chiR.dat.data_with_halos[chiR.function_space().cell_node_list[:, 0]] = thick
    tauR.dat.data_with_halos[tauR.function_space().cell_node_list[:, 0]] = tau
    smin, smax = svert.min(axis=1), svert.max(axis=1)
    mz = SR.function_space().mesh().layers - 1
    S = np.where(thick > 0.5, mz * (smax - smin) / np.where(smin > 0.0, smin, 1.0), 0.0)
    SR.dat.data_with_halos[SR.function_space().cell_node_list[:, 0]] = S
    return None


def margin_inflow(basemesh, mesh, u, chiR, qbm):
    """For each margin column M, put the vertically-integrated flux into M through the vertical facets it shares with thick columns, divided by |M_base|, into qbm (DG0 on basemesh); qbm is zero in thick columns.  The flux u.n is evaluated on the thick side, which is not degenerate.  Because u is in H(div), this is exactly the flux which leaves the thick columns through those facets."""
    nn = fd.FacetNormal(mesh)
    psi = fd.TestFunction(chiR.function_space())
    tp, tm = fd.gt(chiR("+"), 0.5), fd.gt(chiR("-"), 0.5)  # thick on + / - side
    intoplus = fd.conditional(fd.And(tm, fd.Not(tp)), fd.dot(u("-"), nn("-")), 0.0)
    intominus = fd.conditional(fd.And(tp, fd.Not(tm)), fd.dot(u("+"), nn("+")), 0.0)
    infl = fd.assemble((intoplus * psi("+") + intominus * psi("-")) * fd.dS_v)
    DG0base = qbm.function_space()
    Abase = fd.Function(DG0base).interpolate(fd.CellVolume(basemesh))
    qbm.dat.data_with_halos[:] = (
        infl.dat.data_ro_with_halos / Abase.dat.data_ro_with_halos
    )
    return None


def evaluate_shape(basemesh, s, stol=0.01):
    # note: these methods are correct in parallel
    xx = basemesh.coordinates.dat.data_ro
    spos = xx[s.dat.data_ro > stol]
    myl, myr = PETSc.INFINITY, PETSc.NINFINITY
    if len(spos) > 0:
        myl, myr = min(spos), max(spos)
    lmargin = float(basemesh.comm.allreduce(myl, op=MPI.MIN))
    rmargin = float(basemesh.comm.allreduce(myr, op=MPI.MAX))
    with s.dat.vec_ro as ss:
        _, smax = ss.max()  # find all-processes maximum value of a PETSc Vec
    area = fd.assemble(s * fd.dx)
    return lmargin, rmargin, smax, area
