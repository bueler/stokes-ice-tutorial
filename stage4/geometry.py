# Geometry-management tools for an evolving 2D glacier model which solves
# the Stokes equations and the surface kinematical (free-surface)
# equation on an extruded mesh.
#
# Components:
#   * PinchColumnPressure is a class for Dirichlet-type conditions for
#     trivializing the pressure in zero-height columns
#   * PinchColumnVelocity is the same, but for velocity
#   * PinchColumnHdiv is the same, but for H(div) velocity
#   * set_mesh_geometry() sets the extruded mesh geometry from a
#     surface elevation on the base mesh
#   * extend_from_p1base() extends a scalar P1 function on the base mesh
#     to a Q1 function in the R space on the extruded mesh
#   * trace_top_to_basemesh() transfers the top trace of a scalar
#     expression on the extruded mesh to a DG function on the base mesh
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


class PinchColumnHdiv(fd.DirichletBC):
    """For H(div) velocity on an extruded mesh over an interval base mesh.  Compare PinchColumnVelocity.  H(div) dofs are not point values, so the criterion is applied by mesh entity:  dofs on vertical facets (normal flux of u_x) are pinned if s<htol at that base mesh vertex, and dofs interior to a column (horizontal facets and cell interiors) are pinned if s<htol at the base mesh cell midpoint."""

    def __init__(self, V, g, sub_domain, htol=0.01):
        self.htol = htol
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
                        thin = 0.5 * (svert[c, 0] + svert[c, 1]) < self.htol
                    if thin:
                        for ld in ldofs:
                            for l in range(ncelllayers):
                                pinned.add(cnodes[ld] + l * V.offset[ld])
        return np.array(sorted(pinned), dtype=np.int32)


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
