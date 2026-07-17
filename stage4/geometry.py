# Geometry-management tools for an evolving 2D glacier model which solves
# the Stokes equations and the surface kinematical (free-surface)
# equation on an extruded mesh.
#
# Components:
#   * PinchColumnPressure is a class for Dirichlet-type conditions for
#     trivializing the pressure in zero-height columns
#   * PinchColumnVelocity is the same, but for velocity
#   * set_mesh_geometry() sets the extruded mesh geometry from a
#     surface elevation on the base mesh
#   * extend_from_p1base() extends a scalar P1 function on the base mesh
#     to a Q1 function in the R space on the extruded mesh
#   * trace_to_vector_p2base() transfers the trace of a P2 vector field
#     on the extruded mesh to a vector field on the base mesh
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

    def __init__(self, V, g, sub_domain, htol=1.0):
        self.htol = htol
        super().__init__(V, fd.Constant(0.0), None)

    @cached_property
    def nodes(self):
        V = self.function_space()
        # get application ctx from coordinates DM
        ctx = get_appctx(V.mesh().coordinates.function_space().dm)
        assert ctx is not None, f"got None for appctx from {V.mesh()} coordinates DM"
        # return P1 nodes in columns with surface elevation less than htol
        s = fd.Function(V).interpolate(ctx["sR"])
        # print(np.where(s.dat.data_ro_with_halos < self.htol)[0])
        return np.where(s.dat.data_ro_with_halos < self.htol)[0]


class PinchColumnVelocity(fd.DirichletBC):
    """For vector-valued velocity.  Compare PinchColumnPressure."""

    def __init__(self, V, g, sub_domain, htol=1.0):
        self.htol = htol
        super().__init__(V, fd.as_vector([0.0, 0.0]), None)

    @cached_property
    def nodes(self):
        V = self.function_space()
        # get application ctx from coordinates DM
        ctx = get_appctx(V.mesh().coordinates.function_space().dm)
        assert ctx is not None, f"got None for appctx from {V.mesh()} coordinates DM"
        # return vector P2 nodes in columns with height (thickness) less than htol
        ss = fd.Function(V).interpolate(fd.as_vector([ctx["sR"], ctx["sR"]]))
        return np.where(ss.dat.data_ro_with_halos < self.htol)[0]


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


def trace_to_vector_p2base(basemesh, mesh, u, ubm):
    """On an extruded mesh, put the surface trace of a vector-valued function u, which is in P2, into ubm which is a P2 vector-valued function on basemesh."""
    P2V = fd.VectorFunctionSpace(mesh, "CG", 2)
    bc = fd.DirichletBC(P2V, 0.0, "top")
    ubm.dat.data_with_halos[:] = u.dat.data_with_halos[bc.nodes]
    return None


def evaluate_shape(basemesh, s, stol=1.0):
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
