# Geometry-management tools for an evolving glacier model which
# solves the Stokes equations and the surface kinematical (free-surface)
# equation on an extruded mesh:
#   * PinchColumnPressure and PinchColumnVelocity classes are
#     Dirichlet-type conditions for trivializing zero-height columns.
#   * set_mesh_geometry() sets the extruded mesh geometry from a
#     surface elevation on the base mesh.
#   * trace_scalar_to_p1() returns a P1 field on the base mesh from
#     the trace of a field on the extruded mesh.
#
# Known limitation:  These tools currently assume the bed elevation is
# identically zero.  This can easily be fixed.

# More complete geometry-management tools, suitable for certain
# multilevel hierarchies, are in https://github.com/bueler/stokes-extrude

import numpy as np
from functools import cached_property
import firedrake as fd
from firedrake.dmhooks import get_appctx, pop_appctx, push_appctx


class PinchColumnPressure(fd.DirichletBC):
    """A 'pinched column' is one where the degrees of freedom are trivialized (pinned) according to the column height being below a tolerance.  The column height is determined from R-space elevation stored in an application context."""

    def __init__(self, V, g, sub_domain, htol=1.0):
        self.htol = htol
        super().__init__(V, fd.Constant(0.0), None)

    @cached_property
    def nodes(self):
        V = self.function_space()
        mesh = V.mesh()
        # get application ctx from coordinates DM
        ctx = get_appctx(mesh.coordinates.function_space().dm)
        assert ctx is not None, f"got None for appctx from {mesh} coordinates DM"
        # return P1 nodes in columns with surface elevation less than htol
        s = fd.Function(V).interpolate(ctx["sR"])
        return np.where(s.dat.data_ro_with_halos < self.htol)[0]


class PinchColumnVelocity(fd.DirichletBC):
    """For vector-valued velocity.  Compare PinchColumnPressure."""

    def __init__(self, V, g, sub_domain, htol=1.0):
        self.htol = htol
        super().__init__(V, fd.as_vector([0.0, 0.0]), None)

    @cached_property
    def nodes(self):
        V = self.function_space()
        mesh = V.mesh()
        # get application ctx from coordinates DM
        ctx = get_appctx(mesh.coordinates.function_space().dm)
        assert ctx is not None, f"got None for appctx from {mesh} coordinates DM"
        # return vector P2 nodes in columns with height (thickness) less than htol
        # warning: assumes velocity space is P2
        P2scalar = fd.FunctionSpace(V.mesh(), "CG", 2)
        s = fd.Function(P2scalar).interpolate(ctx["sR"])
        ss = fd.Function(V).interpolate(fd.as_vector([s, s]))
        return np.where(ss.dat.data_ro_with_halos < self.htol)[0]


def set_mesh_geometry(mesh, s, xzorig=None):
    Q1R = fd.FunctionSpace(mesh, "CG", 1, vfamily="R", vdegree=0)
    sR = fd.Function(Q1R)
    sR.dat.data[:] = s.dat.data_ro[:]
    Vcoord = mesh.coordinates.function_space()
    if xzorig is None:
        xz = fd.SpatialCoordinate(mesh)
        xzorig = fd.Function(Vcoord).interpolate(xz)
    else:
        xz = fd.Function(Vcoord).interpolate(xzorig)
    XZ = fd.Function(Vcoord).interpolate(fd.as_vector([xz[0], sR * xz[1]]))
    mesh.coordinates.assign(XZ)
    # push application context onto the DM attached to the coordinate function space
    actx = {"sR": sR}
    dm = mesh.coordinates.function_space().dm
    _ = pop_appctx(dm)
    push_appctx(dm, actx)
    return xzorig


def trace_scalar_to_p1(basemesh, mesh, f, nointerpolate=False):
    """On an extruded mesh, compute the trace of any scalar function f
    along the top surface boundary at the P1 nodes.
    Set nointerpolate=True if f is already P1.  Returns a P1 function on
    basemesh."""
    P1 = fd.FunctionSpace(mesh, "CG", 1)
    if nointerpolate:
        fP1 = f
    else:
        fP1 = fd.Function(P1).interpolate(f)
    bc = fd.DirichletBC(P1, 0.0, "top")
    P1basemesh = fd.FunctionSpace(basemesh, "CG", 1)
    fbm = fd.Function(P1basemesh)
    fbm.dat.data_with_halos[:] = fP1.dat.data_with_halos[bc.nodes]
    return fbm


def extend_p1_from_basemesh(mesh, f):
    """On an extruded mesh, extend a P1 function f(x), defined for x
    in basemesh, to the extruded (x,z) mesh.  Returns a function
    on mesh in the 'R' constant-in-the-vertical space."""
    Q1R = fd.FunctionSpace(mesh, "CG", 1, vfamily="R", vdegree=0)
    fextend = fd.Function(Q1R)
    fextend.dat.data[:] = f.dat.data_ro[:]
    return fextend
