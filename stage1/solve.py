#!/usr/bin/env python3

from firedrake import *

mesh = Mesh('domain.msh')
V = VectorFunctionSpace(mesh, 'Lagrange', 2)
W = FunctionSpace(mesh, 'Lagrange', 1)
Z = V * W
up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)

def D(w):               # strain-rate tensor
    return 0.5 * (grad(w) + grad(w).T)

g = 9.81                # m s-2
rho = 910.0             # kg m-3
nu = 1.0e+8             # Pa s

fbody = Constant((0.0, - rho * g))
F = nu * inner(D(u), D(v)) * dx \
    - p * div(v) * dx \
    - div(u) * q * dx \
    - inner(fbody, v) * dx
bcs = [ DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (42,)) ]

par = {'snes_type': 'ksponly',
       'ksp_type': 'preonly',
       'pc_type': 'svd'}
solve(F == 0, up, bcs=bcs, options_prefix='s', solver_parameters=par)

u, p = up.split()
u.rename('velocity')
p.rename('pressure')
File('domain.pvd').write(u, p)
