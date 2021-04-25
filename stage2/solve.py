#!/usr/bin/env python3

from firedrake import *

print('reading dome.pvd ...')
mesh = Mesh('dome.msh')
print('    (mesh with %d elements and %d vertices)' \
      % (mesh.num_cells(), mesh.num_vertices()))

V = VectorFunctionSpace(mesh, 'Lagrange', 2)
W = FunctionSpace(mesh, 'Lagrange', 1)
Z = V * W
up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)

def D(w):               # strain-rate tensor
    return 0.5 * (grad(w) + grad(w).T)

secpera = 31556926.0    # seconds per year
g = 9.81                # m s-2
rho = 910.0             # kg m-3
n = 3.0
A3 = 3.1689e-24         # Pa-3 s-1;  EISMINT I value of ice softness
B3 = A3**(-1.0/3.0)     # Pa s(1/3);  ice hardness
eps = 0.0001
Dtyp = 1.0 / secpera    # s-1

fbody = Constant((0.0, - rho * g))
Du2 = 0.5 * inner(D(u), D(u)) + (eps * Dtyp)**2.0
r = 1.0 / n - 1.0
F = inner(B3 * Du2**(r/2.0) * D(u), D(v)) * dx \
    - p * div(v) * dx \
    - div(u) * q * dx \
    - inner(fbody, v) * dx
bcs = [ DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (42,)) ]

par = {'snes_linesearch_type': 'bt',
       'ksp_type': 'gmres',
       'pc_type': 'fieldsplit',
       'pc_fieldsplit_type': 'schur',
       'pc_fieldsplit_schur_factorization_type': 'full',
       'pc_fieldsplit_schur_precondition': 'a11',
       'fieldsplit_0_ksp_type': 'preonly',
       'fieldsplit_0_pc_type': 'lu',
       'fieldsplit_1_ksp_rtol': 1.0e-3,
       'fieldsplit_1_ksp_type': 'gmres',
       'fieldsplit_1_pc_type': 'none'}
       
print('solving ...')
solve(F == 0, up, bcs=bcs, options_prefix='s', solver_parameters=par)

print('saving to dome.pvd ...')
u, p = up.split()
u.rename('velocity')
p.rename('pressure')
File('dome.pvd').write(u, p)
