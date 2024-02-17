#!/usr/bin/env python3

from firedrake import *
from firedrake.petsc import PETSc

printpar = PETSc.Sys.Print        # print once even in parallel
printpar('reading mesh from dome.msh ...')
mesh = Mesh('dome.msh')
printpar('    (mesh with %d elements and %d vertices)' \
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
nu = 0.5 * B3 * Du2**((1.0 / n - 1.0)/2.0)
F = ( inner(2.0 * nu * D(u), D(v)) \
      - p * div(v) - q * div(u) - inner(fbody, v) ) * dx(degree=3)
bcs = [ DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (42,)) ]

printpar('solving ...')
par = {'snes_converged_reason': None,
       'snes_monitor': None,
       'snes_linesearch_type': 'bt',
       'ksp_type': 'preonly',
       'pc_type': 'lu',
       'pc_factor_shift_type': 'inblocks',
       'pc_factor_mat_solver_type': 'mumps'}
solve(F == 0, up, bcs=bcs, solver_parameters=par)

# integrate 1 to get area of domain
R = FunctionSpace(mesh, 'R', 0)
one = Function(R).assign(1.0)
area = assemble(dot(one,one) * dx)

# print average and maximum velocity
P1 = FunctionSpace(mesh, 'CG', 1)
umagav = assemble(sqrt(dot(u, u)) * dx) / area
umag = Function(P1).interpolate(sqrt(dot(u, u)))
with umag.dat.vec_ro as vumag:
    umagmax = vumag.max()[1]
printpar('  ice speed (m a-1): av = %.3f, max = %.3f' \
         % (umagav * secpera, umagmax * secpera))

printpar('saving to dome.pvd ...')
u = up.subfunctions[0]
p = up.subfunctions[1]
u *= secpera    # save in m/a
p /= 1.0e5      # save in bar
u.rename('velocity (m/a)')
p.rename('pressure (bar)')
File('dome.pvd').write(u, p)
