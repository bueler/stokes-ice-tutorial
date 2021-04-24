# stokes-ice-tutorial

stages of codes
  * domain1.geo  hand-written simple geometry
  * solve1.py    linear Stokes equations
  * domain2.py   generate profile geometry
  * solve2.py    Glen-Stokes equations with simplest-possible solver
  * solve3.py    solver options

FIXME FROM HERE

Generate and solve Stokes problems from Bueler profile geometry.  `domain.py`
is the mesh-generation stage and `solve.py` is the solution stage.

For example:

        $ ./domain.py -N 60 -o dome60.geo
        $ gmsh -2 dome60.geo
        $ ./solve.py -mesh dome60.msh -s_snes_converged_reason -o dome60.pvd

For the second stage to work one needs to activate the venv for firedrake, e.g.:

        $ unset PETSC_DIR;  unset PETSC_ARCH;
        $ source ~/firedrake/bin/activate

Consider PETSc options `-s_snes_monitor`, `-s_snes_max_it`, and `-s_snes_rtol`
for controlling the Newton iteration in `solve.py`.

For help on the two stages do:

         $ ./domain.py -h
         $ ./solve.py -solvehelp               # options for solve.py
         $ ./solve.py -mesh dome60.msh -help   # PETSc options for solver
