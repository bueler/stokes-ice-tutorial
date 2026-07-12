# Functions to set surface geometry from the Halfar time-dependent
# shallow ice approximation (SIA) geometry solution.
#
# The Halfar solution from
#   P. Halfar (1981), On the dynamics of the ice sheets,
#   J. Geophys. Res. 86 (C11), 11065--11072
# is a similarity solution of the isothermal SIA model.  It is a spreading
# dome solution with zero surface mass balance everywhere.  The surface
# has center height H0 and is zero outside of (-R0,R0) at time t0.  The
# time t0 can be interpreted as the time since creation as a delta mass
# of ice.
#
# Note that the Halfar solution is *not* an exact solution of the
# free-surface + Stokes coupled equations.  However, asymptotically for small
# aspect ratio (eps = [H]/[L]) it is arbitrarily close to an solution of
# of the coupled problem.
#
# This module is also the source of the physical constants.

from firedrake import *

# physical constants
secpera = 31556926.0  # seconds per year
g = 9.81  # m s-2
rho = 910.0  # kg m-3
n = 3.0
A3 = 3.1689e-24  # Pa-3 s-1;  EISMINT I value of ice softness
B3 = A3 ** (-1.0 / 3.0)  # Pa s(1/3);  ice hardness

# derived constants
pp = 1.0 + 1.0 / n
rr = n / (2.0 * n + 1.0)
Gamma = 2.0 * A3 * (rho * g) ** 3.0 / 5.0
alpha = 1.0 / 11.0  # n = 3
beta = alpha  # 1D


def get_halfar_characteristic_time(R0=None, H0=None):
    # This formula for t0 is derived by following equations (3)--(9)
    # # in Bueler et al (2005) but in 1D.  Only for n=3.
    t0 = (7.0 / 4.0) ** 3.0 * (beta / Gamma) * R0 ** 4.0 / H0 ** 7.0
    return t0


def set_halfar_from_time(x, s, t=None, R0=10000.0, H0=1000.0):
    t0 = get_halfar_characteristic_time(R0=R0, H0=H0)
    s0 = (t / t0) ** beta * R0  # margin position at time t
    xsc = (t / t0) ** (-beta) * x / R0
    xi = conditional(abs(x) < s0, (1.0 - abs(xsc) ** pp), 0.0)
    s.interpolate(H0 * (t / t0) ** (-alpha) * xi ** rr)


def set_halfar_from_lengths(x, s, R0=None, H0=None):
    xi = conditional(abs(x) < R0, 1.0 - abs(x / R0) ** pp, 0.0)
    s.interpolate(H0 * xi ** rr)
