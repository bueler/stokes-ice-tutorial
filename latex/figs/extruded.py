#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from writeout import writeout

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "cm"
bigfsize = 30.0

x = np.linspace(0.0, 10.0, 11)
# following data was generated a long time ago ...
b = np.array(
    [
        0.5300000000000001,
        0.36185948536513635,
        -0.18136049906158566,
        -0.1558830996397852,
        0.16787164932467638,
        0.07119577782212608,
        0.4226854163999131,
        1.218121471138974,
        1.592419336666987,
        1.5,
        1.7,
    ]
)
s = np.array(
    [
        0.5300000000000001,
        0.36185948536513635,
        -0.18136049906158566,
        1.4801848778600042,
        2.3962987740708668,
        2.4711957778221265,
        2.6511125411461034,
        2.5,
        1.592419336666987,
        1.5,
        1.7,
    ]
)


def isgap(a, b):
    return abs(a - b) > 1.0e-4


def plotextruded(x, b, s, mz=5, Hmin=0.0):
    """Plot the extruded mesh.  mz is the number of element layers.  Hmin > 0
    produces a cliff at the margin.  Make the top of the top element strong."""
    assert all(s >= b)
    plt.plot(x, b, "k-")
    dz = np.maximum(s - b, Hmin) / mz  # everywhere at least Hmin/mz
    # plot verticals
    for j in range(len(x)):
        if (
            isgap(b[j], s[j])
            or (j > 0 and isgap(b[j - 1], s[j - 1]))
            or (j < len(x) - 1 and isgap(b[j + 1], s[j + 1]))
        ):
            snew = b[j] + (mz - 1) * dz[j]
            plt.plot([x[j], x[j]], [b[j], snew], "k-")
    # plot element tops
    for j in range(len(x) - 1):
        if isgap(b[j], s[j]) or isgap(b[j + 1], s[j + 1]):
            for k in range(mz):
                zl = b[j] + k * dz[j]
                zr = b[j + 1] + k * dz[j + 1]
                if (k == 0) or (k + 1 == mz):  # show strong top and bottom
                    plt.plot([x[j], x[j + 1]], [zl, zr], "k-", lw=3.0)
                else:
                    plt.plot([x[j], x[j + 1]], [zl, zr], "k-")
        else:
            plt.plot([x[j], x[j + 1]], [s[j], s[j + 1]], "k-", lw=3.0)
    # plt.text(x[3], s[3]+0.2, r'$s_h$', fontsize=bigfsize, color='k')
    # plt.text(x[6]+0.4, b[6], r'$b_h$', fontsize=bigfsize, color='k')
    # plt.text(x[5]+0.2, 0.08+(b[5] + s[5])/2.0, r'$\mathbf{u}_h$', fontsize=bigfsize, color='k')
    plt.axis("off")


# extruded FE domain figure
plt.figure(figsize=(15, 4))
plotextruded(x, b, s)
writeout("extruded.png")

# add "pinched"
plt.text(x[0], b[0] + 0.2, r"$\mathbf{u}=0$, $p=0$", fontsize=bigfsize, color="firebrick")
plt.plot(x[:3], b[:3], "firebrick", lw=5.0)
plt.text(x[8], b[8] - 0.5, r"$\mathbf{u}=0$, $p=0$", fontsize=bigfsize, color="firebrick")
plt.plot(x[8:], b[8:], "firebrick", lw=5.0)
writeout("pinched.png")
