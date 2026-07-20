# Plot area time series from ascii files written by stage4/:
#   python3 solve.py ... -mx MX -ots areaMX.txt
# for MX in mxrange.

mxrange = [50, 100, 200, 400, 800]

import numpy as np
import matplotlib.pyplot as plt
from writeout import writeout

# get reference area, which is initial area of highest res
ta = np.loadtxt(f"timeseries/area{max(mxrange):d}.txt")
refarea = ta[0,1]

for MX in mxrange:
    ta = np.loadtxt(f"timeseries/area{MX:d}.txt")
    anomalypercent = 100.0 * ((ta[:,1] / refarea) - 1.0)
    plt.plot(ta[:,0], anomalypercent, label=f"MX = {MX:d}")
plt.xlabel("t  (a)", fontsize=14.0)
plt.ylabel("percent ice area anomaly", fontsize=14.0)
plt.legend()
plt.grid("on")
#plt.show()
writeout("areatimeseries.png")
