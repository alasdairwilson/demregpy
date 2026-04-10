"""
================
AIA Single Pixel
================

Run ``dn2dem`` on one pixel from the local AIA test fixtures.
This example shows the smallest realistic AIA workflow in the package: load maps, extract one pixel, run the inversion, and compare the reconstructed counts.
The same pattern extends directly to any other AIA pixel once you have the channel counts, uncertainties, and response matrix in hand.
"""

import matplotlib.pyplot as plt
import numpy as np

from demregpy import dn2dem, load_aia_response
from demregpy.plotting import plot_dem
from demregpy.tests.example_data import load_aia_full_disk_maps

# %%
# This is the smallest map-based workflow that still looks like a real AIA analysis.
# The center pixel is only a convenient stand-in for any other location you might want to extract from your own maps.

maps = load_aia_full_disk_maps()
channels, tresp_logt, trmatrix = load_aia_response()

x = maps[0].data.shape[0] // 2
y = maps[0].data.shape[1] // 2
dn_in = np.array([amap.data[x, y] for amap in maps], dtype=float)
edn_in = 0.1 * dn_in + 1e-8

temps = 10 ** np.linspace(5.7, 7.1, num=17)
mlogt = 0.5 * (np.log10(temps[:-1]) + np.log10(temps[1:]))

print(f"Using pixel ({x}, {y})")
print("Channels:", channels)
print("Input DN:", dn_in)

# %%
# The inversion itself is performed by calling dn2dem.

dem, edem, elogt, chisq, dn_reg = dn2dem(
    dn_in,
    edn_in,
    trmatrix,
    tresp_logt,
    temps,
    nmu=40,
    warn=False,
)

print(f"chi-squared: {chisq:.3f}")

# %%
# The reconstructed counts are often the fastest sanity check when moving from examples to real data.
# If the channel-by-channel fit looks poor, it usually makes sense to revisit the inputs before over-interpreting the DEM curve.
# We use the function plot_dem here to show the DEM curve with error bars.

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

plot_dem(
    mlogt,
    dem,
    elogt=elogt,
    edem=edem,
    ax=axes[0],
    color="tab:red",
    ecolor="mistyrose",
    capsize=0,
)
axes[0].set_title("Recovered DEM")

axes[1].plot(dn_in, "o-", label="Input DN")
axes[1].plot(dn_reg, "s--", label="Reconstructed DN")
axes[1].set_xticks(range(len(channels)), channels, rotation=45)
axes[1].set_ylabel("DN / pix / s")
axes[1].legend()

fig.tight_layout()
plt.show()
