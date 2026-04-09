"""
================
AIA Single Pixel
================

Run ``dn2dem`` on one pixel from the local AIA test fixtures.
"""

import matplotlib.pyplot as plt
import numpy as np

from demregpy import dn2dem, load_aia_response, plot_dem
from demregpy._example_utils import load_aia_test_maps

# %%
# Load the bundled response matrix and the local test FITS files.

maps = load_aia_test_maps()
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
# Recover a DEM for that single pixel.

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
# Plot the recovered DEM and compare the modeled counts to the input counts.

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
