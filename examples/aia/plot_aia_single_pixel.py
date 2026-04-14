"""
================
AIA Single Pixel
================

Minimal ``dn2dem`` inversion on one AIA pixel.
"""

import matplotlib.pyplot as plt
import numpy as np

from demregpy import dn2dem, load_aia_response
from demregpy.plotting import plot_dem
from demregpy.tests.example_data import load_aia_full_disk_maps

# %%
# Extract one pixel from co-aligned AIA maps.
# Only optically thin channels are used; 304 A is excluded because absorption makes it less useful for DEMs.
# Map values are converted to count rates by dividing by exposure time.
# No additional time-dependent degradation correction is applied here.

maps = load_aia_full_disk_maps()
rate_maps = [amap / amap.exposure_time for amap in maps]
channels, tresp_logt, trmatrix = load_aia_response()

x = 500
y = 500
dn_in = np.array([amap.data[x, y] for amap in rate_maps], dtype=float)
edn_in = 0.1 * dn_in + 1

temps = 10 ** np.linspace(5.6, 7.4, num=21)
mlogt = 0.5 * (np.log10(temps[:-1]) + np.log10(temps[1:]))

print("Input DN / pix / s:", dn_in)

# %%
# We use a flat error floor plus 10% fractional uncertainty.
# A full analysis might use a more instrument-specific error model.
# Errors matter: under-estimates cause overfitting and poor convergence; over-estimates produce over-smoothed solutions.

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
# The reconstructed counts are the quickest check of inversion quality.
# If the channel fit looks poor, revisit the inputs and uncertainty model before interpreting the DEM.

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
