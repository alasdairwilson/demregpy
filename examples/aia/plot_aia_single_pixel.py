"""
================
AIA Single Pixel
================

This example shows how to run ``dn2dem`` on a single AIA pixel.
The count rates come from AIA maps and the response matrix comes from the bundled AIA temperature response file.
The example shows the minimum set of ingredients needed for a DEM inversion from real AIA data.
The same workflow extends directly to any other AIA pixel once the channel counts, uncertainties, and response matrix have been prepared.
"""

import matplotlib.pyplot as plt
import numpy as np

from demregpy import dn2dem, load_aia_response
from demregpy.plotting import plot_dem
from demregpy.tests.example_data import load_aia_full_disk_maps

# %%
# Start by extracting one pixel from a set of co-aligned AIA maps.
# The inversion requires the set of filter values, their uncertainties, and the corresponding temperature response matrix.
# We only use the optically thin AIA channels here as absorption makes the 304 channel less useful for DEMs.
# The map values are converted to count rates by dividing by exposure time before the inversion.
# A moderately bright coronal pixel is used here so the signal is clearer than a quiet-Sun pixel near the map centre.
# This compact example does not apply an additional time-dependent degradation correction.

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
# The inversion needs one count vector, one uncertainty vector, the response matrix, and the target temperature grid.
# We use a flat error plus 10% uncertainty model here, but a full AIA analysis might use a more instrument-specific error estimate.
# The errors play a large role in the solution calculated by demregpy;
# Underestimating errors will lead to very low rates of convergence and overfitting, while overestimating errors will lead to very high rates of convergence but a smoother solution that may miss real structure in the DEM.

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
# For real data, the reconstructed counts are usually the quickest check of whether the inversion is behaving sensibly.
# If the channel-by-channel fit looks poor, it is often better to revisit the inputs and the uncertainty model before over-interpreting the DEM curve itself.

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
