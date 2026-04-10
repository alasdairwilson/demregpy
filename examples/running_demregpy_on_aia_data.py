"""
==========================
Using demregpy on AIA data
==========================

Run a small DEM inversion on one pixel from the local AIA test data.
This is the smallest AIA example in the repository and mirrors the more detailed gallery examples.

For more focused examples, see the ``examples/aia`` directory.
"""

import matplotlib.pyplot as plt
import numpy as np

from demregpy import dn2dem, load_aia_response
from demregpy.plotting import plot_dem
from demregpy.tests.example_data import load_aia_full_disk_maps

# Load the bundled AIA response file and one pixel from the local FITS fixtures.
# The map values are converted to count rates by dividing by exposure time before the inversion.
# A moderately bright coronal pixel is used so the example is less noisy than a quiet-Sun pixel near the map centre.
# The example uses local data rather than remote downloads so it is deterministic and quick to run.
maps = load_aia_full_disk_maps()
rate_maps = [amap / amap.exposure_time for amap in maps]
channels, tresp_logt, trmatrix = load_aia_response()

x = 430
y = 520
dn_in = np.array([amap.data[x, y] for amap in rate_maps], dtype=float)
edn_in = 0.1 * dn_in + 1
temps = 10 ** np.linspace(5.6, 7.4, num=21)
mlogt = 0.5 * (np.log10(temps[:-1]) + np.log10(temps[1:]))

print(f"Using pixel ({x}, {y})")
print("channels:", channels)
print("input DN / pix / s:", dn_in)

# Recover a DEM for that pixel.
# The uncertainties here use the same flat-plus-fractional model as the gallery AIA examples.
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
print("reconstructed DN:", dn_reg)

# Plot the recovered DEM and the data fit.
# The reconstructed counts are plotted alongside the input counts so the quality of the fit is visible immediately.
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
