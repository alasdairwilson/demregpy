"""
===================
AIA Patch Inversion
===================

Run ``dn2dem`` on a small AIA patch and inspect the resulting DEM cube.
The useful pattern here is that once the AIA count rates and uncertainties have been stacked into an array with channels on the last axis, the same public ``dn2dem`` call can be used to recover a DEM at every pixel in the patch.
That same pattern carries over to larger cutouts and to time-dependent image stacks.
"""

import matplotlib.pyplot as plt
import numpy as np

from demregpy import dn2dem, load_aia_response
from demregpy.tests.example_data import load_aia_full_disk_maps

# %%
# Start by extracting a submap from each AIA channel.
# The patch is kept small so the example stays quick, but any sized region can be used.
# The map values are converted to count rates by dividing by exposure time before stacking them, our final array has shape ``(nx, ny, nf)``, which is the natural map-like input form for ``dn2dem``.
# This compact example does not apply an additional time-dependent degradation correction.
# We load the aia response file using `load_aia_response`.

maps = load_aia_full_disk_maps()
rate_maps = [amap / amap.exposure_time for amap in maps]
_channels, tresp_logt, trmatrix = load_aia_response()

# Define the patch pixel coordinates
width = 10
height = 10
nx, ny = maps[0].data.shape
x0 = nx // 2 - width // 2
y0 = ny // 2 - height // 2
x1 = x0 + width
y1 = y0 + height

# create the input array
dn_in = np.stack(
    [amap.data[x0:x1, y0:y1] for amap in rate_maps],
    axis=-1,
).astype(float)
edn_in = 0.1 * dn_in + 1

# The temperature grid for the inversion
temps = 10 ** np.linspace(5.6, 7.4, num=21)
mlogt = 0.5 * (np.log10(temps[:-1]) + np.log10(temps[1:]))

# %%
# The inversion call is the same regardless of whether the input describes one spectrum or an n-dimensional grid of data.
# ``dn2dem`` solves each pixel independently and returns arrays with the same spatial axes, with temperature replacing channel on the DEM outputs.
# Here the average DEM over the patch is used to pick one representative temperature bin to display.

dem, edem, elogt, chisq, dn_reg = dn2dem(
    dn_in,
    edn_in,
    trmatrix,
    tresp_logt,
    temps,
    nmu=40,
    warn=False,
)

# %%
# One useful first view is a temperature slice near the part of the solution where the patch is brightest on average.
# Plotting that slice alongside one input channel and the per-pixel chi-squared map helps separate thermal structure from places where the fit is poor and the inputs or uncertainties may need more attention.

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

axes[0].imshow(dn_in[:, :, 2], origin="lower", cmap="sdoaia171")
axes[0].set_title("AIA 171 Input Rate")

peak_bin = int(np.argmax(np.mean(dem, axis=(0, 1))))
peak_logt = mlogt[peak_bin]

im = axes[1].imshow(np.log10(dem[:, :, peak_bin] + 1e-30), origin="lower", cmap="magma")
axes[1].set_title(f"DEM Slice at logT={peak_logt:.2f}")
fig.colorbar(im, ax=axes[1], label="log10(DEM)")

axes[2].imshow(chisq, origin="lower", cmap="viridis")
axes[2].set_title(r"Per-pixel $\chi^2$")

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()
