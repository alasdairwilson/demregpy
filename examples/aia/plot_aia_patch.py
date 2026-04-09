"""
===================
AIA Patch Inversion
===================

Run ``dn2dem`` on a small local AIA patch and inspect one recovered temperature slice.
"""

import matplotlib.pyplot as plt
import numpy as np

from demregpy import dn2dem, load_aia_response
from demregpy.tests.example_data import load_aia_full_disk_maps

# %%
# Load a small patch around the center of the local test maps.

maps = load_aia_full_disk_maps()
_channels, tresp_logt, trmatrix = load_aia_response()

width = 10
height = 10
nx, ny = maps[0].data.shape
x0 = nx // 2 - width // 2
y0 = ny // 2 - height // 2
x1 = x0 + width
y1 = y0 + height

dn_in = np.stack([amap.data[x0:x1, y0:y1] for amap in maps], axis=-1).astype(float)
edn_in = 0.1 * dn_in + 1e-8
temps = 10 ** np.linspace(5.7, 7.1, num=17)
mlogt = 0.5 * (np.log10(temps[:-1]) + np.log10(temps[1:]))

dem, edem, elogt, chisq, dn_reg = dn2dem(
    dn_in,
    edn_in,
    trmatrix,
    tresp_logt,
    temps,
    nmu=40,
    warn=False,
)

peak_bin = int(np.argmax(np.mean(dem, axis=(0, 1))))
peak_logt = mlogt[peak_bin]

print("Patch input shape:", dn_in.shape)
print("DEM cube shape:", dem.shape)
print(f"Mean chi-squared: {np.mean(chisq):.3f}")
print(f"Displaying DEM slice near logT={peak_logt:.2f}")

# %%
# Compare one observed AIA channel with one recovered DEM slice.

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

axes[0].imshow(dn_in[:, :, 2], origin="lower", cmap="sdoaia171")
axes[0].set_title("AIA 171 Input")

im = axes[1].imshow(np.log10(dem[:, :, peak_bin] + 1e-30), origin="lower", cmap="magma")
axes[1].set_title(f"DEM Slice at logT={peak_logt:.2f}")
fig.colorbar(im, ax=axes[1], label="log10(DEM)")

axes[2].imshow(chisq, origin="lower", cmap="viridis")
axes[2].set_title(r"Per-pixel $\chi^2$")

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
