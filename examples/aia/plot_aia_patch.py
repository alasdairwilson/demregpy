"""
==================
AIA Patch Inversion
==================

Run ``dn2dem`` on a small local AIA patch and inspect one recovered temperature slice.
"""

import matplotlib.pyplot as plt
import numpy as np

from demregpy import dn2dem
from demregpy._example_utils import make_aia_patch_problem

# %%
# Load a small patch around the center of the local test maps.

case = make_aia_patch_problem(width=10, height=10)

dem, edem, elogt, chisq, dn_reg = dn2dem(
    case["dn_in"],
    case["edn_in"],
    case["trmatrix"],
    case["tresp_logt"],
    case["temps"],
    nmu=40,
    warn=False,
)

peak_bin = int(np.argmax(np.mean(dem, axis=(0, 1))))
peak_logt = case["mlogt"][peak_bin]

print("Patch input shape:", case["dn_in"].shape)
print("DEM cube shape:", dem.shape)
print(f"Mean chi-squared: {np.mean(chisq):.3f}")
print(f"Displaying DEM slice near logT={peak_logt:.2f}")

# %%
# Compare one observed AIA channel with one recovered DEM slice.

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

axes[0].imshow(case["dn_in"][:, :, 2], origin="lower", cmap="sdoaia171")
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
