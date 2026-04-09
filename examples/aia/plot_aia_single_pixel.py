"""
================
AIA Single Pixel
================

Run ``dn2dem`` on one pixel from the local AIA test fixtures.
"""

import matplotlib.pyplot as plt

from demregpy import dn2dem, plot_dem
from demregpy._example_utils import make_aia_pixel_problem

# %%
# Load the bundled response matrix and the local test FITS files.

case = make_aia_pixel_problem()
print(f"Using pixel ({case['x']}, {case['y']})")
print("Channels:", case["channels"])
print("Input DN:", case["dn_in"])

# %%
# Recover a DEM for that single pixel.

dem, edem, elogt, chisq, dn_reg = dn2dem(
    case["dn_in"],
    case["edn_in"],
    case["trmatrix"],
    case["tresp_logt"],
    case["temps"],
    nmu=40,
    warn=False,
)

print(f"chi-squared: {chisq:.3f}")

# %%
# Plot the recovered DEM and compare the modeled counts to the input counts.

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

plot_dem(
    case["mlogt"],
    dem,
    elogt=elogt,
    edem=edem,
    ax=axes[0],
    color="tab:red",
    ecolor="mistyrose",
    capsize=0,
)
axes[0].set_title("Recovered DEM")

axes[1].plot(case["dn_in"], "o-", label="Input DN")
axes[1].plot(dn_reg, "s--", label="Reconstructed DN")
axes[1].set_xticks(range(len(case["channels"])), case["channels"], rotation=45)
axes[1].set_ylabel("DN / pix / s")
axes[1].legend()

fig.tight_layout()
