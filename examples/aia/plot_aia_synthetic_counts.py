"""
====================
Synthetic AIA Counts
====================

Generate synthetic AIA channel counts from the bundled AIA temperature response functions.
This is a simple way to build controlled AIA test problems from a known DEM model.
"""

import matplotlib.pyplot as plt
import numpy as np

from demregpy import load_aia_response
from demregpy.plotting import plot_dem
from demregpy.synthetic import synthesize_counts

# %%
# Once the AIA response matrix has been loaded, ``synthesize_counts`` can fold any DEM model on that grid into synthetic channel counts.
# This is useful for building controlled inversion tests before moving on to map data.

channels, tresp_logt, trmatrix = load_aia_response()

root2pi = np.sqrt(2.0 * np.pi)
dem_model = (
    (3.0e21 / (root2pi * 0.08)) * np.exp(-((tresp_logt - 6.05) ** 2) / (2 * 0.08 ** 2))
    + (8.0e20 / (root2pi * 0.10)) * np.exp(-((tresp_logt - 6.85) ** 2) / (2 * 0.10 ** 2))
)

synthetic = synthesize_counts(
    dem_model,
    tresp_logt,
    trmatrix,
    error_fraction=0.1,
    noise_fraction=0.05,
    random_state=0,
)

print("Channels:", channels)
print("Noise-free DN:", synthetic.dn_clean)
print("Synthetic DN:", synthetic.dn_in)
print("Synthetic uncertainties:", synthetic.edn_in)

# %%
# The left panel is the input DEM model.
# The right panel is the synthetic AIA dataset produced by folding that model through the bundled response matrix.
# The resulting ``synthetic.dn_in`` and ``synthetic.edn_in`` can be passed directly into ``dn2dem``.

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

plot_dem(
    tresp_logt,
    dem_model,
    ax=axes[0],
    fmt="-",
    color="tab:red",
    ecolor="mistyrose",
)
axes[0].set_title("Input DEM Model")

x = np.arange(len(channels))
axes[1].plot(x, synthetic.dn_clean, "o--", color="0.4", label="Noise-free DN")
axes[1].errorbar(
    x,
    synthetic.dn_in,
    yerr=synthetic.edn_in,
    fmt="o",
    color="tab:blue",
    ecolor="lightskyblue",
    capsize=3,
    label="Synthetic DN",
)
axes[1].set_xticks(x, channels, rotation=45)
axes[1].set_ylabel("DN")
axes[1].set_title("Synthetic AIA Counts")
axes[1].legend()

fig.tight_layout()
plt.show()
