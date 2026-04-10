"""
====================
Synthetic AIA Counts
====================

Generate synthetic AIA channel counts from the bundled AIA temperature response functions.
This is useful for building controlled AIA test problems before moving on to map data or full inversion runs.
"""

import matplotlib.pyplot as plt
import numpy as np

from demregpy import load_aia_response
from demregpy.plotting import plot_dem
from demregpy.synthetic import synthesize_counts

# %%
# The bundled AIA response curves can be used directly to make synthetic AIA count vectors from any DEM model defined on the response temperature grid.
# This is a simple way to build controlled AIA test problems without starting from AIA maps.

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
# The point of this example is to connect three pieces of the workflow.
# A DEM model on the left is folded through the AIA responses in the middle to produce the synthetic channel counts on the right.
# The resulting ``synthetic.dn_in`` and ``synthetic.edn_in`` can then be passed straight into ``dn2dem``.

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

plot_dem(
    tresp_logt,
    dem_model,
    ax=axes[0],
    fmt="-",
    color="tab:red",
    ecolor="mistyrose",
)
axes[0].set_title("Input DEM Model")

for i, channel in enumerate(channels):
    axes[1].plot(tresp_logt, trmatrix[:, i], label=channel)
axes[1].set_xlabel(r"$\log_{10} T$")
axes[1].set_ylabel("AIA Response")
axes[1].set_yscale("log")
axes[1].set_title("Bundled AIA Responses")
axes[1].legend(fontsize=8, ncol=2)

x = np.arange(len(channels))
axes[2].plot(x, synthetic.dn_clean, "o--", color="0.4", label="Noise-free DN")
axes[2].errorbar(
    x,
    synthetic.dn_in,
    yerr=synthetic.edn_in,
    fmt="o",
    color="tab:blue",
    ecolor="lightskyblue",
    capsize=3,
    label="Synthetic DN",
)
axes[2].set_xticks(x, channels, rotation=45)
axes[2].set_ylabel("DN")
axes[2].set_title("Synthetic AIA Counts")
axes[2].legend()

fig.tight_layout()
plt.show()
