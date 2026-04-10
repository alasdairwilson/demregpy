"""
======================
Synthetic Single Pixel
======================

Recover synthetic DEMs for one spectrum at a time.
This example starts with a single Gaussian DEM and then repeats the same workflow for a three-component DEM on a finer temperature grid.
The main lesson is that the public API does not change when the DEM model becomes more structured, but resolving more structure usually benefits from a finer temperature grid and broader response coverage.
"""

import matplotlib.pyplot as plt
import numpy as np

from demregpy import dn2dem
from demregpy.plotting import plot_dem
from demregpy.synthetic import synthesize_counts


def gaussian_component(logt, amplitude, center, width):
    """Return one Gaussian DEM component on a logT grid."""
    root2pi = np.sqrt(2.0 * np.pi)
    return (amplitude / (root2pi * width)) * np.exp(-((logt - center) ** 2) / (2 * width ** 2))


# %%
# The first case is deliberately simple.
# A single broad peak is a good way to see the basic inversion pattern before trying a more structured DEM.

tresp_logt_single = np.linspace(5.7, 6.3, 7)
response_centers_single = np.array([5.75, 5.85, 5.95, 6.05, 6.15, 6.25])
trmatrix_single = np.zeros((tresp_logt_single.size, response_centers_single.size))
for i, center in enumerate(response_centers_single):
    trmatrix_single[:, i] = np.exp(-((tresp_logt_single - center) ** 2) / (2 * 0.08 ** 2))

dem_single = gaussian_component(tresp_logt_single, 4e22, 6.0, 0.12)
synthetic_single = synthesize_counts(dem_single, tresp_logt_single, trmatrix_single, error_fraction=0.1)

temps_single = 10 ** np.linspace(tresp_logt_single.min(), tresp_logt_single.max(), tresp_logt_single.size + 1)
mlogt_single = 0.5 * (np.log10(temps_single[:-1]) + np.log10(temps_single[1:]))

dem_out_single, edem_single, elogt_single, chisq_single, dn_reg_single = dn2dem(
    synthetic_single.dn_in,
    synthetic_single.edn_in,
    trmatrix_single,
    tresp_logt_single,
    temps_single,
    nmu=50,
    warn=False,
)
# The chi-squared value should be close to 1, but this does depend on the accuracy of your errors.
print(f"Single Gaussian chi-squared: {chisq_single:.3f}")

# %%
# In the single-Gaussian case, the input model is recovered accurately and the reconstructed counts stay close to the synthetic counts.
# This is the simplest demonstration of what a well-behaved single-pixel inversion looks like.

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

plot_dem(
    mlogt_single,
    dem_out_single,
    elogt=elogt_single,
    edem=edem_single,
    ax=axes[0],
    label="Recovered DEM",
    color="tab:red",
    ecolor="mistyrose",
    capsize=0,
)
axes[0].plot(tresp_logt_single, dem_single, "--", color="0.3", label="Input DEM")
axes[0].set_title("Single Gaussian")
axes[0].legend()

axes[1].plot(synthetic_single.dn_in, "o-", label="Input DN")
axes[1].plot(dn_reg_single, "s--", label="Reconstructed DN")
axes[1].set_xlabel("Channel")
axes[1].set_ylabel("DN")
axes[1].set_title(rf"Single Gaussian Fit ($\chi^2={chisq_single:.2f}$)")
axes[1].legend()

fig.tight_layout()
plt.show()

# %%
# The second case keeps the same one-spectrum workflow but uses a finer temperature grid and a wider set of synthetic channels.
# That gives the inversion a better chance of distinguishing several peaks in temperature, even though the response curves are still broad.

tresp_logt_multi = np.linspace(5.7, 7.1, 15)
response_centers_multi = np.array([5.75, 5.90, 6.05, 6.20, 6.35, 6.50, 6.65, 6.80, 6.95])
trmatrix_multi = np.zeros((tresp_logt_multi.size, response_centers_multi.size))
for i, center in enumerate(response_centers_multi):
    trmatrix_multi[:, i] = np.exp(-((tresp_logt_multi - center) ** 2) / (2 * 0.09 ** 2))

dem_multi = (
    gaussian_component(tresp_logt_multi, 2.3e22, 5.95, 0.08)
    + gaussian_component(tresp_logt_multi, 1.8e22, 6.30, 0.10)
    + gaussian_component(tresp_logt_multi, 1.1e22, 6.80, 0.09)
)
synthetic_multi = synthesize_counts(dem_multi, tresp_logt_multi, trmatrix_multi, error_fraction=0.1)

temps_multi = 10 ** np.linspace(tresp_logt_multi.min(), tresp_logt_multi.max(), tresp_logt_multi.size + 1)
mlogt_multi = 0.5 * (np.log10(temps_multi[:-1]) + np.log10(temps_multi[1:]))

dem_out_multi, edem_multi, elogt_multi, chisq_multi, dn_reg_multi = dn2dem(
    synthetic_multi.dn_in,
    synthetic_multi.edn_in,
    trmatrix_multi,
    tresp_logt_multi,
    temps_multi,
    nmu=80,
    warn=False,
)

print(f"Triple Gaussian chi-squared: {chisq_multi:.3f}")

# %%
# The three-component case is more demanding.
# The point here is not exact recovery of every peak, but how much of the input structure can be recovered once the temperature grid and the response sampling are made finer.

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

plot_dem(
    mlogt_multi,
    dem_out_multi,
    elogt=elogt_multi,
    edem=edem_multi,
    ax=axes[0],
    label="Recovered DEM",
    color="tab:red",
    ecolor="mistyrose",
    capsize=0,
)
axes[0].plot(tresp_logt_multi, dem_multi, "--", color="0.3", label="Input DEM")
axes[0].set_title("Triple Gaussian")
axes[0].legend()

axes[1].plot(synthetic_multi.dn_in, "o-", label="Input DN")
axes[1].plot(dn_reg_multi, "s--", label="Reconstructed DN")
axes[1].set_xlabel("Channel")
axes[1].set_ylabel("DN")
axes[1].set_title(rf"Triple Gaussian Fit ($\chi^2={chisq_multi:.2f}$)")
axes[1].legend()

fig.tight_layout()
plt.show()
