"""
======================
Synthetic Single Pixel
======================

Recover synthetic DEMs for one spectrum at a time.
This example starts with a single Gaussian DEM and then repeats the same workflow for a three-component DEM on a finer temperature grid.
The first case shows the basic pattern of building synthetic counts, running ``dn2dem``, and checking the result in both DEM space and data space.
The second case shows the same workflow on a more structured DEM, `demregpy` can recover any shape of DEM as long as it is smooth.
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
# Start with a deliberately simple synthetic problem.
# A single broad Gaussian is useful because there is a clear input model, so the comparison between the recovered DEM and the original synthetic model is easy to read.
# The basic pattern for using dn2dem: define counts and uncertainties, run ``dn2dem``, and then compare the recovered DEM and the reconstructed counts to the inputs.

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
# In this simple case, the recovered DEM follows the input model closely and the reconstructed counts stay close to the synthetic counts.
# That is what a well-behaved single-pixel inversion looks like when the temperature structure is smooth and the synthetic problem is well resolved by the responses.

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
# We can increase the complexity of the model plasma without changing any of our methods.
# By using a more complex model atmosphere, we are tasked with recovering more structure in the DEM.
# We have also included a finer temperature grid and a wider set of synthetic channels make it more realistic to ask how much multi-thermal structure can be recovered from the same inversion machinery.

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
# The point is not exact recovery of every peak, because broad temperature responses and regularization both smooth the solution.
# The useful question is whether the recovered DEM captures the main temperature structure and whether the reconstructed counts still explain the synthetic data well.
# Note that unlike methods which rely on parametric models, the solver does not know that the input DEM is a sum of Gaussians, so it is not trying to fit a small set of model parameters.

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
