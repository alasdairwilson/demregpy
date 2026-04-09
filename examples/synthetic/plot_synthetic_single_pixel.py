"""
======================
Synthetic Single Pixel
======================

Recover a compact synthetic DEM from one set of channel counts.
"""

import matplotlib.pyplot as plt
import numpy as np

from demregpy import dn2dem
from demregpy.plotting import plot_dem
from demregpy.synthetic import synthesize_counts

# %%
# Build a small synthetic response matrix and one Gaussian DEM model.

tresp_logt = np.linspace(5.7, 6.3, 7)
response_centers = np.array([5.75, 5.85, 5.95, 6.05, 6.15, 6.25])
trmatrix = np.zeros((tresp_logt.size, response_centers.size))
for i, center in enumerate(response_centers):
    trmatrix[:, i] = np.exp(-((tresp_logt - center) ** 2) / (2 * 0.08 ** 2))

root2pi = np.sqrt(2.0 * np.pi)
dem_model = (4e22 / (root2pi * 0.12)) * np.exp(-((tresp_logt - 6.0) ** 2) / (2 * 0.12 ** 2))
synthetic = synthesize_counts(dem_model, tresp_logt, trmatrix, error_fraction=0.1)

temps = 10 ** np.linspace(tresp_logt.min(), tresp_logt.max(), tresp_logt.size + 1)
mlogt = 0.5 * (np.log10(temps[:-1]) + np.log10(temps[1:]))

# %%
# Run the inversion.

dem, edem, elogt, chisq, dn_reg = dn2dem(
    synthetic.dn_in,
    synthetic.edn_in,
    trmatrix,
    tresp_logt,
    temps,
    nmu=50,
    warn=False,
)

print(f"chi-squared: {chisq:.3f}")

# %%
# Compare the recovered DEM to the synthetic truth and check the data fit.

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

plot_dem(
    mlogt,
    dem,
    elogt=elogt,
    edem=edem,
    ax=axes[0],
    label="Recovered DEM",
    color="tab:red",
    ecolor="mistyrose",
    capsize=0,
)
axes[0].plot(tresp_logt, dem_model, "--", color="0.3", label="Input DEM")
axes[0].legend()

axes[1].plot(synthetic.dn_in, "o-", label="Input DN")
axes[1].plot(dn_reg, "s--", label="Reconstructed DN")
axes[1].set_xlabel("Channel")
axes[1].set_ylabel("DN")
axes[1].legend()

fig.tight_layout()
