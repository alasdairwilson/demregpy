"""
=====================
Synthetic Time Series
=====================

Run ``dn2dem`` on a small time series to produce a DEMogram.
"""

import matplotlib.pyplot as plt
import numpy as np

from demregpy import dn2dem
from demregpy._example_utils import make_synthetic_time_series

# %%
# Build a short sequence of synthetic spectra.

case = make_synthetic_time_series(n_steps=8)

# %%
# The input shape is ``(time, channel)``.

dem, edem, elogt, chisq, dn_reg = dn2dem(
    case["dn_in"],
    case["edn_in"],
    case["trmatrix"],
    case["tresp_logt"],
    case["temps"],
    nmu=50,
    warn=False,
)

print("DEMogram shape:", dem.shape)
print("Chi-squared range:", float(np.min(chisq)), float(np.max(chisq)))

# %%
# Plot the DEMogram and the per-step goodness of fit.

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

im = axes[0].imshow(
    dem.T,
    origin="lower",
    aspect="auto",
    cmap="magma",
    extent=[0, dem.shape[0] - 1, case["mlogt"][0], case["mlogt"][-1]],
)
axes[0].set_xlabel("Time Step")
axes[0].set_ylabel(r"$\log_{10} T$")
axes[0].set_title("Recovered DEMogram")
fig.colorbar(im, ax=axes[0], label="DEM")

axes[1].plot(chisq, marker="o")
axes[1].set_xlabel("Time Step")
axes[1].set_ylabel(r"$\chi^2$")
axes[1].set_title("Fit Quality")

fig.tight_layout()
