"""
======================
Synthetic Single Pixel
======================

Recover a compact synthetic DEM from one set of channel counts.
"""

import matplotlib.pyplot as plt

from demregpy import dn2dem, plot_dem
from demregpy._example_utils import make_synthetic_case

# %%
# Build a small synthetic test problem.

case = make_synthetic_case()

# %%
# Run the inversion.

dem, edem, elogt, chisq, dn_reg = dn2dem(
    case["dn_in"],
    case["edn_in"],
    case["trmatrix"],
    case["tresp_logt"],
    case["temps"],
    nmu=50,
    warn=False,
)

print(f"chi-squared: {chisq:.3f}")

# %%
# Compare the recovered DEM to the synthetic truth and check the data fit.

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

plot_dem(
    case["mlogt"],
    dem,
    elogt=elogt,
    edem=edem,
    ax=axes[0],
    label="Recovered DEM",
    color="tab:red",
    ecolor="mistyrose",
    capsize=0,
)
axes[0].plot(case["tresp_logt"], case["dem_mod"], "--", color="0.3", label="Input DEM")
axes[0].legend()

axes[1].plot(case["dn_in"], "o-", label="Input DN")
axes[1].plot(dn_reg, "s--", label="Reconstructed DN")
axes[1].set_xlabel("Channel")
axes[1].set_ylabel("DN")
axes[1].legend()

fig.tight_layout()
