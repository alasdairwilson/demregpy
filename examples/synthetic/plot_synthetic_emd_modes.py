"""
======================
Compare EMD-Like Modes
======================

Compare the standard DEM inversion with the EMD-oriented options.
"""

import matplotlib.pyplot as plt

from demregpy import dn2dem
from demregpy._example_utils import make_synthetic_case

# %%
# Use a slightly broader synthetic DEM so the mode differences are easier to see.

case = make_synthetic_case(
    dem_peaks=[
        {"m": 5.95, "s": 0.11, "d": 2.5e22},
        {"m": 6.12, "s": 0.10, "d": 2.0e22},
    ]
)

solutions = {
    "DEM": dn2dem(
        case["dn_in"],
        case["edn_in"],
        case["trmatrix"],
        case["tresp_logt"],
        case["temps"],
        nmu=50,
        warn=False,
    ),
    "L_EMD": dn2dem(
        case["dn_in"],
        case["edn_in"],
        case["trmatrix"],
        case["tresp_logt"],
        case["temps"],
        l_emd=True,
        nmu=50,
        warn=False,
    ),
    "EMD Internal": dn2dem(
        case["dn_in"],
        case["edn_in"],
        case["trmatrix"],
        case["tresp_logt"],
        case["temps"],
        emd_int=True,
        emd_ret=False,
        gloci=1,
        nmu=50,
        warn=False,
    ),
}

# %%
# Plot the three recovered curves.

fig, ax = plt.subplots(figsize=(8, 4.5))
for label, result in solutions.items():
    ax.plot(case["mlogt"], result[0], marker="o", label=f"{label} ($\\chi^2={result[3]:.2f}$)")

ax.plot(case["tresp_logt"], case["dem_mod"], "--", color="0.3", label="Input DEM")
ax.set_xlabel(r"$\log_{10} T$")
ax.set_ylabel("DEM")
ax.set_yscale("log")
ax.legend()
fig.tight_layout()
