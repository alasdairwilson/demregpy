"""
===========================
Compare Weighting Strategies
===========================

Compare the default self-normalized constraint, ``gloci``, and a user-supplied weight.
"""

import matplotlib.pyplot as plt

from demregpy import dn2dem
from demregpy._example_utils import make_synthetic_case, make_user_weight

# %%
# Build one synthetic problem and solve it three ways.

case = make_synthetic_case()
user_weight = make_user_weight(case["mlogt"], case["tresp_logt"], case["dem_mod"])

solutions = {
    "Default": dn2dem(
        case["dn_in"],
        case["edn_in"],
        case["trmatrix"],
        case["tresp_logt"],
        case["temps"],
        nmu=50,
        warn=False,
    ),
    "Gloci": dn2dem(
        case["dn_in"],
        case["edn_in"],
        case["trmatrix"],
        case["tresp_logt"],
        case["temps"],
        gloci=1,
        nmu=50,
        warn=False,
    ),
    "User Weight": dn2dem(
        case["dn_in"],
        case["edn_in"],
        case["trmatrix"],
        case["tresp_logt"],
        case["temps"],
        dem_norm0=user_weight,
        nmu=50,
        warn=False,
    ),
}

for label, result in solutions.items():
    print(f"{label:>12s} chi-squared: {result[3]:.3f}")

# %%
# Plot the recovered DEMs together.

fig, ax = plt.subplots(figsize=(8, 4.5))
for label, result in solutions.items():
    ax.plot(case["mlogt"], result[0], marker="o", label=f"{label} ($\\chi^2={result[3]:.2f}$)")

ax.plot(case["tresp_logt"], case["dem_mod"], "--", color="0.3", label="Input DEM")
ax.set_xlabel(r"$\log_{10} T$")
ax.set_ylabel("DEM")
ax.set_yscale("log")
ax.legend()
fig.tight_layout()
