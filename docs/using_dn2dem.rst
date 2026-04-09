***********************
dn2dem Tutorial
***********************

This tutorial works through an example of using demregpy to recover a Differential Emission Measure (DEM) from synthetic data.
The aim is to:

- build a simple DEM model
- generate synthetic counts using a fake temperature response matrix
- recover a DEM using :func:`demregpy.dn2dem`
- compare the recovered counts with the input data
- briefly explore some of the different optional arguments to ``dn2dem`` that affect the weighting and solution space
- see how the same principles extend to larger input arrays

By the end, you should have a clear picture of using ``dn2dem`` to produce DEMs.

Set Up a Small Synthetic Problem
================================

Start with a compact temperature response matrix and one DEM profile.

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   from demregpy import dn2dem
   from demregpy.plotting import plot_dem
   from demregpy.synthetic import synthesize_counts

   tresp_logt = np.linspace(5.7, 6.3, 7)
   response_centers = np.array([5.75, 5.85, 5.95, 6.05, 6.15, 6.25])
   trmatrix = np.zeros((tresp_logt.size, response_centers.size))

   for i, center in enumerate(response_centers):
       trmatrix[:, i] = np.exp(-((tresp_logt - center) ** 2) / (2 * 0.08 ** 2))

   root2pi = np.sqrt(2.0 * np.pi)
   dem_model = (4e22 / (root2pi * 0.12)) * np.exp(
       -((tresp_logt - 6.0) ** 2) / (2 * 0.12 ** 2)
   )

   synthetic = synthesize_counts(
       dem_model,
       tresp_logt,
       trmatrix,
       error_fraction=0.1,
   )

   temps = 10 ** np.linspace(tresp_logt.min(), tresp_logt.max(), tresp_logt.size + 1)
   mlogt = 0.5 * (np.log10(temps[:-1]) + np.log10(temps[1:]))

At this point you have:

- ``synthetic.dn_in``: the counts you will pass to ``dn2dem``
- ``synthetic.edn_in``: the corresponding uncertainties
- ``trmatrix`` and ``tresp_logt``: the response matrix and its log10(T) grid
- ``temps``: the temperature-bin edges for the recovered DEM

Recover the DEM
===============

The primary way to recover a DEM is via :func:`demregpy.dn2dem`, which uses regularized inversion to solve for the unknown DEM.

.. code-block:: python

   dem, edem, elogt, chisq, dn_reg = dn2dem(
       synthetic.dn_in,
       synthetic.edn_in,
       trmatrix,
       tresp_logt,
       temps,
       nmu=50,
       warn=False,
   )

The returned arrays are:

- ``dem``: the recovered DEM
- ``edem``: vertical uncertainties on the DEM
- ``elogt``: horizontal temperature resolution estimates
- ``chisq``: the final reduced chi-squared
- ``dn_reg``: counts reconstructed from the recovered DEM

Plot the DEM and Check the Data Fit
===================================

To check the quality of the recovered DEM, you can:

1. compare the recovered DEM with what you expect
2. compare ``dn_reg`` with the original channel counts

.. code-block:: python

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

If ``dn_reg`` and the input counts differ strongly, look at ``chisq`` and then
check the solver settings, the temperature grid, and the response matrix.

Try Different Weighting Choices
===============================

The default solve uses the self-normalized weighting built inside ``dn2dem``.
Two common alternatives are using the EM loci curves with ``gloci`` and
providing a manual weighting curve with ``dem_norm0``.

``gloci``
---------

Set ``gloci=1`` to build the weighting from the minimum of the EM loci curves
from all filters.

.. code-block:: python

   dem_gloci, edem_gloci, elogt_gloci, chisq_gloci, dn_reg_gloci = dn2dem(
       synthetic.dn_in,
       synthetic.edn_in,
       trmatrix,
       tresp_logt,
       temps,
       gloci=1,
       warn=False,
   )

You can also pass a length-``nf`` 0/1 mask to use only selected filters.
This can be useful if you want the loci weighting to come from only part of
the input data.

.. code-block:: python

   dem_gloci, edem_gloci, elogt_gloci, chisq_gloci, dn_reg_gloci = dn2dem(
       synthetic.dn_in,
       synthetic.edn_in,
       trmatrix,
       tresp_logt,
       temps,
       gloci=[1, 1, 0, 0, 1, 1],
       warn=False,
   )

``dem_norm0``
-------------

If you already have a DEM-shaped weighting curve, pass it through
``dem_norm0``.

.. code-block:: python

   dem_weight = dem_model / dem_model.max()

   dem_prior, edem_prior, elogt_prior, chisq_prior, dn_reg_prior = dn2dem(
       synthetic.dn_in,
       synthetic.edn_in,
       trmatrix,
       tresp_logt,
       temps,
       dem_norm0=dem_weight,
       warn=False,
   )

See :doc:`weighting` for a fuller explanation of these choices.

Try EMD Space
=============

You can also run the internal solve in emission measure distribution space.
This changes the internal weighting and can be useful to compare with the
default DEM-space solve.

.. code-block:: python

   dem_emd, edem_emd, elogt_emd, chisq_emd, dn_reg_emd = dn2dem(
       synthetic.dn_in,
       synthetic.edn_in,
       trmatrix,
       tresp_logt,
       temps,
       emd_int=True,
       warn=False,
   )

If you want the returned result in EMD units as well, add ``emd_ret=True``.

.. code-block:: python

   dem_emd, edem_emd, elogt_emd, chisq_emd, dn_reg_emd = dn2dem(
       synthetic.dn_in,
       synthetic.edn_in,
       trmatrix,
       tresp_logt,
       temps,
       emd_int=True,
       emd_ret=True,
       warn=False,
   )

Using ``dn2dem`` with Other Input Shapes
========================================

Carrying out many DEMs at once is not only easy but is also significantly faster.
The last axis of the input arrays, ``dn_in`` and ``edn_in``, is always the filter or channel, corresponding to the columns of the response matrix.
The remaining leading axes are treated as independent spectra, and the same solve is carried out for each spectrum in turn.
You can provide up to 3 leading axes, so the input can be up to 4D, with the filter axis last.

Common shapes are:

- ``(nf,)`` for a single spectrum
- ``(n, nf)`` for a spatial line or time series
- ``(nx, ny, nf)`` for a map
- ``(ntime, nx, ny, nf)`` for a stack of maps

For example, if you have ten spectra stacked in time:

.. code-block:: python

   dn_series = np.repeat(synthetic.dn_in[np.newaxis, :], 10, axis=0)
   edn_series = np.repeat(synthetic.edn_in[np.newaxis, :], 10, axis=0)

   dem_series, edem_series, elogt_series, chisq_series, dn_reg_series = dn2dem(
       dn_series,
       edn_series,
       trmatrix,
       tresp_logt,
       temps,
       warn=False,
   )

The output has the same leading shape, with the filter axis replaced by temperature bin.
In this case:

- ``dn_series`` has shape ``(10, nf)``
- ``dem_series`` has shape ``(10, nt)``
- ``chisq_series`` has shape ``(10,)``

Where To Go Next
================

- :doc:`weighting` for more detail on the weighting-related options
- :doc:`generated/gallery/index` for more runnable examples
- :doc:`api` for the full function signatures and lower-level routines
