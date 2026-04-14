***********************
dn2dem Tutorial
***********************

This page works through a synthetic DEM recovery with :func:`demregpy.dn2dem`, covering the default solve, weighting options, EMD mode, and multi-dimensional inputs.

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

Recover the DEM
===============

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

Plot the DEM and Check the Data Fit
===================================

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

If ``dn_reg`` and the input counts differ strongly, look at ``chisq`` and then check the solver settings, the temperature grid, and the response matrix.

Try Different Weighting Choices
===============================

The default solve uses the self-normalized weighting built inside ``dn2dem``.
Two common alternatives are using the EM loci curves with ``gloci`` and providing a manual weighting curve with ``dem_norm0``.

``gloci``
---------

Set ``gloci=1`` to build the weighting from the minimum of the EM loci curves from all filters.

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
This can be useful if you want the loci weighting to come from only part of the input data.

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

If you already have a DEM-shaped weighting curve, pass it through ``dem_norm0``.

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
This changes the internal weighting and can be useful to compare with the default DEM-space solve.

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

The last axis of ``dn_in`` and ``edn_in`` is always the channel axis.
Leading axes are treated as independent spectra and solved in parallel.
Up to three leading axes are supported (4D input, channel-last).

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

Output leading axes match the input; the channel axis is replaced by temperature bins.

See Also
========

- :doc:`method` — inversion details.
- :doc:`weighting` — weighting options.
- :doc:`generated/gallery/index` — runnable examples.
- :doc:`api` — full function signatures.
