**********************
Regularised Inversion
**********************

``demregpy`` uses deterministic regularised inversion to recover differential emission measures from multi-channel solar data.
The method follows the approach described by `Hannah and Kontar (2012) <https://doi.org/10.1051/0004-6361/201117576>`_ and the earlier `demreg implementation in IDL <https://github.com/ianan/demreg>`_.

The Forward Problem
===================

Each observed channel measures emission from a broad range of temperatures.
The data in one channel can be written as an integral of the DEM against that channel's temperature response.

.. math::

   g_i = \int K_i(T)\,\xi(T)\,\mathrm{d}T

Here :math:`g_i` is the observed count or intensity in channel :math:`i`, :math:`K_i(T)` is the temperature response, and :math:`\xi(T)` is the DEM.
After choosing a temperature grid, the problem becomes a matrix equation of the form :math:`g = K\xi`.

Ill-Posedness
=============

The inverse problem is ill-posed: few channels, broad responses, and noise mean that a naive inverse produces unstable or oscillatory solutions.

Regularization
==============

Regularization balances data fidelity against a weighted smoothness constraint on the DEM.
The regularization parameter is chosen from a grid of trial values to achieve a target :math:`\chi_\nu^2`, set by ``reg_tweak``.

Algorithm
=========

:func:`demregpy.dn2dem` carries out the following steps:

1. Interpolate the response matrix onto the requested temperature grid.
2. Build the forward operator in DEM or EMD space.
3. Construct a diagonal constraint matrix from the chosen weighting (self-normalized, EM loci, or user-supplied).
4. Solve the regularised inverse via GSVD.
5. If needed, increase the :math:`\chi_\nu^2` target and re-solve until the DEM is non-negative (unless ``non_pos=True``).
6. Return the DEM, uncertainties, and reconstructed counts.

The lower-level work is done in :func:`demregpy.demmap.demmap` and :func:`demregpy.demmap.dem_pix`.

Weighting and Constraints
=========================

The weighting curve controls where the regularization is stronger or weaker across temperature.
If you do not pass ``dem_norm0``, the default path first computes a rough solution and uses that as the weighting.
If you pass ``gloci=1`` or a 0/1 mask, the weighting is built from the minimum of the selected EM loci curves.
If you pass ``dem_norm0``, that shape is used directly.

The standard diagonal constraint scales like :math:`\sqrt{\Delta\log T} / \sqrt{w}`.
If ``emd_int=True``, the solve is carried out in EMD space and ``l_emd=True`` is enabled internally.
That changes the diagonal constraint to :math:`1 / w`.

Positivity
==========

The basic inverse problem is linear, but a purely linear solve can return negative DEM values.
``demregpy`` handles this by repeating the solve with a progressively looser :math:`\chi_\nu^2` target until the solution is non-negative or ``max_iter`` is reached.
If you set ``non_pos=True``, that positivity-enforcing loop is skipped and the first solution is returned.

Returned Quantities
===================

``dem`` is the recovered DEM or EMD, depending on the solve and return options.
``dn_reg`` is the data reconstructed from that solution, a direct way to check how well the inversion reproduces the input counts.
``edem`` is the vertical uncertainty estimate returned by the regularised inverse.
``elogt`` is a temperature-resolution estimate derived from the width of the solver response in temperature space, not an uncertainty on the temperature grid itself.
``chisq`` is the final reduced chi-squared, :math:`\chi_\nu^2`, of the reconstructed data.

Relation to Other DEM Methods
=============================

Unlike parametric fitting, ``demregpy`` does not assume a fixed functional form for the DEM, making results data-driven but more sensitive to response-matrix errors.

Unlike sampling-based approaches (e.g. `demcmc <https://demcmc.readthedocs.io/en/latest/index.html>`_), ``demregpy`` uses a deterministic GSVD solve, enabling rapid batch processing of maps and time series.

``demregpy`` requires uncertainties on the input data because the solve is carried out in a weighted space and the regularization parameter is chosen against a target :math:`\chi_\nu^2`.

See Also
========

- :doc:`using_dn2dem` for a worked tutorial.
- :doc:`weighting` for the weighting-related options.
- :doc:`generated/gallery/index` for runnable examples.
