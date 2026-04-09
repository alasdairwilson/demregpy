********************
How the Method Works
********************

``demregpy`` uses deterministic regularized inversion to recover differential emission measures from multi-channel solar data.
The method follows the approach described by `Hannah and Kontar (2012) <https://doi.org/10.1051/0004-6361/201117576>`_ and the earlier `demreg implementation in IDL <https://github.com/ianan/demreg>`_.

The Forward Problem
===================

Each observed channel measures emission from a broad range of temperatures.
The data in one channel can be written as an integral of the DEM against that channel's temperature response.

.. math::

   g_i = \int K_i(T)\,\xi(T)\,\mathrm{d}T

Here ``g_i`` is the observed count or intensity in channel ``i``, ``K_i(T)`` is the temperature response, and ``\xi(T)`` is the DEM.
After choosing a temperature grid, the problem becomes a matrix equation of the form ``g = K \xi``.

Why the Inversion Is Hard
=========================

This inverse problem is ill-posed.
There are usually only a small number of channels, the temperature responses are broad, and the data contain noise.
A naive inverse can fit the data while producing an unstable or highly oscillatory DEM.

What Regularization Does
========================

Regularization adds a constraint that prefers smooth and stable solutions over noisy ones.
In ``demregpy``, the solution is found by balancing agreement with the observed data within their uncertainties against a weighted constraint on the DEM shape.

The regularization parameter is chosen from a grid of trial values so that the solution is close to the requested target reduced chi-squared.
In practice this means the solver looks for a solution with chi-squared close to the level set by ``reg_tweak``.

What ``dn2dem`` Does
====================

The public :func:`demregpy.dn2dem` wrapper is the primary way to interact with ``demregpy`` and carries out the following steps.

1. It interpolates the response matrix onto the requested temperature grid.
2. It builds the matrix for either a DEM-space solve or an EMD-space solve.
3. It chooses a weighting curve from the default self-normalized solve, from EM loci curves, or from a user-supplied weighting curve.
4. It builds a diagonal constraint matrix from that weighting.
5. It solves the regularized inverse problem using a GSVD-based formulation.
6. It increases the chi-squared target if needed until a non-negative solution is found, unless ``non_pos=True``.
7. It returns the recovered DEM together with reconstructed data and uncertainty estimates.

The lower-level work is done in :func:`demregpy.demmap.demmap` and :func:`demregpy.demmap.dem_pix`.

Weighting and Constraints
=========================

The weighting curve controls where the regularization is stronger or weaker across temperature.
If you do not pass ``dem_norm0``, the default path first computes a rough solution and uses that as the weighting.
If you pass ``gloci=1`` or a 0/1 mask, the weighting is built from the minimum of the selected EM loci curves.
If you pass ``dem_norm0``, that shape is used directly.

The standard diagonal constraint scales like ``sqrt(dlogt) / sqrt(weight)``.
If ``emd_int=True``, the solve is carried out in EMD space and ``l_emd=True`` is enabled internally.
That changes the diagonal constraint to ``1 / weight``.

Positivity
==========

The basic inverse problem is linear, but a purely linear solve can return negative DEM values.
``demregpy`` handles this by repeating the solve with a progressively looser chi-squared target until the solution is non-negative or ``max_iter`` is reached.
If you set ``non_pos=True``, that positivity-enforcing loop is skipped and the first solution is returned.

What the Returned Quantities Mean
=================================

``dem`` is the recovered DEM or EMD, depending on the solve and return options.
``dn_reg`` is the data reconstructed from that solution, a direct way to check how well the inversion reproduces the input counts.
``edem`` is the vertical uncertainty estimate returned by the regularized inverse.
``elogt`` is a temperature-resolution estimate derived from the width of the solver response in temperature space, not an uncertainty on the temperature grid itself.
``chisq`` is the final reduced chi-squared of the reconstructed data.

Relation to Other DEM Methods
=============================

Compared to parametric DEM fitting methods, ``demregpy`` does not begin by assuming that the DEM is a single Gaussian, a small sum of components, or some other fixed functional form.
That makes it less prescriptive; the result is controlled by the data, the response matrix, and the regularization choices rather than by a small set of model parameters.
This also makes it more vulnerable to systematic errors in e.g. in the temperature response functions.

Compared to sampling-based approaches such as `demcmc <https://demcmc.readthedocs.io/en/latest/index.html>`_, ``demregpy`` uses a deterministic GSVD-based solve rather than Monte Carlo sampling.
That makes it computationally fast and practical for lines, maps, and time-dependent data where many DEMs need to be recovered in one run.

This speed comes with a different set of assumptions and inputs.
``demregpy`` requires uncertainties on the input data because the inversion is carried out in a weighted space and the regularization parameter is chosen against a target reduced chi-squared.
In return, the package provides not only a recovered DEM and reconstructed data, but also a vertical uncertainty estimate ``edem`` and a horizontal temperature-resolution estimate ``elogt``.

See Also
========

- :doc:`using_dn2dem` for a worked tutorial.
- :doc:`weighting` for the weighting-related options.
- :doc:`generated/gallery/index` for runnable examples.
