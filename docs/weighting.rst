*****************
Weighting Schemes
*****************

``dn2dem`` supports three main weighting schemes, along with a few related options that change how the inversion is carried out.

See :doc:`api` for the full function signatures and the :doc:`generated/gallery/index` for example scripts.
See :doc:`method` for a broader description of the inversion itself.

Overview
========

The main weighting-related arguments in :func:`demregpy.dn2dem` are:

- ``gloci``: choose between the default self-normalized weighting and EM loci weighting, either for all filters or for a selected subset.
- ``dem_norm0``: supply your own weighting curve directly.
- ``emd_int`` and ``l_emd``: related options that change how the constraint is applied.

Default: Self-Normalized Weighting
==================================

If you do not pass ``dem_norm0``, and if ``gloci`` does not select any filters, :func:`demregpy.dn2dem` uses the default self-normalized weighting.

In this case:

1. A first regularized solve is used to estimate a DEM-like shape.
2. That estimated shape is turned into the weighting used in the main solve.

The weighting is estimated from the inversion itself rather than supplied by the caller.

Use it like this:

.. code-block:: python

   dem, edem, elogt, chisq, dn_reg = dn2dem(
       dn_in,
       edn_in,
       trmatrix,
       tresp_logt,
       temps,
   )

EM Loci Weighting
=================

A Loci curve for a given filter is useful as an upper bound on the emission measure.
The curve is :math:`\mathrm{EM}(T)` for each :math:`T` that would produce the observed data number in that filter if that were the only plasma that was observed.
That is,

.. math::

   \mathrm{EM}(T) = \frac{\mathrm{DN}}{R(T)}

where :math:`R(T)` is the temperature response of the filter.

This means that the EM loci curve of a filter is the absolute maximum possible EM at each temperature that is consistent with the observed data number in that filter.
If the EM were above the loci curve then the instrument would have observed a larger data number than it did.

If you pass ``gloci=1``, the inversion uses the minimum of the EM loci curves from all filters to build the weighting.

You can also pass a length-``nf`` 0/1 mask to use only selected filters.

Use it like this:

.. code-block:: python

   dem, edem, elogt, chisq, dn_reg = dn2dem(
       dn_in,
       edn_in,
       trmatrix,
       tresp_logt,
       temps,
       gloci=1,
   )

or

.. code-block:: python

   dem, edem, elogt, chisq, dn_reg = dn2dem(
       dn_in,
       edn_in,
       trmatrix,
       tresp_logt,
       temps,
       gloci=[1, 1, 0, 0, 1, 1],
   )

Here the weighting has come from the EM loci curves rather than the self-normalized first pass.

User-Supplied Weighting
=======================

If you already have a DEM-shaped weighting curve, you can pass it through ``dem_norm0``.
Only the relative shape matters, not the absolute scale.

Use it like this:

.. code-block:: python

   dem_weight = dem_guess / dem_guess.max()

   dem, edem, elogt, chisq, dn_reg = dn2dem(
       dn_in,
       edn_in,
       trmatrix,
       tresp_logt,
       temps,
       dem_norm0=dem_weight,
   )

Here the weighting comes directly from the supplied shape.
Good choices might be a log-normal curve, or a DEM from a previous solve, or a DEM from a different instrument.

You can provide the same weighting curve for every pixel by passing in ``dem_norm0`` of shape ``(nt,)``, or you can provide a different weighting curve for each pixel by passing an array with the same shape as the output DEM, for example ``(nx, ny, nt)`` for a 2D map.


Related Options
===============

These are not separate weighting schemes, but they change how the weighted problem is solved.

``emd_int=True``
----------------

This performs the internal regularization in EMD space rather than DEM space.
It also enables ``l_emd=True`` internally.

``l_emd=True``
--------------

This changes the diagonal constraint from :math:`\sqrt{\Delta\log T} / \sqrt{w}` to :math:`1 / w`.
That removes the :math:`\sqrt{\Delta\log T}` factor and applies a stronger penalty to bins with low weighting.
This form is used automatically when ``emd_int=True``.

``non_pos=True``
----------------

This disables the positivity-enforcing iterations, meaning there will always only be a single pass.

See Also
========

The following gallery examples show the same ideas in runnable form:

- ``examples/synthetic/plot_synthetic_weighting_modes.py``
- ``examples/synthetic/plot_synthetic_emd_modes.py``
- ``examples/aia/plot_aia_single_pixel.py``
