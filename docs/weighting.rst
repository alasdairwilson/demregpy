*****************
Weighting Schemes
*****************

:func:`demregpy.dn2dem` supports three weighting modes via ``gloci``, ``dem_norm0``, and the default self-normalized path.
See :doc:`method` for the inversion itself and :doc:`api` for full signatures.

Default: Self-Normalized Weighting
==================================

When neither ``dem_norm0`` nor ``gloci`` is specified, ``dn2dem`` runs a two-pass solve: the first pass estimates a DEM shape which is then used as the constraint weighting for the second pass.

Use it by simply passing no weighting, like this:

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

A loci curve for a given filter is the maximum emission measure at each temperature consistent with the observed count in that filter:

.. math::

   \mathrm{EM}(T) = \frac{\mathrm{DN}}{R(T)}

where :math:`R(T)` is the filter response.
The minimum envelope across all channels gives the tightest upper bound on the DEM from the data alone.

If you pass ``gloci=1``, the inversion uses this minimum envelope as the weighting.

For example,

.. code-block:: python

   dem, edem, elogt, chisq, dn_reg = dn2dem(
       dn_in,
       edn_in,
       trmatrix,
       tresp_logt,
       temps,
       gloci=1,
   )

or,

.. code-block:: python

   dem, edem, elogt, chisq, dn_reg = dn2dem(
       dn_in,
       edn_in,
       trmatrix,
       tresp_logt,
       temps,
       gloci=[1, 1, 0, 0, 1, 1],
   )

Here the weighting comes from selected EM loci curves.

User-Supplied Weighting
=======================

Pass a DEM-shaped weighting curve via ``dem_norm0``.
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

Here the weighting comes from the supplied shape.
A log-normal curve, a DEM from a previous solve, or a DEM from a different instrument are all reasonable choices.

A scalar ``(nt,)`` array broadcasts across all pixels; alternatively, pass an array matching the output DEM shape (e.g. ``(nx, ny, nt)``) for per-pixel weighting.


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
