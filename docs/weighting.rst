*****************
Weighting Schemes
*****************

``dn2dem`` supports three main weighting schemes, along with a few related options
that change how the inversion is carried out.

For the API details, see :doc:`api`.
For runnable scripts, see the example gallery.

Overview
========

The main entry point is ``demregpy.dn2dem``. The weighting-related arguments are:

- ``gloci``: choose between the default self-normalized weighting and EM loci weighting,
  either for all filters or for a selected subset.
- ``dem_norm0``: supply your own weighting curve directly.
- ``emd_int`` and ``l_emd``: related options that change how the constraint is applied.

Default Self-Normalized Weighting
=================================

If you do not pass either ``gloci=1`` or ``dem_norm0``, and if ``gloci`` does not select
any filters, :func:`demregpy.dn2dem` uses the default self-normalized approach described
in its parameter documentation.

In this case:

1. A first regularized solve is used to estimate a DEM-like shape.
2. That estimated shape is turned into the weighting used in the main solve.

The weighting comes from the inversion itself rather than from an external prior.

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

If you pass ``gloci=1``, the inversion uses the minimum of the EM loci curves from all
filters to build the weighting instead of using the default self-normalized first pass.

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

Here the weighting is built from the minimum of the EM loci curves rather than from
the self-normalized first pass. It is best treated as a different assumption about the
problem, not as a general improvement over the default scheme.

User-Supplied Weighting
=======================

If you already have a DEM shape you want to use as a prior, you can pass it via
``dem_norm0``.
Only the relative shape matters, not the absolute normalization.

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

Here the weighting comes entirely from the supplied shape. Only the relative variation
matters, not the absolute scale.

Related Options
===============

These are not separate weighting schemes, but they interact closely enough with the
weighting to be worth noting here.

``emd_int=True``
----------------

This performs the internal regularization in EMD space rather than DEM space. That
changes the space in which the weighting is applied internally.

``l_emd=True``
--------------

This changes the form of the constraint matrix used in the solve.

``non_pos=True``
----------------

This disables the positivity-enforcing iteration by forcing a single pass. It does not
define a separate weighting scheme by itself, but it does change how the weighted
solution is filtered.

Comparing Modes
===============

The three main weighting paths differ in where the weighting comes from:

- default mode: inferred from an initial self-normalized solve,
- ``gloci=1`` or a 0/1 mask: inferred from EM loci curves,
- ``dem_norm0``: supplied directly by the caller.

In all three cases, the same output API is returned:

- ``dem``
- ``edem``
- ``elogt``
- ``chisq``
- ``dn_reg``

Comparing those outputs across modes is a simple way to see how strongly a result
depends on the weighting assumption.

See Also
========

The following gallery examples show the same ideas in runnable form:

- ``examples/synthetic/plot_synthetic_weighting_modes.py``
- ``examples/synthetic/plot_synthetic_emd_modes.py``
- ``examples/aia/plot_aia_single_pixel.py``
