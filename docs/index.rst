**********************
demregpy Documentation
**********************

``demregpy`` recovers differential emission measures from multi-channel data
using regularized inversion. It works with count arrays and temperature
response matrices, so the same interface can be used for synthetic data, AIA
examples, time series, and small maps.

.. toctree::
   :hidden:
   :maxdepth: 2

   api
   weighting
   generated/gallery/index
   whatsnew/index

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Synthetic Example
      :link: generated/gallery/running_demregpy
      :link-type: doc

      Run ``dn2dem`` on a small synthetic DEM.

   .. grid-item-card:: AIA Example
      :link: generated/gallery/running_demregpy_on_aia_data
      :link-type: doc

      Run one local AIA inversion using the bundled test data.

   .. grid-item-card:: Weighting Schemes
      :link: weighting
      :link-type: doc

      See how ``gloci``, ``dem_norm0``, and related options affect the solve.

   .. grid-item-card:: API Reference
      :link: api
      :link-type: doc

      Reference for the public functions and lower-level routines.

Quick Start
===========

The main entry point is :func:`demregpy.dn2dem`. The core inputs are:

- ``dn_in``: channel counts
- ``edn_in``: uncertainties on those counts
- ``tresp`` and ``tresp_logt``: the temperature response matrix and its log10(T) grid
- ``temps``: temperature-bin edges for the recovered DEM

A minimal call looks like this:

.. code-block:: python

   from demregpy import dn2dem

   dem, edem, elogt, chisq, dn_reg = dn2dem(
       dn_in,
       edn_in,
       tresp,
       tresp_logt,
       temps,
   )

``dem`` is the recovered solution. ``edem`` and ``elogt`` give the vertical and
horizontal uncertainties, ``chisq`` is the final reduced chi-squared, and
``dn_reg`` is the reconstruction in data space.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Full Example Gallery
      :link: generated/gallery/index
      :link-type: doc

      Synthetic data, AIA pixels, patches, and DEMograms.

   .. grid-item-card:: What Changed
      :link: whatsnew/index
      :link-type: doc

      Release notes and changelog entries.

Example Highlights
==================

The gallery uses short runnable examples rather than long scripts.

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: Single Synthetic DEM
      :link: generated/gallery/synthetic/plot_synthetic_single_pixel
      :link-type: doc

      A basic inversion with a known input DEM.

   .. grid-item-card:: Compare Weighting Modes
      :link: generated/gallery/synthetic/plot_synthetic_weighting_modes
      :link-type: doc

      Compare the main weighting paths on one synthetic case.

   .. grid-item-card:: AIA Single Pixel
      :link: generated/gallery/aia/plot_aia_single_pixel
      :link-type: doc

      Recover a DEM from one pixel in the local AIA test maps.

   .. grid-item-card:: AIA Patch
      :link: generated/gallery/aia/plot_aia_patch
      :link-type: doc

      Run the inversion on a small AIA patch.

   .. grid-item-card:: AIA DEMogram
      :link: generated/gallery/aia/plot_aia_demogram
      :link-type: doc

      Build an area-summed DEMogram from a small flare time series.

The AIA examples use bundled local test data.

Reference Guide
===============

- :doc:`api` documents the public API and the lower-level inversion functions.
- :doc:`weighting` explains the main weighting paths and related solver options.
- :doc:`generated/gallery/index` collects the runnable examples.
- :doc:`whatsnew/index` tracks changes across releases.
