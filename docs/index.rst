**********************
demregpy Documentation
**********************

``demregpy`` is a python package for recovering differential emission measures (DEMs), and their associated errors, from multi-channel data solar data using regularized inversion.
It can recover a DEM from any data source where we have data arrays and instrument temperature response matrices, so the same interface can be used for any solar instrument, e.g. AIA, XRT, EIS, RHESSI, etc., in any data shape, single-pixel, 1D, 2D, or timeseries.
The documentation includes a quick start guide, an example gallery, and a complete API reference.

.. toctree::
   :hidden:
   :maxdepth: 2

   api
   installation
   using_dn2dem
   weighting
   generated/gallery/index
   whatsnew/index

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Install ``demregpy`` and the optional AIA-related dependencies.

   .. grid-item-card:: Quick Start
      :link: using_dn2dem
      :link-type: doc

      Tutorial on using demregpy, by recovering DEMs from synthetic data.

   .. grid-item-card:: Example Gallery
      :link: generated/gallery/index
      :link-type: doc

      Gallery of examples for using demregpy.

   .. grid-item-card:: API Reference
      :link: api
      :link-type: doc

      A complete API reference for demregpy.

   .. grid-item-card:: Topic Guides
      :link: weighting
      :link-type: doc

      In depth explanation of aspects of using demregpy.

Release Notes
=============

- :doc:`whatsnew/index` to see what has changed.
