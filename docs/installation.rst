************
Installation
************

Install ``demregpy`` from PyPI with ``pip``.

Basic Install
=============

For the core package:

.. code-block:: bash

   pip install demregpy

This installs `demregpy` along with the core dependencies.
After installation, :func:`demregpy.dn2dem` is the main entry point for converting data numbers to differential emission measures.

Install with AIA Support
========================

If you want to use the bundled AIA response loader and the AIA examples, install the ``aia`` extra:

.. code-block:: bash

   pip install "demregpy[aia]"

This adds the optional dependencies used for reading AIA response files and SunPy maps.

Install from a Local Checkout
=============================

If you are working from a local clone:

.. code-block:: bash

   pip install -e .

For a local editable install with the AIA-related dependencies:

.. code-block:: bash

   pip install -e ".[aia]"

Check the Install
=================

You can check that the package imports with:

.. code-block:: bash

   python -c "from demregpy import dn2dem; print(dn2dem.__name__)"

Next Steps
==========

- :doc:`using_dn2dem` for a worked tutorial
- :doc:`generated/gallery/index` for runnable examples
- :doc:`api` for the full API reference
