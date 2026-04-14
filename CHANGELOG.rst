1.0.0 (2026-04-14)
==================

Breaking Changes
----------------

- Updated the minimum supported dependencies for ``demregpy`` 1.0.
  ``demregpy`` now targets Python 3.12 and newer, uses NumPy 2 or newer, and adds matplotlib as a core dependency. (`#21 <https://github.com/alasdairwilson/demregpy/pull/21>`__)


New Features
------------

- Extended :func:`demregpy.dn2dem` to accept channel-last inputs with up to three leading axes, including stacks such as ``(ntime, nx, ny, nf)``.
  The weighting inputs are also more flexible, including support for per-filter ``gloci`` masks and broadcasting a single ``dem_norm0`` curve across larger inputs, and invalid shapes or response matrices now raise explicit errors earlier in the solve. (`#21 <https://github.com/alasdairwilson/demregpy/pull/21>`__)
- Added small public helper APIs for common workflows, including :func:`demregpy.load_aia_response` for bundled AIA response files, :func:`demregpy.plotting.plot_dem` for one-dimensional DEM plots, and :func:`demregpy.synthetic.synthesize_counts` for generating synthetic observations from response matrices. (`#21 <https://github.com/alasdairwilson/demregpy/pull/21>`__)
- Improved inversion performance substantially, with typical speedups of around 4 to 7 times depending on the input shape and solver settings. (`#21 <https://github.com/alasdairwilson/demregpy/pull/21>`__)
- Added :func:`demregpy.plotting.plot_loci_curves` for plotting filter loci curves on their own or overplotting them on a DEM axis.
  The helper supports bundled AIA responses and synthetic response matrices, and uses the same DEM-space scaling as :func:`demregpy.dn2dem` by default so loci curves can be compared directly with recovered DEMs. (`#22 <https://github.com/alasdairwilson/demregpy/pull/22>`__)


Documentation
-------------

- Reworked the documentation and example gallery around smaller, focused examples instead of a few monolithic scripts.
  The updated docs now cover installation, a ``dn2dem`` tutorial, topic guides, AIA response tooling, synthetic-count generation, and a wider range of runnable gallery examples. (`#21 <https://github.com/alasdairwilson/demregpy/pull/21>`__)
- Added synthetic and AIA gallery examples showing how to plot loci curves and how to overlay them on recovered DEMs. (`#22 <https://github.com/alasdairwilson/demregpy/pull/22>`__)
