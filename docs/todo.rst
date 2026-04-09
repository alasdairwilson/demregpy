*****
To Do
*****

This page keeps short notes on follow-up work that would be useful for the docs and public API.

Plotting
========

- Add a helper for plotting DEMograms from time-dependent DEM outputs.
- Add a helper for plotting EM loci curves.
- Add an overlay mode for plotting EM loci curves together with a recovered DEM, so the loci curves can be shown as an upper boundary on the same axes.

AIA Support
===========

- Add a helper for calculating AIA count uncertainties from the standard error terms already used in practice, including shot noise / Poisson terms, dark-current-related terms, and readout or quantization contributions where appropriate.
- Allow that helper to add an optional systematic uncertainty term as a fixed fraction of the data, since a simple fractional error such as 10% is still a common choice in some workflows.
