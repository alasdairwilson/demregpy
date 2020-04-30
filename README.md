DEMREG-PY

Readme (along with the rest of the documentation) to be expanded in the future.

This is a python implementation of Hannah & Kontar (2012)'s regularised inversion method. The code is tightly based on the IDL mapping version of the DEM reg-inv code found at https://github.com/ianan/demreg

This python version has been shown to recover the same DEM as the IDL version (to within approximately 4 significant figures).

To use: simply call dn2dem_pos with either a single pixel, 1d slice or 2d map of DN values as a function of filter (and associated error on DN), an array of temperatures over which to perform the DEM analysis and a temperature response for those filters.

For examples on how to use see the "tests.py" file

For large datasets, beyond 256 pixels, the code should switch to a parallel pool based execution using concurrent.futures, providing significant speedups.

The code is very new, it is likely bugs remain.