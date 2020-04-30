DEMREG-PY

Readme (along with the rest of the documentation) to be expanded in the future.

This is a python implementation of Hannah & Kontar (2012)'s regularised inversion method. The code is tightly based on the IDL mapping version of the DEM reg-inv code found at https://github.com/ianan/demreg

The philosophy was to produce as similar a piece of software as the original version and as such, this python version has been shown to recover the same DEM as the IDL version (to within approximately 4 significant figures). It is likely this philosophy has lead to performance hits and I plan to go back and address the more hacky parts of the code at a later date.

To use: simply call dn2dem_pos with either a single pixel, 1d slice or 2d map of DN values as a function of filter (and associated error on DN), an array of temperatures over which to perform the DEM analysis and a temperature response for those filters.

For examples on how to use see the "tests.py" file.

For large datasets, beyond 256 pixels, the code switches to a parallel pool based execution using concurrent.futures, providing significant speedups which more than makes up for the loss in performance over the IDL version.

The code is very new, it is likely bugs remain.