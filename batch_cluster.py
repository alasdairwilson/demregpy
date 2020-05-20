from batch_dem_jp2 import batch_dem_jp2
fits_dir='/home/awilson/code/DEM/demreg-py/demreg-py/test'
jp2_dir='/home/awilson/code/DEM/demreg-py/demreg-py/test'
t_start='2014-01-01 00:00:00.000'
cadence=1
nobs=1
batch_dem_jp2(t_start,cadence,nobs,fits_dir,jp2_dir)
