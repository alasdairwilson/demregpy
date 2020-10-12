from batch_dem_jp2 import batch_dem_jp2
fits_dir='/scratch/butterfly/awilson/demreg-py/fits/'
jp2_dir='/scratch/butterfly/awilson/demreg-py/test/'
t_start='2013-12-31 12:00:00.000'
cadence=60*60
nobs=25
dem=batch_dem_jp2(t_start,cadence,nobs,fits_dir,jp2_dir,fe_min=5,plot_out=False,plot_loci=False,mk_jp2=True)