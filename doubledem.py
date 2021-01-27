from dn2dem_pos_selfnorm import dn2dem_pos_selfnorm

import numpy as np
from matplotlib import pyplot as plt
from demmap_pos import demmap_pos
from dn2dem_pos import dn2dem_pos
import scipy.interpolate
from aiapy.calibrate import degradation, register, update_pointing
from aiapy.calibrate.util import get_correction_table
import aiapy.response
from astropy import time
import astropy.units as u
from astropy.units import imperial
from astropy.visualization import time_support
from pandas import read_csv
import os
from sunpy.net import Fido, attrs
from sunpy.map import Map
from sunpy.instr.aia import aiaprep
import dateutil.parser
import cProfile
import pstats
from io import StringIO
import threadpoolctl
threadpoolctl.threadpool_limits(1)

nt=14
fits_dir='/mnt/c/Users/Alasdair/Documents/reginvpy/test/'
os.chdir('/mnt/c/Users/Alasdair/Documents/reginvpy')
# os.chdir('C:/Users/Alasdair/Documents/reginvpy')
# fits_dir="C:/Users/Alasdair/Documents/reginvpy/test/"
# fits_dir='/home/awilson/code/DEM/demreg-py/demreg-py/test/'
# os.chdir('/home/awilson/code/DEM/demreg-py/demreg-py/')
temperatures=10**np.linspace(5.7,7.1,num=nt+1)
logtemps=np.linspace(5.7,7.1,num=nt+1)
tresp = read_csv('tresp.csv').to_numpy()
# print(tresp_logt.keys())
# data=np.ones([nx,ny,nf])
# edata=np.ones([nx,ny,nf])/10
# dem_norm=np.ones([nx,ny,nt])
data=np.array([3.4,13.8,184,338,219.55,12.22])
edata=np.array([0.2,0.43,7.83,12.9,5.80,0.23])
dem_norm=np.array([ 0.082588151,0.18005607,0.30832890,0.47582966, 0.66201794,0.83059740,0.93994260,0.95951378 ,0.88358527,0.73393929, 0.54981130, 0.37136465,0.22609001 , 0.11025056])


correction_table = get_correction_table()
wavenum=['94','131','171','193','211','335']
channels = []
for i in np.arange(len(wavenum)):
    channels.append(float(wavenum[i])*u.angstrom)

time_calibration = time.Time('2014-01-01T00:00:00', scale='utc')

time_test = time.Time('2014-01-01T00:00:00', scale='utc')

# deg_calibration = {}
deg_calibration = np.zeros([len(channels)])
# deg = {}
deg = np.zeros([len(channels)])

for i,c in enumerate(channels):
    deg_calibration[i] = degradation(c,time_calibration, correction_table=correction_table)
    deg[i] = degradation(c,time_test, correction_table=correction_table)

tresp_logt=tresp[:,0]
tresp_calibration=tresp[:,1:]/deg_calibration
trmatrix=deg[:]*tresp_calibration
dem,edem,elogt,chisq,dn_reg=dn2dem_pos_selfnorm(data,edata,trmatrix,tresp_logt,temperatures,max_iter=50)
data1d=np.zeros([10,6])
edata1d=np.zeros([10,6])
for ii in range(10):
    data1d[ii,:]=data[:]
    edata1d[ii,:]=edata[:]
dem,edem,elogt,chisq,dn_reg=dn2dem_pos_selfnorm(data1d,edata1d,trmatrix,tresp_logt,temperatures,max_iter=50)
data2d=np.zeros([100,10,6])
edata2d=np.zeros([100,10,6])
for ii in range(100):
    data2d[ii,:,:]=data1d[:]
    edata2d[ii,:,:]=edata1d[:]
dem,edem,elogt,chisq,dn_reg=dn2dem_pos_selfnorm(data2d,edata2d,trmatrix,tresp_logt,temperatures,max_iter=50)