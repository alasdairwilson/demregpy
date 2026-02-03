"""
==========================
Using demregpy on AIA data
==========================

This example shows how to use the `demregpy` package to run a Differential Emission Measure (DEM) analysis on AIA data.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
from aiapy.calibrate import correct_degradation, get_correction_table
from astropy import time
from astropy import units as u

from sunpy.map import Map
from sunpy.net import Fido, attrs

from demregpy import dn2dem
from demregpy.tresp import aia_tresp

##############################################################################
# Lets try this on AIA data.

# Load in the SSWIDL generated response functions
# which was produced by make_aiaresp_forpy.pro
trin = io.readsav(aia_tresp)

# Get rid of the byte in the string name
for i in np.arange(len(trin['channels'])):
    trin['channels'][i]=trin['channels'][i].decode("utf-8")

# Get the temperature response functions in the correct form for demreg
tresp_logt=np.array(trin['logt'])
nt=len(tresp_logt)
nf=len(trin['tr'][:])
trmatrix=np.zeros((nt,nf))
for i in range(0,nf):
    trmatrix[:,i]=trin['tr'][i]

# Setup some AIA colours
clrs = ['darkgreen','darkcyan','gold','sienna','indianred','darkslateblue']
# For some DEM model (i.e. a Gaussian) produce the synthetic DN/s/px for each AIA channel
d1=4e22
m1=6.5
s1=0.15
root2pi=(2.*math.pi)**0.5
dem_mod=(d1/(root2pi*s1))*np.exp(-(tresp_logt-m1)**2/(2*s1**2))

time_test = time.Time('2014-01-01T00:00:00', scale='utc')
wavenum=['94','131','171','193','211','335']
nt=16
temperatures=10**np.linspace(5.7,7.1,num=nt+1)
logtemps=np.linspace(5.7,7.1,num=nt+1)
channels = [float(w)*u.angstrom for w in wavenum]
td = time.TimeDelta(11,format='sec')


q=Fido.search(
    attrs.Time(time_test, time_test+td),
    attrs.Instrument('AIA'),
    attrs.Wavelength(channels[0]) | attrs.Wavelength(channels[1]) | attrs.Wavelength(channels[2]) | attrs.Wavelength(channels[3]) | attrs.Wavelength(channels[4]) | attrs.Wavelength(channels[5]),
)
print(q)
files=Fido.fetch(q)

maps = [Map(f) for f in files]
maps = sorted(maps, key=lambda x: x.wavelength)
# maps = [aiaprep(m) for m in maps]
maps = [correct_degradation(m, correction_table=get_correction_table("JSOC")) / m.exposure_time for m in maps]

nx=int(maps[0].dimensions.x.value)
ny=int(maps[0].dimensions.y.value)
nf=len(files)
data=np.zeros([nx,ny,nf])
#convert from our list to an array of data
for j in np.arange(nf):
    data[:,:,j]=maps[j].data
data[data < 0]=0
serr_per=10.0
#errors in dn/px/s
npix=4096.**2/(nx*ny)
edata=np.zeros([nx,ny,nf])
gains=np.array([18.3,17.6,17.7,18.3,18.3,17.6])
dn2ph=gains*[94,131,171,193,211,335]/3397.0
rdnse=1.15*np.sqrt(npix)/npix
drknse=0.17
qntnse=0.288819*np.sqrt(npix)/npix
for j in np.arange(nf):
    etemp=np.sqrt(rdnse**2.+drknse**2.+qntnse**2.+(dn2ph[j]*abs(data[:,:,j]))/(npix*dn2ph[j]**2))
    esys=serr_per*data[:,:,j]/100.
    edata[:,:,j]=np.sqrt(etemp**2. + esys**2.)

y1=2850
y2=3050
x1=1600
x2=1800
mlogt=([np.mean([(np.log10(temperatures[i])),np.log10(temperatures[i+1])]) \
        for i in np.arange(0,len(temperatures)-1)])
demwght0=10**np.interp(mlogt,tresp_logt,np.log10(dem_mod))
demwght0/=max(demwght0)
dem_norm = np.array([[demwght0 for i in range(200)] for j in range(200)])
dem,edem,elogt,chisq,dn_reg=dn2dem(data[x1:x2,y1:y2,:],edata[x1:x2,y1:y2,:],trmatrix,tresp_logt,temperatures, dem_norm0=dem_norm, emd_int=True)

fig=plt.figure(figsize=(16, 16))
for j in range(16):
    fig=plt.subplot(4,4,j+1)
    plt.imshow(np.log10(dem[:,:,j]+1e-20),'inferno',vmin=17,vmax=25,origin='lower')
    ax=plt.gca()
    ax.set_title('%.1f'%(5.6+j*0.1))
    ax.set_yticklabels([])
    ax.set_xticklabels([])

plt.tight_layout()

yr=[2e19,4e23]
xr=[5.7,7.2]
fig = plt.figure(figsize=(8, 4.5))
plt.errorbar(mlogt,dem[94,106,:],xerr=elogt[94,106,:],yerr=edem[94,106,:],fmt='or',\
             ecolor='lightcoral', elinewidth=3, capsize=0)
plt.errorbar(mlogt,dem[100,100,:],xerr=elogt[100,100,:],yerr=edem[94,106,:],fmt='ob',\
             ecolor='lightskyblue', elinewidth=3, capsize=0)
plt.xlabel(r'$\mathrm{\log_{10}T\;[K]}$')
plt.ylabel(r'$\mathrm{DEM\;[cm^{-5}\;K^{-1}]}$')
plt.ylim(yr)
plt.xlim(xr)
plt.rcParams.update({'font.size': 16})
plt.yscale('log')

plt.figure()
plt.imshow(np.sqrt(maps[2].data[x1:x2,y1:y2]),vmax=np.sqrt(4000), cmap=maps[2].cmap, origin='lower')
print(channels)

plt.figure()
maps[2].plot()
maps[0].plot()
print(nt)

plt.show()
