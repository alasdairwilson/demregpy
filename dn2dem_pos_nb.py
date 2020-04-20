import numpy as np
from matplotlib import pyplot as plt
from demmap_pos import demmap_pos
import pprint
import scipy.interpolate
import aiapy
import aiapy.calibrate
from aiapy.calibrate import degradation
from aiapy.calibrate.util import get_correction_table
import aiapy.response
import pandas as pd
import astropy
from astropy import time
import astropy.units as u
from astropy.visualization import time_support



def dn2dem_pos_nb(dn_in,edn_in,tresp,tresp_logt,temps,dem,edem,elogt,chisq,dn_reg,reg_tweak=1.0,max_iter=10,gloci=0,rgt_fact=1.5,dem_norm0=0):
    # Performs a Regularization on solar data, returning the Differential Emission Measure (DEM)
    # using the method of Hannah & Kontar A&A 553 2013
    # Basically getting DEM(T) out of g(f)=K(f,T)#DEM(T)


    #create our bin averages:
    logt=np.log10([np.mean([(temps[i]),(temps[i+1])]) for i in np.arange(0,len(temps)-1)])
    #and widths
    dlogt=(temps[1:]-temps[:-1])
    #number of DEM entries
    nt=len(logt)
    #hopefully we can deal with a variety of data, nx,ny,nf
    sze=dn_in.shape
    #for a single pixel
    if len(sze)==1:
        nx=1
        ny=1
        nf=sze[0]
        dn=np.zeros([1,1,nf])
        dn[0,0,:]=dn_in
        edn=np.zeros([1,1,nf])
        edn[0,0,:]=edn_in
        if ((dem_norm0.ndim) > 0):
            dem0=np.zeros([1,1,nt])
            dem0[0,0,:]=dem_norm0
    #for a row of pixels
    if len(sze)==2:
        nx=sze[0]
        ny=1
        nf=sze[1]
        dn=np.zeros([nx,1,nf])
        dn[:,0,:]=dn_in
        edn=np.zeros([nx,1,nf])
        edn[:,0,:]=edn_in
        if ((dem_norm0.ndim) > 0):
            dem0=np.zeros([nx,1,nt])
            dem0[:,0,:]=dem_norm0
    #for 2d image
    if len(sze)==3:
        nx=sze[0]
        ny=sze[1]
        nf=sze[2]
        dn=np.zeros([nx,ny,nf])
        dn[:,:,:]=dn_in
        edn=np.zeros([nx,ny,nf])
        edn[:,:,:]=edn_in
        if ((dem_norm0.ndim) > 0):
            dem0=np.zeros([nx,ny,nt])
            dem0[:,:,:]=dem_norm0

    glc=np.zeros(nf)
    glc.astype(int)


    if len(tresp[0,:])!=nf:
        print('Tresp needs to be the same number of wavelengths/filters as the data.')
    
    truse=np.zeros([tresp[:,0].shape[0],nf])
    #check the tresp has no elements <0
    #replace any it finds with the mimimum tresp from the same filter
    for i in np.arange(0,nf):
        #keep good TR data
        truse[tresp[:,i] > 0]=tresp[tresp[:,i] > 0]
        #set bad data to the minimum
        truse[tresp[:,i] <= 0]=np.min(tresp[tresp[:,i] > 0])

    tr=np.zeros([nt,nf])
    for i in np.arange(nf):
        f=scipy.interpolate.interp1d(tresp_logt,truse[:,i])
        tr[:,i]=f(logt)
    # fig = plt.figure()
    # ax = fig.gca()
    # for i in np.arange(nf):
    #     plt.plot(logt,np.log10(tr[:,i]))
    # plt.legend(wavenum)
    # plt.show()   

    rmatrix=np.zeros([nt,nf])
    #Put in the 1/K factor (remember doing it in logT not T hence the extra terms)

    for i in np.arange(nf):
        rmatrix[:,i]=tr[:,i]*10.0**logt
    #Just scale so not dealing with tiny numbers
    sclf=1E15
    rmatrix=rmatrix*sclf
    #time it
    t_start = astropy.time.Time.now()


    dn1d=np.reshape(dn,[nx*ny,nf])
    edn1d=np.reshape(edn,[nx*ny,nf])
#create our 1d arrays for output
    dem1d=np.zeros([nx*ny,nt])
    chisq1d=np.zeros([nx*ny])
    edem1d=np.zeros([nx*ny,nt])
    elogt1d=np.zeros([nx*ny,nt])
    dn_reg1d=np.zeros([nx*ny,nf])


# *****************************************************
#  Actually doing the DEM calculations
# *****************************************************
# Do we have an initial DEM guess/constraint to send to demmap_pos as well?
    if ( dem0.ndim==dn.ndim ):
        dem01d=np.reshape(dem0,[nx*ny,nt])
        demmap_pos(dn1d,edn1d,rmatrix,logt,dlogt,glc,dem1d,chisq1d,\
            edem1d,elogt1d,dn_reg1d,reg_tweak=reg_tweak,max_iter=max_iter,\
                rgt_fact=rgt_fact,dem_norm0=dem01d)
    # else:
    #     demmap_pos(dn1d,edn1d,RMatrix,logt,dlogt,glc,dem1d,chisq1d,\
    #         edem1d,elogt1d,dn_reg1d,reg_tweak=reg_tweak,max_iter=max_iter,\
    #             rgt_fact=rgt_fact)
    
    #reshape the 1d arrays to original dimensions and squeeze extra dimensions
    dem=(np.reshape(dem1d,[nx,ny,nt])*sclf).squeeze
    edem=(np.reshape(edem1d,[nx,ny,nt])*sclf).squeeze
    elogt=(np.reshape(elogt1d,[ny,nx,nt])/(2.0*np.sqrt(2.*np.log(2.)))).squeeze
    chisq=(np.reshape(chisq1d,[ny,nx])).squeeze
    dn_reg=(np.reshape(dn_reg1d,[nx,ny,nf])).squeeze
    t_end = astropy.time.Time.now()
    print('total elapsed time =', astropy.time.Time(t_end-t_start,format='datetime'))

    #end the timing

nx=1024
ny=1024
nf=6
nt=16

temperatures=10**np.linspace(5.7,7.3,num=nt+1)
tresp = pd.read_csv("tresp.csv").to_numpy()
# print(tresp_logt.keys())
data=np.zeros([nx,ny,nf])
dem_norm=np.zeros([nx,ny,nt])

# dem_norm[0,0,:]=np.arange(16)
# dem_norm[:,0:100,0]=np.arange(100)
# dem1d=np.reshape(dem_norm,[nx*ny,nt])
# print(dem1d.shape)
# print(dem1d[0,:])
# dem=np.reshape(dem1d,[nx,ny,nt])
# print(dem.shape)
# print(dem[:,0,0])

correction_table = get_correction_table()
wavenum=['94','131','171','193','211','335']
channels = np.zeros(len(wavenum))
for i in np.arange(len(wavenum)):
    channels[i]  = float(wavenum[i])

time_calibration = astropy.time.Time('2012-01-01T00:00:00', scale='utc')

time_test = astropy.time.Time('2020-01-01T00:00:00', scale='utc')

# deg_calibration = {}
deg_calibration = np.zeros([len(channels)])
# deg = {}
deg = np.zeros([len(channels)])

for i,c in enumerate(channels):
    deg_calibration[i] = degradation(c*u.angstrom,time_calibration, correction_table=correction_table)
    deg[i] = degradation(c*u.angstrom,time_test, correction_table=correction_table)

tresp_logt=tresp[:,0]
tresp_calibration=tresp[:,1:]/deg_calibration
trmatrix=tresp_calibration*deg
print(tresp_calibration.shape)

# time_0 = astropy.time.Time('2010-06-01T00:00:00', scale='utc')
# now = astropy.time.Time.now()
# time = time_0 + np.arange(0, (now - time_0).to(u.day).value, 7) * u.day

# deg = {}
# for c in channels:
#     deg[c] = [degradation(c*u.angstrom, t, correction_table=correction_table) for t in time]
# time_support()  # Pass astropy.time.Time directly to matplotlib
# fig = plt.figure()
# ax = fig.gca()
# for i,c in enumerate(channels):
#     ax.plot(time, deg[c])
# ax.set_xlim(time[[0, -1]])
# ax.legend(frameon=False, ncol=4, bbox_to_anchor=(0.5, 1), loc='lower center')
# ax.set_xlabel('Time')
# ax.set_ylabel('Degradation')
# plt.show()

dn2dem_pos_nb(data,1,trmatrix,tresp_logt,temperatures,1,1,1,1,1,dem_norm0=dem_norm)


