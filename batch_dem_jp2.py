from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [10, 9]  # make plots larger
from astropy.time import Time, TimeDelta
from astropy.visualization import ImageNormalize, SqrtStretch, time_support
from sunpy.map import Map
from sunpy.instr.aia import aiaprep
from sunpy.net import Fido, attrs as a
import numpy as np
import pprint
from aiapy.calibrate import degradation, register, update_pointing, correct_degradation
from aiapy.calibrate.util import get_correction_table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import time
from astropy.visualization import ImageNormalize, SqrtStretch, time_support
from aiapy.response import Channel
import warnings
import dateutil.parser
from pandas import read_csv
from dn2dem_pos import dn2dem_pos
import pdb
import os
import wget
import threadpoolctl
from dataclasses import dataclass
threadpoolctl.threadpool_limits(1)

@dataclass
class Dem:
    data:np.ndarray=None
    edem:np.ndarray=None
    elogt:np.ndarray=None
    chisq:np.ndarray=None
    dn_reg:np.ndarray=None
    minTemp:float=None
    maxTemp:float=None
    minC:np.float64=None
    maxC:np.float64=None
    minData:np.float64=None
    maxData:np.float64=None
    imindex:int=None
    nimg:int=None
    filt_use:str=None
    crpix1:float=None
    crpix2:float=None
    crval1:float=None
    crval2:float=None
    crota1:float=None
    crota2:float=None
    naxis1:int=None
    naxis2:int=None
    cdelt1:int=None
    cdelt2:int=None

def batch_dem_jp2(t_start,cadence,nobs,fits_dir,jp2_dir,get_fits=0,serr_per=10,min_snr=2,fe_min=2,sat_lvl=1.5e4):
    
    #we only want optically thin coronal wavelengths
    wavenum=['94','131','171','193','211','335']

    #convert our string into a datetime object
    t=(dateutil.parser.parse(t_start))

    #deconstruct the datetime object into a synoptica data filename
    file_str=[('AIA'+str(t.year).zfill(4)+str(t.month).zfill(2)+str(t.day).zfill(2)+'_'+str(t.hour).zfill(2)+str(t.minute).zfill(2)+'_'+"{}".format(wave.zfill(4))+'.fits') for j,wave in enumerate(wavenum)]
    dir_str=str(t.year).zfill(4)+'/'+str(t.month).zfill(2)+'/'+str(t.day).zfill(2)+'/'
    syn_url='http://jsoc.stanford.edu/data/aia/synoptic/'
   
    
    list1 = []

    for file in file_str:
        list1.append(os.path.isfile(fits_dir+dir_str+file))
    print(list1)
    if not os.path.isdir(fits_dir+dir_str):
        os.makedirs(fits_dir+dir_str)
    if not all(list1):
        # not all the files exist. Run wget.
        print('Downloading synoptic data')
        [print(syn_url+dir_str+'H'+str(t.hour).zfill(4)+'/'+file_str[i]) for i,c in enumerate(file_str)]
        [wget.download(syn_url+dir_str+'H'+str(t.hour).zfill(4)+'/'+file_str[i],fits_dir+dir_str) for i,c in enumerate(file_str)]
    
    #find the files in their directory
    fits_files=[fits_dir+dir_str+file_str[j] for j in np.arange(len(file_str))]
    #load the fits with sunpy
    aia = Map(fits_files) 
    correction_table = get_correction_table()
    #correct the images for degradation
    aia_corrected = [correct_degradation(m, correction_table=correction_table) for m in aia]
    #read dimensions from the header
    # print(aia[0].shape)
    nx=aia[0].meta['naxis1']
    ny=aia[0].meta['naxis2']
    dem=Dem()


    nf=len(file_str)
    channels = [aia[i].wavelength for i in range(nf)]
    nt=16
    logtemps=np.linspace(5.7,7.3,num=nt+1)
    temperatures=10**np.linspace(5.7,7.3,num=nt+1)
    logt_bin=np.zeros(nt)
    for i in np.arange(nt):
        logt_bin[i]=(logtemps[i]+logtemps[i+1])/2
    print(logt_bin)
    tresp = read_csv('tresp.csv').to_numpy()
    time_calibration = time.Time('2014-01-01T00:00:00', scale='utc')
    deg_calibration=np.zeros([nf])
    tresp_calibrated=np.zeros([tresp.shape[0],nf+1])
    for i,c in enumerate(channels):
        deg_calibration[i] = degradation(c,time_calibration, correction_table=correction_table)
    tresp_calibrated[:,:-1]=tresp[:,1:]#/deg_calibration
    print(deg_calibration)
    #create data array
    # print(deg_calibration)
    # fig=plt.figure()
    # for i in range(nf):
    #     ax = fig.add_subplot(2, 1, 1)
    #     plt.plot(np.log10(tresp[:,i+1]))
    #     ax = fig.add_subplot(2, 1, 2)
    #     plt.plot(np.log10(tresp_calibrated[:,i]))
    # plt.show()
    #The following code is heavily AIA 6 channel based, we calculate the iron 18 contribution to the a94 channel and separate it.
    data=np.zeros([nx,ny,nf+1])
    #convert from our list to an array of data
    for j in range(nf):
        data[:,:,j]=aia_corrected[j].data
    #calculate the hot component of aia 94
    a94_fe18=np.zeros([nx,ny])
    a94_fe18[:,:]=data[:,:,0]-data[:,:,4]/120.0-data[:,:,2]/450.0
    #threshold of fe_min for the hot component
    a94_fe18[np.where(a94_fe18 < fe_min)]=0
    data[:,:,6]=a94_fe18
    #now we need fe18 temp response in a94
    trfe=np.zeros([tresp_calibrated.shape[0]])
    # trfe= (tresp_calibrated[:,0]-tresp_calibrated[:,4]/120.-tresp_calibrated[:,2]
    # /450.)
    trfe=tresp_calibrated[:,0]
    trfe[tresp[:,0] < 6.5]=tresp_calibrated[:,0][tresp[:,0] <6.5]*0.001
    #remove low peak

    tresp_calibrated[:,6]=trfe
    #next we do normalisation.
    #std
    norm_std=0.25
    #mean
    norm_mean=6.4
    dem_norm = gaussian(logt_bin,norm_mean,norm_std)
    dem_norm0=np.zeros([nx,ny,nt])
    dem_norm0[:,:,:]=dem_norm
    print(dem_norm)
    tresp_logt=tresp[:,0]

    serr_per=10.0
    #errors in dn/px/s
    npix=4096.**2/(nx*ny)
    edata=np.zeros([nx,ny,nf+1])
    gains=np.array([18.3,17.6,17.7,18.3,18.3,17.6])
    dn2ph=gains*[94,131,171,193,211,335]/3397.0
    rdnse=1.15*np.sqrt(npix)/npix
    drknse=0.17
    qntnse=0.288819*np.sqrt(npix)/npix
    for j in np.arange(nf):
        etemp=np.sqrt(rdnse**2.+drknse**2.+qntnse**2.+(dn2ph[j]*abs(data[:,:,j]))/(npix*dn2ph[j]**2))
        esys=serr_per*data[:,:,j]/100.
        edata[:,:,j]=np.sqrt(etemp**2. + esys**2.)
    #errors on fe18 are trickier...
    edata[:,:,6]=serr_per/100*data[:,:,6]+2.0
    #from here we have our datacube,errors,tresp and normalisation so we can call dn2dem

    print((np.argmax(data[:,:,6])-np.mod(np.argmax(data[:,:,6]),nx))/nx,np.mod(np.argmax(data[:,:,6]),nx))
    # fig=plt.figure()
    # plt.imshow(np.log(data[:,:,6]),origin='lower')
    # plt.show()

    # aia_corrected[0].peek()


    plt.rcParams.update({'font.size': 10})
    # dem,edem,elogt,chisq,dn_reg=dn2dem_pos(data[x1:x2,y1:y2,:filt_use],edata[x1:x2,y1:y2,:filt_use],tresp_calibrated[:,:filt_use],tresp_logt,temperatures,dem_norm0=dem_norm0[x1:x2,y1:y2,:],max_iter=10)
    x1=0
    x2=nx
    y1=0
    y2=ny

    # for j in range(4):
    #     fig=plt.subplot(2,2,j+1)
    #     plt.errorbar(logt_bin,dem[100,85+5*j,:],color=c,xerr=elogt[100,85+5*j,:],yerr=edem[100,85+5*j,:],fmt='or',ecolor='gray', elinewidth=3, capsize=0)
    #     ax=plt.gca()
    #     ax.set_title(str(j))
    #     plt.ylim([1e19,1e23])
    #     plt.xlim([5.7,7.3])
    #     plt.xlabel('$\mathrm{\log_{10}T\;[K]}$')
    #     plt.ylabel('$\mathrm{DEM\;[cm^{-5}\;K^{-1}]}$')
    #     plt.yscale('log')

    # plt.gcf().suptitle("7 filt", fontsize=14)



    # fig=plt.figure(figsize=(8, 7))
    # for j in range(4):
    #     fig=plt.subplot(2,2,j+1)
    #     plt.imshow(np.log10(dem[:,:,j*3]),vmin=19,vmax=24,origin='lower')
    #     ax=plt.gca()
    #     ax.set_title(str(j))
    # plt.gcf().suptitle("7 filt", fontsize=14)

    dem.data=np.zeros([nx,ny,nt])

    filt_use=6
    dem.data,dem.edem,dem.elogt,dem.chisq,dem.dn_reg=dn2dem_pos(data[x1:x2,y1:y2,:filt_use],edata[x1:x2,y1:y2,:filt_use],tresp_calibrated[:,:filt_use],tresp_logt,temperatures,dem_norm0=dem_norm0[x1:x2,y1:y2,:],max_iter=15)
    fig = plt.figure(figsize=(8, 7))
    
    for j in range(6):
        fig=plt.subplot(2,3,j+1)
        xp=400
        yp=485+j*5
        em_loci=data[xp,yp,:]/tresp_calibrated
        plt.errorbar(logt_bin,dem.data[xp,yp,:],color=c,xerr=dem.elogt[xp,yp,:],yerr=dem.edem[xp,yp,:],fmt='or',ecolor='gray', elinewidth=3, capsize=0)
        for i in range(6):
            em_loci[:-1,i]=em_loci[:-1,i]/(10**tresp_logt[1:]-10**tresp_logt[:-1])
        plt.plot(tresp_logt[:-1],em_loci[:-1,:6])
        ax=plt.gca()
        plt.ylim([1e19,1e23])
        plt.xlim([5.7,7.3])
        plt.xlabel('$\mathrm{\log_{10}T\;[K]}$')
        plt.ylabel('$\mathrm{DEM\;[cm^{-5}\;K^{-1}]}$')
        plt.yscale('log')
        ax.label_outer()
    plt.gcf().suptitle("6 Filter", fontsize=14)
    plt.gcf().tight_layout(pad=2.0)
    print(em_loci)
    # fig = plt.figure(figsize=(8, 7))
    # for j in range(6):
    #     fig=plt.subplot(2,3,j+1)
    #     plt.errorbar(logt_bin,dem.data[400,385+5*j,:],color=c,xerr=elogt[400,385+5*j,:],yerr=edem[400,385+5*j,:],fmt='or',ecolor='gray', elinewidth=3, capsize=0)
    #     ax=plt.gca()
    #     ax.set_title('%.1f'%(j))
    #     plt.ylim([1e19,1e23])
    #     plt.xlim([5.7,7.3])
    #     plt.xlabel('$\mathrm{\log_{10}T\;[K]}$')
    #     plt.ylabel('$\mathrm{DEM\;[cm^{-5}\;K^{-1}]}$')
    #     plt.yscale('log')
    #     ax.label_outer()
    # plt.gcf().suptitle("6 filt", fontsize=14)
    # plt.gcf().tight_layout(pad=2.0)



    # fig=plt.figure(figsize=(8, 7))
    # for j in range(6):
    #     fig=plt.subplot(2,3,j+1)
    #     plt.imshow(np.log10(dem.data[:,:,j*3]),'inferno',vmin=19,vmax=24,origin='lower')
    #     ax=plt.gca()
    #     ax.set_title('%.1f'%(5.7+j*3*0.1))
    # plt.gcf().suptitle("6 filt", fontsize=14)

    # filt_use=7
    # dirt_fact=1E-3
    # data[:,:,6]+=dirt_fact*data[:,:,0]
    # tresp_calibrated[:,6]=trfe+dirt_fact*tresp_calibrated[:,0]
    # data[a94_fe18 < fe_min,:] = 0
    # #std
    # norm_std=0.3
    # #mean
    # norm_mean=6.35
    # dem_norm = gaussian(logt_bin,norm_mean,norm_std)
    # dem_norm0[:,:,:]=dem_norm  
    

    # dem2,edem,elogt,chisq,dn_reg=dn2dem_pos(data[x1:x2,y1:y2,:filt_use],edata[x1:x2,y1:y2,:filt_use],tresp_calibrated[:,:filt_use],tresp_logt,temperatures,dem_norm0=dem_norm0[x1:x2,y1:y2,:],max_iter=20)
    # dem2[a94_fe18<fe_min]=0
    # dem1[dem1<0]=0
    # dem2[dem2<0]=0
    


  

    # fig = plt.figure(figsize=(8, 7))
    # for j in range(6):
    #     fig=plt.subplot(2,3,j+1)
    #     plt.errorbar(logt_bin,dem.data[400,485+5*j,:],color=c,xerr=elogt[400,485+5*j,:],yerr=edem[400,485+5*j,:],fmt='or',ecolor='gray', elinewidth=3, capsize=0)
    #     ax=plt.gca()
    #     ax.set_title('%.1f'%(5.7+j*3*0.1))
    #     plt.ylim([1e19,1e23])
    #     plt.xlim([5.7,7.3])
    #     plt.xlabel('$\mathrm{\log_{10}T\;[K]}$')
    #     plt.ylabel('$\mathrm{DEM\;[cm^{-5}\;K^{-1}]}$')
    #     plt.yscale('log')
    #     ax.label_outer()
    # plt.gcf().suptitle("combo", fontsize=14)
    # plt.gcf().tight_layout(pad=2.0)
    
    # fig=plt.figure(figsize=(8, 7))
    # for j in range(6):
    #     fig=plt.subplot(2,3,j+1)
    #     plt.imshow(np.log10(dem1[:,:,j*3]+1),'inferno',vmin=19,vmax=24,origin='lower')
    #     ax=plt.gca()
    #     ax.set_title('%.1f'%(5.7+j*3*0.1))
    #     plt.gcf().suptitle("6", fontsize=14)

    # fig=plt.figure(figsize=(8, 7))
    # for j in range(6):
    #     fig=plt.subplot(2,3,j+1)
    #     plt.imshow(np.log10(dem2[:,:,j*3]+1),'inferno',vmin=19,vmax=24,origin='lower')
    #     ax=plt.gca()
    #     ax.set_title('%.1f'%(5.7+j*3*0.1))
    # plt.gcf().suptitle("dirty", fontsize=14)


    fig=plt.figure(figsize=(8, 7))
    for j in range(6):
        fig=plt.subplot(2,3,j+1)
        plt.imshow(np.log10(dem.data[:,:,j*3]+1),'inferno',vmin=19,vmax=24,origin='lower')
        ax=plt.gca()
        ax.set_title('%.1f'%(5.7+j*3*0.1))
    plt.gcf().suptitle("combo", fontsize=14)

    fig=plt.figure(figsize=(8, 7))
    plt.imshow(np.sqrt(dem.data[:,:,2]+1),'viridis',origin='lower')
    ax=plt.gca()
    ax.set_title('%.1f'%(5.7+2*0.1))


    plt.show()
    return dem
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

if __name__ == "__main__":
    fits_dir='/mnt/c/Users/Alasdair/Documents/reginvpy/test/'
    jp2_dir='/mnt/c/Alasdair/Documents/reginvpy/test/'
    t_start='2014-05-10 00:00:00.000'
    cadence=1
    nobs=1
    dem=batch_dem_jp2(t_start,cadence,nobs,fits_dir,jp2_dir,fe_min=20.0)

    # plt.show()
    # fig = plt.figure(figsize=(3*5,6*5))
    # for i, (m, mc) in enumerate(zip(aia, aia_corrected)):
    #     norm = ImageNormalize(vmin=0,vmax=np.mean(mc.data)*10,stretch=SqrtStretch())
    #     ax = fig.add_subplot(2, len(aia), i+1, projection=m)
    #     m.plot(axes=ax, norm=norm, annotate=False)
    #     ax.set_title(m.wavelength)
    #     ax.coords[0].set_ticks_visible(False)
    #     ax.coords[0].set_ticklabel_visible(False)
    #     ax.coords[1].set_ticks_visible(False)
    #     ax.coords[1].set_ticklabel_visible(False)
    #     ax = fig.add_subplot(2, len(aia), i+1+len(aia), projection=mc)
    #     mc.plot(axes=ax, norm=norm, annotate=False,)
    #     ax.coords[0].set_ticks_visible(False)
    #     ax.coords[0].set_ticklabel_visible(False)
    #     ax.coords[1].set_ticks_visible(False)
    #     ax.coords[1].set_ticklabel_visible(False)
    # plt.show()
    


