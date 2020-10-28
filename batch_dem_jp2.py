from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


plt.rcParams['figure.figsize'] = [10, 9]  # make plots larger
from astropy.time import Time, TimeDelta, TimeString
from astropy.visualization import ImageNormalize, SqrtStretch, time_support
from astropy import units as u
from astropy.coordinates import SkyCoord

from datetime import datetime

from sunpy.map import Map
import sunpy.visualization.colormaps as cm
from sunpy.instr.aia import aiaprep
from sunpy.net import Fido, attrs as a
from sunpy.coordinates.sun import B0,angular_radius

import numpy as np

import pprint

from aiapy.calibrate import degradation, register, update_pointing, correct_degradation
from aiapy.calibrate.util import get_correction_table
from aiapy.response import Channel

import warnings

import dateutil.parser
from pandas import read_csv

from dn2dem_pos import dn2dem_pos
from dem2jp2 import dem2jp2
from dem_class import Dem

import pdb
import os
import wget
import requests
# import httplib2
import threadpoolctl
import scipy.io as io

threadpoolctl.threadpool_limits(1)

def batch_dem_jp2(t_start,cadence,nobs,fits_dir,jp2_dir,get_fits=0,serr_per=10,min_snr=2,fe_min=2,use_fe=False,sat_lvl=1.5e4,mk_jp2=False,plot_out=False,plot_loci=False,mk_fits=False,xp=370,yp=750):
    """
    batch script for loading (or downloading) synoptic data from jsoc, setting up the AIA degradation and temperature response etc.
    running demregpy to produce 2-d DEM maps. Finally the code has optional very basic plotting routines and an optional call to
    dem2jp2 to make jpeg2000 greyscale, bytescaled 8 bit images for hv. 


    """
    version_number=1.1
    contact_email='alasdair.wilson@glasgow.ac.uk'
    location='University of Glasgow A+A'
    #we only want optically thin coronal wavelengths
    wavenum=['94','131','171','193','211','335']
    t_start=Time(t_start)
    print(t_start)
    for obs in np.arange(nobs):
        t_obs=(t_start+TimeDelta(obs*cadence,format='sec'))
        print(t_obs)
        t_obs.precision=0
        #convert our string into a datetime object
        # t=dateutil.parser.parse(TimeString(t_obs))
        t=t_obs.to_datetime()

        #deconstruct the datetime object into a synoptica data filename
        file_str=[('AIA'+str(t.year).zfill(4)+str(t.month).zfill(2)+str(t.day).zfill(2)+'_'+str(t.hour).zfill(2)+str(t.minute).zfill(2)+'_'+"{}".format(wave.zfill(4))+'.fits') for j,wave in enumerate(wavenum)]
        dir_str=str(t.year).zfill(4)+'/'+str(t.month).zfill(2)+'/'+str(t.day).zfill(2)+'/'
        syn_url='http://jsoc.stanford.edu/data/aia/synoptic/'
        nf=len(file_str)
        
        cklist = []

        for file in file_str:
            cklist.append(os.path.isfile(fits_dir+dir_str+file))
        if not os.path.isdir(fits_dir+dir_str):
            os.makedirs(fits_dir+dir_str)
        if not all(cklist):
            # not all the files exist.
            print('Downloading synoptic data')
            url=[(syn_url+dir_str+'H'+str(t.hour).zfill(2)+'00'+'/'+file_str[jj]) for jj,c in enumerate(file_str)]
            for jj,c in enumerate(file_str):
                # h = httplib2.Http()
                # open the webpage
                # resp = h.request(url[jj], 'HEAD')
                request = requests.get(url[jj])
                if request.status_code < 400:
                    # webpage exists so download
                    wget.download(url[jj],fits_dir+dir_str)
            cklist=[]
            for file in file_str:
                # check for files again
                cklist.append(os.path.isfile(fits_dir+dir_str+file))
            if not all(cklist):
                #skipping this observation
                print('Missing data for '+str(t)+'...Skipping to next...')
                continue
        
        #find the files in their directory
        fits_files=[fits_dir+dir_str+file_str[j] for j in np.arange(len(file_str))]
        #load the fits with sunpy
        aia = Map(fits_files) 
        correction_table = get_correction_table()
        # correction_table=get_correction_table('aiapy/aiapy/tests/data/aia_V8_20171210_050627_response_table.txt')  
        cal_ver=9
        #correct the images for degradation
        aia = [correct_degradation(m, correction_table=correction_table,calibration_version=cal_ver) for m in aia]
        aia = [update_pointing(m) for m in aia]

        channels = [aia[i].wavelength for i in range(nf)]
    
        nt=28
        t_space=0.05
        t_min=5.8
        logtemps=np.linspace(t_min,t_min+t_space*nt,num=nt+1)
        temperatures=10**logtemps
        logt_bin=np.zeros(nt)
        for i in np.arange(nt):
            logt_bin[i]=(logtemps[i]+logtemps[i+1])/2

        tren=io.readsav('aia_trespv9_en.dat')
        tresp_logt=tren.logt
        tresp_calibrated=np.zeros([tresp_logt.shape[0],nf+1])
        tresp_calibrated[:,:-1]=tren.tr.T

        #initialise structure
        dem=Dem()
        dem.bitpix=8
        nx=aia[0].meta['naxis1']
        ny=aia[0].meta['naxis2']
        dem.naxis1=nx
        dem.naxis2=ny
        dem.crota2=0
        dem.crval1=aia[0].meta['crval1']
        dem.crval2=aia[0].meta['crval2']
        dem.crpix1=512.5
        dem.crpix2=512.5
        dem.cdelt1=aia[0].meta['cdelt1']*4.0
        dem.cdelt2=aia[0].meta['cdelt2']*4.0   
        dem.cunit1=aia[0].meta['cunit1']
        dem.cunit2=aia[0].meta['cunit2']
        dem.dsun_obs=aia[0].meta['dsun_obs']
        dem.crlt_obs=aia[0].meta['crlt_obs']
        dem.crln_obs=aia[0].meta['crln_obs']
        dem.hglt_obs=B0(t_obs).value
        dem.hgln_obs=0
        dem.temperatures=temperatures
        dem.minTemp=logtemps[0]
        dem.maxTemp=logtemps[-1]
        dem.t_obs=t_obs
        dem.filt_use=6
        dem.rsun_ref = 6.957E+08
        dem.rsun_obs = angular_radius(t_obs).value
        dem.hv_zero=np.log10(dem.dem_min)
        dem.hv_scale=(np.log10(dem.dem_max)-np.log10(dem.dem_min))/255
        dem.contact=contact_email
        dem.produced='Produced at '+location+' on: '+datetime.today().strftime('%Y-%m-%d')
        dem.dem_ver=version_number


        dem1=Dem()
        dem1.bitpix=8
        dem1.naxis1=nx
        dem1.naxis2=ny
        dem1.crota2=aia[0].meta['crota2']   
        dem1.crval1=aia[0].meta['crval1']
        dem1.crval2=aia[0].meta['crval2']
        dem1.crpix1=512.5
        dem1.crpix2=512.5
        dem1.cdelt1=aia[0].meta['cdelt1']*4.0
        dem1.cdelt2=aia[0].meta['cdelt2']*4.0
        dem1.cunit1=aia[0].meta['cunit1']
        dem1.cunit2=aia[0].meta['cunit2']
        dem1.dsun_obs=aia[0].meta['dsun_obs']
        dem1.crlt_obs=aia[0].meta['crlt_obs']
        dem1.crln_obs=aia[0].meta['crln_obs']
        dem1.hglt_obs=B0(t_obs).value
        dem1.hgln_obs=0
        dem1.minTemp=logtemps[0]
        dem1.maxTemp=logtemps[-1]
        dem1.filt_use=7
        dem1.rsun_ref = 6.957E+08
        dem1.rsun_obs = np.rad2deg(np.arctan2(dem1.rsun_ref, dem1.dsun_obs))*3600
        dem1.hv_zero=np.log10(dem1.dem_min)
        dem1.hv_scale=(np.log10(dem1.dem_max)-np.log10(dem1.dem_min))/255
        dem1.contact=contact_email
        dem1.produced='Produced at '+location+' on: '+datetime.today().strftime('%Y-%m-%d')
        dem1.dem_ver=version_number
        
        data=np.zeros([nx,ny,nf+1])
       



        #errors in dn/px
        npix=4096.**2/(nx*ny)
        edata=np.zeros([nx,ny,nf+1])
        gains=np.array([18.3,17.6,17.7,18.3,18.3,17.6])
        dn2ph=gains*[94,131,171,193,211,335]/3397.0
        rdnse=1.15*np.sqrt(npix)/npix
        drknse=0.17
        qntnse=0.288819*np.sqrt(npix)/npix
        for f in range(nf):
            #convert from our list to an array of data
            data[:,:,f]=aia[f].data
        for j in np.arange(nf):
            shotnoise=(dn2ph[j]*data[:,:,j])**0.5/dn2ph[j]
            esys=serr_per/100.0*data[:,:,j]
            etemp=np.sqrt(rdnse**2.+drknse**2.+qntnse**2.+shotnoise**2)
            edata[:,:,j]=np.sqrt(esys**2+etemp**2)

        for f in range(nf):
            #convert to values per second
            data[:,:,f]=data[:,:,f]/aia[f].exposure_time.to(u.s).value
            edata[:,:,f]=edata[:,:,f]/aia[f].exposure_time.to(u.s).value

            

        #calculate the hot component of aia 94
        a94_fe18=np.zeros([nx,ny])
        a94_warm=np.zeros([nx,ny])
        a94_fe18[:,:]=data[:,:,0]-data[:,:,4]/120.0-data[:,:,2]/450.0
        a94_warm=data[:,:,0]-a94_fe18[:,:]
        
        #threshold of fe_min for the hot component

        a94_fe18[a94_fe18<=0]=0.0001
        a94_warm[a94_warm<=0]=0.0001
        data[:,:,6]=a94_fe18
    
        #now we need fe18 temp response in a94
        trfe= (tresp_calibrated[:,0]-tresp_calibrated[:,4]/120.0-tresp_calibrated[:,2]/450.0)
        trfe[tresp_logt <= 6.4]=1e-38
        #remove low peak

        tresp_calibrated[:,6]=trfe+1e-3*tresp_calibrated[:,0]
        #errors on fe18 are a little arbitary, percentage error and a flat of 2...
        edata[:,:,6]=serr_per/100*data[:,:,6]+2.0
        plt.rcParams.update({'font.size': 10})
        # dem,edem,elogt,chisq,dn_reg=dn2dem_pos(data[x1:x2,y1:y2,:filt_use],edata[x1:x2,y1:y2,:filt_use],tresp_calibrated[:,:filt_use],tresp_logt,temperatures,dem_norm0=dem_norm0[x1:x2,y1:y2,:],max_iter=10)
        x1=0
        x2=nx
        y1=0
        y2=ny
        filt_use=6
        norm_mean=6.35
        norm_std=0.35
        dem_norm0=np.zeros([nx,ny,nt])
        dem_norm=gaussian(logt_bin,norm_mean,norm_std)
        dem_norm0[:,:,:]=dem_norm[:]
        dem1.data,dem1.edem,dem1.elogt,dem1.chisq,dem1.dn_reg=dn2dem_pos(data[x1:x2,y1:y2,:filt_use],edata[x1:x2,y1:y2,:filt_use],tresp_calibrated[:,:filt_use],tresp_logt,dem.temperatures,max_iter=15,dem_norm0=dem_norm0)
        
        if plot_out==True:
            aia_col=['#c2c3c0','#g0r0r0']
            fig = plt.figure(figsize=(8, 7))
            for j in range(int(nt/2)):
                fig=plt.subplot(4,4,j+1)

                em_loci=data[xp,yp,:]/tresp_calibrated
                plt.errorbar(logt_bin,dem1.data[xp,yp+j*5,:],color='c',xerr=dem1.elogt[xp,yp+j*5,:],yerr=dem1.edem[xp,yp+j*5,:],fmt='or',ecolor='gray', elinewidth=3, capsize=0)
                for i in range(7):
                    em_loci[:-1,i]=em_loci[:-1,i]/(10**tresp_logt[1:]-10**tresp_logt[:-1])
                if plot_loci==True:
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


            fig=plt.figure(figsize=(8, 7))
            for j in range(int(nt/2)):
                fig=plt.subplot(4,4,j+1)
                plt.imshow(np.log10(dem1.data[:,:,2*j]+1),'inferno',vmin=19,vmax=24,origin='lower')
                ax=plt.gca()
                ax.set_title('%.1f'%(t_min+2*j*0.05))
            plt.gcf().suptitle("dem1", fontsize=14)






        if use_fe==True:
            filt_use=7

        dem_norm0=np.zeros([nx,ny,nt])
        mxdem=np.max(dem1.data)
        for ii in np.arange(nx):
            for jj in np.arange(ny):
                dem_norm0[ii,jj,:]=(np.convolve(dem1.data[ii,jj,1:-1],np.ones(5)/5))[1:-1]/mxdem
        dem_norm0[dem_norm0<=1e-12]=1e-12

        if plot_out==True:
            aia_col=['#c2c3c0','#g0r0r0']
            fig = plt.figure(figsize=(8, 7))
            for j in range(int(nt/2)):
                fig=plt.subplot(4,4,j+1)

                em_loci=data[xp,yp,:]/tresp_calibrated
                plt.errorbar(logt_bin,dem_norm0[xp,yp+j*5,:]*mxdem,color='c',xerr=dem1.elogt[xp,yp+j*5,:],yerr=dem1.edem[xp,yp+j*5,:],fmt='or',ecolor='gray', elinewidth=3, capsize=0)
                for i in range(7):
                    em_loci[:-1,i]=em_loci[:-1,i]/(10**tresp_logt[1:]-10**tresp_logt[:-1])
                if plot_loci==True:
                    plt.plot(tresp_logt[:-1],em_loci[:-1,:6])
                ax=plt.gca()
                plt.ylim([1e19,1e23])
                plt.xlim([5.7,7.3])
                plt.xlabel('$\mathrm{\log_{10}T\;[K]}$')
                plt.ylabel('$\mathrm{DEM\;[cm^{-5}\;K^{-1}]}$')
                plt.yscale('log')
                ax.label_outer()
            plt.gcf().suptitle("DEM NORM", fontsize=14)
            plt.gcf().tight_layout(pad=2.0)

        if use_fe==True:
            data[a94_fe18<fe_min,:]=0
        dem.data,dem.edem,dem.elogt,dem.chisq,dem.dn_reg=dn2dem_pos(data[x1:x2,y1:y2,:filt_use],edata[x1:x2,y1:y2,:filt_use],tresp_calibrated[:,:filt_use],tresp_logt,temperatures,dem_norm0=dem_norm0[x1:x2,y1:y2,:],max_iter=15)
        if use_fe==True:
            dem.data[a94_fe18<fe_min,:]=dem1.data[a94_fe18<fe_min,:]
        dem.data[dem.data<=0]=1

        if plot_out==True:
            aia_col=['#c2c3c0','#g0r0r0']
            fig = plt.figure(figsize=(8, 7))
            for j in range(int(np.floor(nt/2))):
                fig=plt.subplot(4,4,j+1)

                em_loci=data[xp,yp,:]/tresp_calibrated
                plt.errorbar(logt_bin,dem.data[xp,yp+j*5,:],color='c',xerr=dem.elogt[xp,yp+j*5,:],yerr=dem.edem[xp,yp+j*5,:],fmt='or',ecolor='gray', elinewidth=3, capsize=0)
                for i in range(7):
                    em_loci[:-1,i]=em_loci[:-1,i]/(10**tresp_logt[1:]-10**tresp_logt[:-1])
                if plot_loci==True:
                    plt.plot(tresp_logt[:-1],em_loci[:-1,:6])
                ax=plt.gca()
                plt.ylim([1e19,1e23])
                plt.xlim([5.7,7.3])
                plt.xlabel('$\mathrm{\log_{10}T\;[K]}$')
                plt.ylabel('$\mathrm{DEM\;[cm^{-5}\;K^{-1}]}$')
                plt.yscale('log')
                ax.label_outer()
            plt.gcf().suptitle("7", fontsize=14)
            plt.gcf().tight_layout(pad=2.0)

            fig=plt.figure(figsize=(8, 7))
            for j in range(int(nt/2)):
                fig=plt.subplot(4,4,j+1)
                plt.imshow(np.log10(dem.data[:,:,j*2]+1),'inferno',vmin=19,vmax=24,origin='lower')
                ax=plt.gca()
                ax.set_title('%.1f'%(t_min+j*2*0.05))
            plt.gcf().suptitle("7", fontsize=14)
            plt.gcf().tight_layout(pad=2.0)


        if plot_out==True:
            aia_col=['#c2c3c0','#g0r0r0']
            fig = plt.figure(figsize=(8, 7))
            for j in range(int(np.floor(nt/2))):
                fig=plt.subplot(4,4,j+1)

                em_loci=data[xp,yp,:]/tresp_calibrated
                plt.errorbar(logt_bin,dem.data[xp,yp+j*5,:],color='c',xerr=dem.elogt[xp,yp+j*5,:],yerr=dem.edem[xp,yp+j*5,:],fmt='or',ecolor='gray', elinewidth=3, capsize=0)
                for i in range(7):
                    em_loci[:-1,i]=em_loci[:-1,i]/(10**tresp_logt[1:]-10**tresp_logt[:-1])
                if plot_loci==True:
                    plt.plot(tresp_logt[:-1],em_loci[:-1,:6])
                ax=plt.gca()
                plt.ylim([1e19,1e23])
                plt.xlim([5.7,7.3])
                plt.xlabel('$\mathrm{\log_{10}T\;[K]}$')
                plt.ylabel('$\mathrm{DEM\;[cm^{-5}\;K^{-1}]}$')
                plt.yscale('log')
                ax.label_outer()
            plt.gcf().suptitle("Combo", fontsize=14)
            plt.gcf().tight_layout(pad=2.0)

            fig=plt.figure(figsize=(8, 7))
            for j in range(int(nt/2)):
                fig=plt.subplot(4,4,j+1)
                plt.imshow(np.log10(dem.data[:,:,j*2]+1),'inferno',vmin=19,vmax=24,origin='lower')
                ax=plt.gca()
                ax.set_title('%.1f'%(t_min+j*2*0.05))
            plt.gcf().suptitle("Combo", fontsize=14)
            plt.gcf().tight_layout(pad=2.0)
            aia[0].peek()
            plt.show()
        dem.nimg=int(np.floor(nt/4))
        if mk_jp2==True:
            if not os.path.isdir(jp2_dir+dir_str):
                os.makedirs(jp2_dir+dir_str)
            for i in range(dem.nimg):
                img_data=(dem.data[:,:,i*2]+dem.data[:,:,i*2+1]+dem.data[:,:,i*2+2]+dem.data[:,:,i*2+3])/4
                jp2_fname=(str(t.year).zfill(4)+'_'+str(t.month).zfill(2)+'_'+str(t.day).zfill(2)+'__'+str(t.hour).zfill(2)+'_'+str(t.minute).zfill(2)+'_'+str(t.second).zfill(2)+'_00'+'__DEM_REGINV_T_'+'%.2f_%.2f'%(logtemps[i*4],logtemps[i*4+4]))
                tmin=logtemps[i*4]
                tmax=logtemps[(i+1)*4]
                print('writing '+jp2_fname + ' img ' + str(i+1) +' of '+str(dem.nimg))
                dem2jp2(img_data,dem,jp2_dir+dir_str+jp2_fname,i,tmin,tmax,mk_fits=mk_fits)
    return dem


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

if __name__ == "__main__":
    fits_dir='/mnt/h/fits/'
    jp2_dir='/mnt/h/data/'
    t_start='2014-01-01 00:02:00.000'
    cadence=60
    nobs=1
    dem=Dem()
    dem=batch_dem_jp2(t_start,cadence,nobs,fits_dir,jp2_dir,fe_min=5,use_fe=False,plot_out=False,plot_loci=True,mk_jp2=True,mk_fits=True)
    pout='dem_saved.pickle'
    # with open(pout,'wb') as f:
    #     pickle.dump(dem, f)
    
    fig=plt.figure()
    ax=plt.gca()
    ims=[]
    im=plt.imshow(np.log10(dem.data[:,:,0]),'inferno',vmin=19.7,vmax=23,origin='lower',animated=True)
    cbar = fig.colorbar(im, ticks=[19.7, 21, 22,23])
    cbar.ax.set_yticklabels(['< 5E19', '1E21','1E22',' > 1E23'])
    cbar.set_label('$cm^{-5}K^{-1}$')

    for i in np.arange(28):
        im=plt.imshow(np.log10(dem.data[:,:,i]),'inferno',vmin=19.7,vmax=24,origin='lower',animated=True)
        ttl = plt.text(0.5, 1.01, t_start+' logT = %.2f'%(5.8+0.05*i), horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
        ims.append([im,ttl])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=500)
    writer = animation.PillowWriter(fps=5)
    ani.save("demo.gif", writer=writer)