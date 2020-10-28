import numpy as np
import glymur
from lxml import etree as ET
from dataclasses import dataclass
from skimage.util import img_as_ubyte
from dem_class import Dem
import os
from datetime import datetime
from sunpy import io
def dem2jp2(img_data,dem,fname,i,bin_min,bin_max,mk_fits=False):
    #log 10 the data
    datasc=np.log10(img_data+1)
    #take logs of our data range
    logmin=np.log10(dem.dem_min)
    logmax=np.log10(dem.dem_max)
    #floor the data to datamin
    datasc[datasc < logmin]=logmin
    #ceiling the data to datamax
    datasc[datasc > logmax]=logmax
    #scale the data between 0-1
    datasc = ((datasc)-logmin)/(logmax-logmin)
    #convert to unsigned byte
    bytesc=img_as_ubyte(datasc)
    #create the jp2 file
    jp2=glymur.Jp2k(fname+'.jp2',np.flipud(bytesc))
    #create the header
    demxml(dem,fname,i,bin_min,bin_max)
    #load the header
    xmlbox = glymur.jp2box.XMLBox(filename=fname+'.xml')

    jp2.append(xmlbox)
    os.remove(fname+'.xml')
    if mk_fits==True:
        #create the fits
        #dictionary 
        demdict=dem.__dict__.copy()
        #remove keys
        del demdict['data']
        del demdict['edem']
        del demdict['elogt']
        del demdict['temperatures']
        del demdict['chisq']
        del demdict['dn_reg']
        demdict['trange'] = '{:1}-{:1}'.format(dem.minTemp,dem.maxTemp)
        demdict["trangek"] = '{:.2e}-{:.2e}'.format(10**dem.minTemp,10**dem.maxTemp)
        demdict["t_bin"]='{:.2e}-{:.2e}'.format(bin_min,bin_max)
        demdict["DATE-OBS"] = dem.t_obs.isot
        demdict['t_obs']=dem.t_obs.isot+'Z'
        demdict["img_id"] = '{} of {}'.format(i+1,dem.nimg)  
        demdict["img_sc"]='LOG10'
        demdict['github'] ='https://github.com/alasdairwilson/demreg-py'
        demdict["qnty"]='DEM(T)'
        demdict["method"]="Regularised Inversion (Hannah and Kontar 2012)"
        demdict["dem_unit"] = 'cm-5 K-1'
        demdict["contact"] ='alasdair.wilson@glasgow.ac.uk'
        demdict["detector"] = 'demregpy'
        demdict["ctype1"] = 'HPLN-TAN'
        demdict["ctype2"] = 'HPLT-TAN'
        demdict["TELESCOP"] = 'derived-DEM'  
        demdict["instrume"] = 'LOGT {:.2f}-{:.2f}'.format(bin_min,bin_max)
        demdict["filters"] ='AIA94 AIA131 AIA171 AIA193 AIA211 AIA335'
        if os.path.isfile(fname+'.fits'):
            os.remove(fname+'.fits')
        io.fits.write(fname+'.fits',bytesc,demdict)   
        #add the header to the jp2 file.
    return

def demxml(dem,fname,i,bin_min,bin_max):
    metax = ET.Element("meta")
    fitsx = ET.SubElement(metax,"fits")
    heliox = ET.SubElement(metax, "helioviewer")
    derivex = ET.SubElement(heliox,"derived_data")
    
    # ET.SubElement(fitsx, "BUNIT").text = 'LOG 10 cm^-5 K^-1'
    # ET.SubElement(fitsx, "HV_ZERO").text = '{:.2f}'.format(np.log10(dem.dem_min))
    # ET.SubElement(fitsx, "HV_SCALE").text = '{:.2f}'.format((np.log10(dem.dem_max)-np.log10(dem.dem_min))/255)
    ET.SubElement(fitsx, "BITPIX").text = '{:}'.format(dem.bitpix)
    ET.SubElement(fitsx, "OBSERVATORY").text = 'derived-data'
    ET.SubElement(fitsx, "INSTRUMENT").text = 'DEM'
    ET.SubElement(fitsx, "DETECTOR").text = 'demregpy'
    ET.SubElement(fitsx, "TELESCOP").text = 'derived-DEM'
    ET.SubElement(fitsx, "INSTRUME").text = 'LOGT {:.2f}-{:.2f}'.format(bin_min,bin_max)
    ET.SubElement(fitsx, "T_OBS").text = dem.t_obs.isot+'Z'
    ET.SubElement(fitsx, "DATE-OBS").text = dem.t_obs.isot
    ET.SubElement(fitsx, "NAXIS").text = '2'
    ET.SubElement(fitsx, "NAXIS1").text = '{:}'.format(dem.naxis1)
    ET.SubElement(fitsx, "NAXIS2").text = '{:}'.format(dem.naxis2)
    ET.SubElement(fitsx, "CDELT1").text = '{:}'.format(dem.cdelt1)
    ET.SubElement(fitsx, "CDELT2").text = '{:}'.format(dem.cdelt2)
    ET.SubElement(fitsx, "CTYPE1").text = 'HPLN-TAN'
    ET.SubElement(fitsx, "CTYPE2").text = 'HPLT-TAN'   
    ET.SubElement(fitsx, "CRPIX1").text = '{:}'.format(dem.crpix1)
    ET.SubElement(fitsx, "CRPIX2").text = '{:}'.format(dem.crpix2)
    ET.SubElement(fitsx, "CUNIT1").text = '{:}'.format(dem.cunit1)
    ET.SubElement(fitsx, "CUNIT2").text = '{:}'.format(dem.cunit2)
    ET.SubElement(fitsx, "CRVAL1").text = '{:}'.format(dem.crval1)
    ET.SubElement(fitsx, "CRVAL2").text = '{:}'.format(dem.crval2)    
    ET.SubElement(fitsx, "CROTA2").text = '{:}'.format(dem.crota2)
    ET.SubElement(fitsx, "DSUN_OBS").text = '{:.3e}'.format(dem.dsun_obs)
    ET.SubElement(fitsx, "RSUN_REF").text = '{:.3e}'.format(dem.rsun_ref)
    ET.SubElement(fitsx, "RSUN_OBS").text = '{:.3e}'.format(dem.rsun_obs)
    ET.SubElement(fitsx, "CRLT_OBS").text = '{:.3e}'.format(dem.crlt_obs)
    ET.SubElement(fitsx, "CRLN_OBS").text = '{:.3e}'.format(dem.crln_obs)
    ET.SubElement(fitsx, "HGLT_OBS").text = '{:}'.format(dem.hglt_obs)
    ET.SubElement(fitsx, "HGLN_OBS").text = '{:}'.format(dem.hgln_obs)
    dem.hgln_obs=0



    ET.SubElement(derivex,"qnty").text = "DEM(T)"
    ET.SubElement(derivex,"method").text = "Regularised Inversion (Hannah and Kontar 2012)"
    ET.SubElement(derivex,"filters").text = 'AIA94 AIA131 AIA171 AIA193 AIA211 AIA335'
    ET.SubElement(derivex,"trange").text = '{:1}-{:1}'.format(dem.minTemp,dem.maxTemp)
    ET.SubElement(derivex,"trangek").text = '{:.2e}-{:.2e}'.format(10**dem.minTemp,10**dem.maxTemp)
    ET.SubElement(derivex,"t_bin").text = '{:.2e}-{:.2e}'.format(bin_min,bin_max)
    ET.SubElement(derivex,"dem_unit").text = 'cm-5 K-1'
    ET.SubElement(derivex,"dem_min").text = '{:.1e}'.format(dem.dem_min)
    ET.SubElement(derivex,"dem_max").text = '{:.1e}'.format(dem.dem_max)
    ET.SubElement(derivex,"img_sc").text = 'LOG 10'
    ET.SubElement(derivex,"img_id").text = '{} of {}'.format(i+1,dem.nimg)
    ET.SubElement(derivex,"github").text = 'https://github.com/alasdairwilson/demreg-py'
    ET.SubElement(derivex,"dem_ver").text = '{:}'.format(dem.dem_ver)
    ET.SubElement(derivex,"jp2_ver").text = '1.0'
    ET.SubElement(derivex,"produced").text = dem.produced
    ET.SubElement(derivex,"contact").text = 'alasdair.wilson@glasgow.ac.uk'
    ET.SubElement(heliox, "HV_ROTATION").text = '0'
#info for mouseover values
    ET.SubElement(heliox, "BUNIT").text = 'LOG10 cm-5 K-1'
    ET.SubElement(heliox, "DATAMAX").text = '255'
    ET.SubElement(heliox, "HV_ZERO").text = '{:.2f}'.format(np.log10(dem.dem_min))
    ET.SubElement(heliox, "HV_SCALE").text = '{:.2f}'.format((np.log10(dem.dem_max)-np.log10(dem.dem_min))/255)




    tree = ET.ElementTree(metax)
    tree.write(fname+'.xml', pretty_print=True)
    return


# if __name__=="__main__":
#     data=np.zeros([1024,1024])+1e19 
#     data[112:912,112:912]=5e19
#     data[212:812,212:812]=1e20
#     data[312:712,312:712]=1e21
#     data[412:612,412:612]=1e22
#     data[492:532,492:532]=1e23
#     dem=Dem(data=data,minTemp=5.7,maxTemp=5.9,minC=np.array(1E19),maxC=np.array(1e23),maxData=np.max(data),minData=np.min(data),imindex=0,nimg=7,naxis1=data.shape[0],naxis2=data.shape[1],cdelt1=2.4,cdelt2=2.4,filt_use='AIA94 AIA131 AIA171 AIA193 AIA211 AIA335 AIA94FE18',crpix1=512,crpix2=512)
#     dem2jp2(dem,'test2')



