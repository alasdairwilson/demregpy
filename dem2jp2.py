import numpy as np
import glymur
from lxml import etree as ET
from dataclasses import dataclass
from skimage.util import img_as_ubyte
from dem_class import Dem
import os
from datetime import datetime
def dem2jp2(img_data,dem,fname,i,bin_min,bin_max):
    #log 10 the data
    datasc=np.log10(img_data+1)
    #take logs of our data range
    logmin=np.log10(dem.minC)
    logmax=np.log10(dem.maxC)
    #floor the data to datamin
    datasc[datasc < logmin]=logmin
    #ceiling the data to datamax
    datasc[datasc > logmax]=logmax
    #scale the data between 0-1
    datasc = ((datasc)-logmin)/(logmax-logmin)
    #convert to unsigned byte
    bytesc=img_as_ubyte(datasc)
    #create the jp2 file
    jp2=glymur.Jp2k(fname+'.jp2',bytesc)
    #create the header
    demxml(dem,fname,i,bin_min,bin_max)
    #load the header
    xmlbox = glymur.jp2box.XMLBox(filename=fname+'.xml')
    #add the header to the jp2 file.
    jp2.append(xmlbox)
    os.remove(fname+'.xml')
    return

def demxml(dem,fname,i,bin_min,bin_max):
    metax = ET.Element("meta")
    fitsx = ET.SubElement(metax,"fits")
    heliox = ET.SubElement(metax, "helioviewer")
    derivex = ET.SubElement(heliox,"derived_data")
    ET.SubElement(fitsx, "BUNIT").text = 'LOG 10 cm^-5 K^-1'
    ET.SubElement(fitsx, "HV_ZERO").text = '{:.2f}'.format(np.log10(dem.minC))
    ET.SubElement(fitsx, "HV_SCALE").text = '{:.2f}'.format((np.log10(dem.maxC)-np.log10(dem.minC))/255)
    ET.SubElement(fitsx, "T_OBS").text = dem.t_obs.isot+'Z'
    ET.SubElement(fitsx, "DATE-OBS").text = dem.t_obs.isot
    ET.SubElement(fitsx, "NAXIS1").text = '{:}'.format(dem.naxis1)
    ET.SubElement(fitsx, "NAXIS2").text = '{:}'.format(dem.naxis2)
    ET.SubElement(fitsx, "CDELT1").text = '{:}'.format(dem.cdelt1)
    ET.SubElement(fitsx, "CDELT2").text = '{:}'.format(dem.cdelt2)
    ET.SubElement(fitsx, "CRPIX1").text = '{:}'.format(dem.crpix1)
    ET.SubElement(fitsx, "CRPIX2").text = '{:}'.format(dem.crpix2)
    ET.SubElement(fitsx, "CUNIT1").text = '{:}'.format(dem.cunit1)
    ET.SubElement(fitsx, "CUNIT2").text = '{:}'.format(dem.cunit2)
    ET.SubElement(fitsx, "CRVAL1").text = '{:}'.format(dem.crval1)
    ET.SubElement(fitsx, "CRVAL2").text = '{:}'.format(dem.crval2)    
    ET.SubElement(fitsx, "CROTA2").text = '{:}'.format(dem.crota2)



    ET.SubElement(derivex,"derived_quantity").text = "DEM(T)"
    ET.SubElement(derivex,"derivation_method").text = "Regularised Inversion (Hannah and Kontar 2012)"
    ET.SubElement(derivex,"filters").text = dem.filt_use
    ET.SubElement(derivex,"DEM_temp_range").text = '{:1}-{:1}'.format(dem.minTemp,dem.maxTemp)
    ET.SubElement(derivex,"temp_range_K").text = '{:.2e}-{:.2e}'.format(10**dem.minTemp,10**dem.maxTemp)
    ET.SubElement(derivex,"temp_bin").text = '{:.2e}-{:.2e}'.format(bin_min,bin_max)
    ET.SubElement(derivex,"dem_unit").text = 'cm^-5 K^-1'
    ET.SubElement(derivex,"img_mindata").text = '{:.1e}'.format(dem.minC)
    ET.SubElement(derivex,"img_maxdata").text = '{:.1e}'.format(dem.maxC)
    ET.SubElement(derivex,"img_scale").text = 'LOG 10'
    ET.SubElement(derivex,"img_id").text = '{} of {}'.format(i+1,dem.nimg)
    ET.SubElement(derivex,"github").text = 'https://github.com/alasdairwilson/demreg-py'
    ET.SubElement(derivex,"demregpy_version").text = '1.1'
    ET.SubElement(derivex,"jp2gen_version").text = '1.0'
    datetime.today().strftime('%Y-%m-%d')
    ET.SubElement(derivex,"produced_at").text = 'University of Glasgow'
    ET.SubElement(derivex,"produced_on").text = datetime.today().strftime('%Y-%m-%d')
    ET.SubElement(derivex,"contact_email").text = 'alasdair.wilson@glasgow.ac.uk'
#ADD THE GITHUB/CONTACT ETC
    ET.SubElement(heliox, "BUNIT").text = 'LOG 10 cm^-5 K^-1'
    ET.SubElement(heliox, "HV_ZERO").text = '{:.2f}'.format(np.log10(dem.minC))
    ET.SubElement(heliox, "HV_SCALE").text = '{:.2f}'.format((np.log10(dem.maxC)-np.log10(dem.minC))/255)
    ET.SubElement(heliox, "HV_ROTATION").text = '0'
#info for mouseover values


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



