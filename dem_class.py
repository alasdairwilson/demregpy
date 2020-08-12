from dataclasses import dataclass
import numpy as np
from astropy.time import Time
@dataclass
class Dem:
    data:np.ndarray=None
    edem:np.ndarray=None
    elogt:np.ndarray=None
    chisq:np.ndarray=None
    dn_reg:np.ndarray=None
    temperatures:np.ndarray=None
    naxis:int=2
    minTemp:float=None
    maxTemp:float=None
    minC:np.float64=5e19
    maxC:np.float64=5e23
    hv_zero:np.float64=None
    hv_scale:np.float64=None
    datamax:int=255
    bunit:str='LOG10 cm-5 K-1'
    nimg:int=None
    filt_use:str=None
    crpix1:float=None
    crpix2:float=None
    cunit1:float=None
    cunit2:float=None
    crval1:float=None
    crval2:float=None
    BITPIX:int=8
    naxis1:int=None
    naxis2:int=None
    cdelt1:int=None
    cdelt2:int=None
    dsun_obs:float=None
    rsun_obs:float=None
    rsun_ref:float=None
    crlt_obs:float=None
    crln_obs:float=None
    hglt_obs:float=None
    hgln_obs:float=0
    t_obs:Time=None
    contact:str=None
    produced:str=None
    dem_ver:float=None
