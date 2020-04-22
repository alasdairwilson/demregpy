import numpy as np
def dem_reg_map(sigmaa,sigmab,U,W,data,err,reg_tweak,nmu=500):
    """
    dem_reg_map
    computes the regularisation parameter
    
    Inputs


    sigmaa: 
        gsv vector
    sigmab: 
        gsv vector
    U:      
        gsvd matrix
    V:      
        gsvd matrix
    data:   
        dn data
    err:    
        dn error
    reg_tweak: 
        how much to adjust the chisq each iteration

    Outputs

    opt:
        regularization paramater

    """

    ndata=data.shape[0]
    nreg=sigmaa.shape[0]

    arg=np.zeros([nreg,nmu])
    discr=np.zeros([nmu])

    maxx=max(sigmaa[:ndata-1]/sigmab[:ndata-1])
    minx=min(sigmaa[:ndata-1]/sigmab[:ndata-1])

    step=(np.log(maxx)-np.log(minx))/(nmu-1.)
    mu=exp(arange(nmu)*step)*minx
    
    for kk in np.arange(ndata):
        coef=data@u(kk,:)-sigmaa(kk)
        for ii in np.arange(nmu):
            arg[kk,ii]=(mu[ii]*sigmab[kk]**2*coef/(sigmaa[kk]**2+mu[ii]*sigmab[kk]**2))**2
    
    discr=np.sum(arg,axis=0)-np.sum(err**2)*reg_tweak
    opt=mu[np.argmim(abs(discr))]


    return opt