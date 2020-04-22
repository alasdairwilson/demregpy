import numpy as np
import scipy

def demmap_pos(dd,ed,rmatrix,logt,dlogt,glc,dem,chisq, \
  edem,elogt,dn_reg,reg_tweak=1,max_iter=10,rgt_fact=1.5, \
  dem_norm0=0):
    na=dd.shape[0]
    nf=rmatrix.shape[1]
    nt=logt.shape[0]
    print(na,nf,nt)
    #set up some arrays
    dem=np.zeros([na,nt])
    edem=np.zeros([na,nt])
    elogt=np.zeros([na,nt])
    rmatrixin=np.zeros([nt,nf])
    kdag=np.zeros([nf,nt])
    filt=np.zeros([nf,nt])
    chisq=np.zeros([na])
    kdagk=np.zeros([nt,nt])
    dn_reg=np.zeros([na,nf])
    ednin=np.zeros([nf])
    ltt=min(logt)+(max(logt)-min(logt))*np.arange(51)/(51-1.0)
    nmu=42
    #for each dem
    for i in np.arange(na):
        #select a single set of dn values and remove singleton dimensions
        dnin=dd[i,:].squeeze()
        ednin[:]=ed[i,:].squeeze()
        #does dem_norm0 exist
        if (dem_norm0.shape[0] != 0):
        #single norm from the array
            dem_reg_wght=dem_norm0[i,:].squeeze()
        for kk in np.arange(nf):
            #response matrix
            rmatrixin[:,kk]=rmatrix[:,kk]/ednin[kk]

        dn=dnin/ednin
        edn=ednin/ednin

        # checking for Inf and NaN
        if ( sum(np.isnan(dn)) == 0 and sum(np.isinf(dn)) == 0 ):
            ndem=1
            piter=0
            rgt=reg_tweak

            L=np.zeros([nt,nt])

            test_dem_reg=(np.zeros(1)).astype(int)
            
    #  If you have supplied an initial guess/constraint normalized DEM then don't
    #  need to calculate one (either from L=1/sqrt(dLogT) or min of EM loci)
      
    #  Though need to check what you have supplied is correct dimension 
    #  and no element 0 or less.
            if( len(dem_reg_wght) == nt):
                if (np.prod(dem_reg_wght) > 0):
                    test_dem_reg=np.ones(1).astype(int)
            # use the min of the emloci as the initial dem_reg
            if ((test_dem_reg).shape[0] == nt):
                if (np.sum(glc) > 0.0):
                    gdglc=(glc>0).nonzero()[0]
                    emloci=np.zeros(nt,gdglc.shape[0])
                    #for each gloci take the minimum and work out the emission measure
                    for ee in np.arrange(gdglc.shape[0]):
                        emloci[:,ee]=dnin[gdglc[ee]]/(rmatrix[:,gdglc[ee]])
                    #for each temp we take the min of the loci curves as the estimate of the dem
                    for ttt in np.arange(nt):
                        dem_model[ttt]=min(emloci[ttt,:] > 0.)
                    dem_model=np.convolve(dem_model,np.ones(3)/3)[1:-1]
                    dem_reg=dem_model/max(dem_model)+1e-10
                else:
                    # Calculate the initial constraint matrix
                    # Just a diagional matrix scaled by dlogT
                    L=diag(1.0/sqrt(dlogt[:]))
                    #run gsvd
                    sva,svb,U,V,W=dem_inv_gsvd(RMatrixin,L)
                    #run reg map
                    lamb=dem_reg_map(sva,svb,U,W,DN,eDN,rgt,nmu)
                    #filt, diagonal matrix
                    filt=diag(sva[/(sva[kk]**2+svb[kk]**2*lamb))
                    kdag=W@(U[:nf,:nf].T@filt)
                    dr0=(kdag@dn).squeeze
                    # only take the positive with certain amount (fcofmx) of max, then make rest small positive
                    fcofmx=1e-4
                    mask=np.where(dr0 > 0) and (dr0 > fcofmax*np.max(dr0)) 
                    dem_reg[mask]=dr0[mask]
                    dem_reg[~mask]=1
                    #scale and then smooth by convolution with boxcar width 3
                    dem_reg=np.convolve(dem_reg/(fcofmx*max(dr0),np.ones(3)/3))[1:-1]
            else:
                dem_reg=dem_reg_wght

            while(ndem > 0 and piter < max_iter):
                for kk in np.arange(nt):
                    L=np.sqrt(dlotT[kk])/np.sqrt(abs(dem_reg[kk])) 
                
                sva,svb,U,V,W = dem_inv_gsvdcsq(RMatrixin,L)
                lamb=dem_inv_reg_parameter_map(sva,svb,U,W,DN,eDN,rgt,nmu)

                
    # now we work on deminv_gsvdcsq...to be continued.....
                
