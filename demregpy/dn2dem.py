"""Top-level DEM inversion interface."""

import numpy as np

from demregpy.demmap import demmap

__all__ = [
    'dn2dem',
]


def _normalize_gloci(gloci, nf):
    """Normalize the public ``gloci`` input to a per-filter 0/1 mask."""
    gloci_arr = np.asarray(gloci)
    if gloci_arr.ndim == 0:
        if gloci_arr == 0:
            return np.zeros(nf, dtype=int)
        if gloci_arr == 1:
            return np.ones(nf, dtype=int)
        raise ValueError("gloci scalar must be 0 or 1")

    if gloci_arr.shape != (nf,):
        raise ValueError(f"gloci array must have shape ({nf},)")
    if not np.all(np.isfinite(gloci_arr)):
        raise ValueError("gloci array must contain only finite values")
    if not np.all((gloci_arr == 0) | (gloci_arr == 1)):
        raise ValueError("gloci array must contain only 0/1 values")
    return gloci_arr.astype(int)


def dn2dem(dn_in, edn_in, tresp, tresp_logt, temps, reg_tweak=1.0, max_iter=10, gloci=0,
           rgt_fact=1.5, dem_norm0=None, nmu=40, warn=False, emd_int=False, emd_ret=False, l_emd=False, non_pos=False):
    """
    Recover a differential emission measure from channel counts.

    This is the main public wrapper around the lower-level regularised
    inversion routines. It accepts single-pixel, time-series, or small map-like
    arrays whose last axis is filter/channel.

    Parameters
    ----------
    dn_in : array_like
        Input channel counts. The last axis must be filter/channel, so valid
        shapes include ``(nf,)``, ``(n, nf)``, and ``(nx, ny, nf)``.
    edn_in : array_like
        Uncertainties on ``dn_in`` with the same shape and units.
    tresp : array_like
        Temperature response matrix with shape ``(nt_resp, nf)``.
    tresp_logt : array_like
        Log10 temperature grid for the first axis of ``tresp``.
    temps : array_like
        Temperature-bin edges at which to recover the DEM.
    reg_tweak : float, optional
        The initial normalised chisq target. Default is 1.0.
    max_iter : int, optional
        The maximum number of iterations to attempt, code iterates if negative DEM is produced.
        If max iter is reached before a suitable solution is found then the current solution is
        returned instead (which may contain negative values).
        Default is only 10 - although non_pos=True will set as 1.
    gloci : int or array_like, optional
        Weighting mode used when ``dem_norm0`` is not supplied. A scalar ``0``
        uses the self-normalised weighting scheme, a scalar ``1`` uses all EM
        loci curves, and a length-``nf`` 0/1 mask selects which filters
        contribute to the EM loci weighting.
        Default is 0.
    rgt_fact : float, optional
        The factor by which rgt_tweak increases each iteration.
        As the target chisq increases there is more flexibility allowed on the DEM.
        Default is 1.5.
    dem_norm0 : array_like, optional
        Initial DEM-shaped weighting for the constraint matrix. Only the
        relative values matter. If omitted, the weighting is determined from
        ``gloci``.
        Default is None.
    nmu : int, optional
        Number of reg param samples to calculate (default (or <=40) 500 for 0D, 42 for map).
        Default is 40.
    warn : bool, optional
        Print out any warnings (always warn for 1D, default no for higher dim data).
        Default is False.
    emd_int : bool, optional
        Perform the inversion in emission measure distribution space rather
        than DEM space.
        Default is False.
    emd_ret : bool, optional
        Return the result in EMD units instead of DEM units.
        Default is False.
    l_emd : bool, optional
        Remove the square-root factor in the constraint matrix. This is mainly
        useful with ``emd_int=True``.
        Default is False.
    non_pos : bool, optional
        Return the first solution even if it contains negative values. This is
        implemented by forcing ``max_iter=1``.
        Default is False.

    Returns
    -------
    dem : ndarray
        Recovered DEM or EMD. The output shape matches the input shape with the
        filter axis replaced by temperature bin.
    edem : ndarray
        Vertical uncertainties on ``dem``.
    elogt : ndarray
        Horizontal temperature resolution estimates in log10(T).
    chisq : ndarray
        Final reduced chi-squared values.
    dn_reg : ndarray
        Counts reconstructed from the recovered solution, with the same shape as
        ``dn_in``.
    """
    if len(temps) < 4:
        raise ValueError("temps must define at least 3 DEM bins")

    if dem_norm0 is not None:
        dem_norm0 = np.asarray(dem_norm0)
        finite_mask = np.isfinite(dem_norm0)
        if not finite_mask.all():
            if finite_mask.any():
                raise ValueError("dem_norm0 must contain only finite values")
            dem_norm0 = None
        elif np.all(dem_norm0 == 0):
            dem_norm0 = None

    # create our bin averages:
    logt = ([np.mean([(np.log10(temps[i])), np.log10(temps[i+1])]) for i in np.arange(0, len(temps)-1)])
    # and widths
    dlogt = (np.log10(temps[1:])-np.log10(temps[:-1]))
    nt = len(dlogt)
    logt = (np.array([np.log10(temps[0])+(dlogt[i]*(float(i)+0.5)) for i in np.arange(nt)]))
    # number of DEM entries

    # hopefully we can deal with a variety of data, nx,ny,nf
    sze = dn_in.shape

    # for a single pixel
    dem0 = None
    # for a single pixel
    if len(sze) == 1:
        nx = 1
        ny = 1
        nf = sze[0]
        dn = np.zeros([1, 1, nf])
        dn[0, 0, :] = dn_in
        edn = np.zeros([1, 1, nf])
        edn[0, 0, :] = edn_in
        if dem_norm0 is not None:
            dem0 = np.zeros([1, 1, nt])
            dem0[0, 0, :] = dem_norm0
        if warn is False:
            warn = True
        if nmu <= 40:
            nmu = 500

    # for a row of pixels
    if len(sze) == 2:
        nx = sze[0]
        ny = 1
        nf = sze[1]
        dn = np.zeros([nx, 1, nf])
        dn[:, 0, :] = dn_in
        edn = np.zeros([nx, 1, nf])
        edn[:, 0, :] = edn_in
        if dem_norm0 is not None:
            dem0 = np.zeros([nx, 1, nt])
            dem0[:, 0, :] = dem_norm0
        if nmu <= 40:
            nmu = 42

    # for 2d image
    if len(sze) == 3:
        nx = sze[0]
        ny = sze[1]
        nf = sze[2]
        dn = np.zeros([nx, ny, nf])
        dn[:, :, :] = dn_in
        edn = np.zeros([nx, ny, nf])
        edn[:, :, :] = edn_in
        if dem_norm0 is not None:
            dem0 = np.zeros([nx, ny, nt])
            dem0[:, :, :] = dem_norm0
        if nmu <= 40:
            nmu = 42
    # If want to ignore positivity constraint then set max_iter=1 and no need for the warnings
    if non_pos:
        max_iter = 1
        warn = False

    # If rgt_fact <=1 then the positivity loop won't work so warn about it
    if (warn and (rgt_fact <= 1)):
        print('Warning, rgt_fact should be > 1, for postivity loop to iterate properly.')

    # gloci can be 0/1 for all filters, or a per-filter 0/1 mask selecting
    # which filters contribute to the EM loci weighting.
    glc = _normalize_gloci(gloci, nf)

    if len(tresp[0, :]) != nf:
        raise ValueError("tresp must have the same number of filters as the data")

    truse = np.zeros([tresp[:, 0].shape[0], nf])
    # check the tresp has no elements <0
    # replace any it finds with the minimum tresp from the same filter
    for i in np.arange(0, nf):
        good = tresp[:, i] > 0
        if not np.any(good):
            raise ValueError(f"tresp filter {i} must contain at least one positive response value")
        min_positive = np.min(tresp[good, i])
        truse[:, i] = np.where(good, tresp[:, i], min_positive)

    tr = np.zeros([nt, nf])
    for i in np.arange(nf):
        #       Ideally should be interp in log-space, so changed
        # Not as big an issue for purely AIA filters, but more of an issue for steeper X-ray ones
        tr[:, i] = 10**np.interp(logt, tresp_logt, np.log10(truse[:, i]))
    #     Previous version
    #         tr[:,i]=np.interp(logt,tresp_logt,truse[:,i])

    rmatrix = np.zeros([nt, nf])
    # Put in the 1/K factor (remember doing it in logT not T hence the extra terms)
    dlogTfac = 10.0**logt*np.log(10.0**dlogt)
    # Do regularization of EMD or DEM
    if emd_int:
        l_emd = True
        for i in np.arange(nf):
            rmatrix[:, i] = tr[:, i]
    else:
        for i in np.arange(nf):
            rmatrix[:, i] = tr[:, i]*dlogTfac
    # Just scale so not dealing with tiny numbers
    sclf = 1E15
    rmatrix = rmatrix*sclf

    dn1d = np.reshape(dn, [nx*ny, nf])
    edn1d = np.reshape(edn, [nx*ny, nf])
    # create our 1d arrays for output
    dem1d = np.zeros([nx*ny, nt])
    chisq1d = np.zeros([nx*ny])
    edem1d = np.zeros([nx*ny, nt])
    elogt1d = np.zeros([nx*ny, nt])
    dn_reg1d = np.zeros([nx*ny, nf])

    # *****************************************************
    #  Actually doing the DEM calculations
    # *****************************************************
    # Should always be just running the first part of if here as setting dem01d to array of 1s if nothing given
    # So now more a check dimensions of things are correct
    if dem0 is not None and (dem0.ndim == dn.ndim):
        dem01d = np.reshape(dem0, [nx*ny, nt])
        dem1d, edem1d, elogt1d, chisq1d, dn_reg1d = \
            demmap(dn1d, edn1d, rmatrix, logt, dlogt, glc,
                   reg_tweak=reg_tweak, max_iter=max_iter,
                   rgt_fact=rgt_fact, dem_norm0=dem01d, nmu=nmu, warn=warn, l_emd=l_emd)
    else:
        dem1d, edem1d, elogt1d, chisq1d, dn_reg1d = \
            demmap(dn1d, edn1d, rmatrix, logt,
                   dlogt, glc, reg_tweak=reg_tweak, max_iter=max_iter,
                   rgt_fact=rgt_fact, dem_norm0=None, nmu=nmu, warn=warn, l_emd=l_emd)
    # reshape the 1d arrays to original dimensions and squeeze extra dimensions
    dem = ((np.reshape(dem1d, [nx, ny, nt]))*sclf).squeeze()
    edem = ((np.reshape(edem1d, [nx, ny, nt]))*sclf).squeeze()
    elogt = (np.reshape(elogt1d, [nx, ny, nt])/(2.0*np.sqrt(2.*np.log(2.)))).squeeze()
    chisq = (np.reshape(chisq1d, [nx, ny])).squeeze()
    dn_reg = (np.reshape(dn_reg1d, [nx, ny, nf])).squeeze()

    # There's probably a neater way of doing this (and maybe provide info of what was done as well?)
    # but fine for now as it works
    if emd_int and emd_ret:
        return dem, edem, elogt, chisq, dn_reg
    if emd_int and not emd_ret:
        return dem/dlogTfac, edem/dlogTfac, elogt, chisq, dn_reg
    if not emd_int and emd_ret:
        return dem*dlogTfac, edem*dlogTfac, elogt, chisq, dn_reg
    return dem, edem, elogt, chisq, dn_reg
