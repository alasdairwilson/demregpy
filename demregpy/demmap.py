"""Lower-level DEM inversion routines used by :func:`demregpy.dn2dem`."""

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from numpy.linalg import svd
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits
from tqdm import tqdm

__all__ = [
    'dem_pix',
    'dem_unwrap',
    'demmap',
]

def demmap(
    dd, ed, rmatrix, logt, dlogt, glc, reg_tweak=1.0, max_iter=10,
    rgt_fact=1.5, dem_norm0=None, nmu=42, warn=False, l_emd=False
):
    """
    Recover DEMs for a stack of one-dimensional observations.

    Each row of ``dd`` is treated as an independent observation with the same
    temperature response matrix. This function is the lower-level workhorse used
    by :func:`demregpy.dn2dem` after the input arrays have been reshaped.

    Parameters
    ----------
    dd : array_like
        Input counts with shape ``(na, nf)``.
    ed : array_like
        Uncertainties on ``dd`` with the same shape.
    rmatrix : array_like
        Response matrix with shape ``(nt, nf)``.
    logt : array_like
        Temperature-bin centres in log10(T).
    dlogt : array_like
        Width of each temperature bin in log10(T).
    glc : array_like
        Length-``nf`` 0/1 mask selecting the filters used for EM loci
        weighting.
    reg_tweak : float, optional
        Initial chisq target. Default is 1.0.
    max_iter : int, optional
        Maximum number of times to attempt the gsvd before giving up, returns the last attempt if max_iter reached.
        Default is 10.
    rgt_fact : float, optional
        Scale factor for the increase in chi-sqaured target for each iteration. Default is 1.5.
    dem_norm0 : array_like, optional
        Provides a "guess" dem as a starting point, if none is supplied one is created. Default is None.
    nmu : int, optional
        Number of reg param samples to use. Default is 42.
    warn : bool, optional
        Print out warnings. Default is False.
    l_emd : bool, optional
        Remove sqrt from constraint matrix (best with EMD). Default is False.

    Returns
    -------
    dem : ndarray
        DEM values with shape ``(na, nt)``.
    edem : ndarray
        Vertical uncertainties on ``dem``.
    elogt : ndarray
        Horizontal temperature resolution estimates in log10(T).
    chisq : ndarray
        Reduced chi-squared values for each observation.
    dn_reg : ndarray
        Reconstructed counts with shape ``(na, nf)``.
    """
    na = dd.shape[0]
    nf = rmatrix.shape[1]
    nt = logt.shape[0]
    if dem_norm0 is None:
        dem_norm0 = np.ones((na, nt))
    elif np.isscalar(dem_norm0):
        raise ValueError("dem_norm0 must be an array matching (na, nt), not a scalar")
    dem = np.zeros([na, nt])
    edem = np.zeros([na, nt])
    elogt = np.zeros([na, nt])
    chisq = np.zeros([na])
    dn_reg = np.zeros([na, nf])
    # do we have enough DEM's to make parallel make sense?
    if (na >= 200):
        n_par = 100
        niter = (int(np.floor((na)/n_par)))
        # Put this here to make sure running dem calc in parallel, not the underlying np/gsvd stuff (this correct/needed?)
        with threadpool_limits(limits=1):
            with ProcessPoolExecutor() as exe:
                futures = [exe.submit(dem_unwrap, dd[i*n_par:(i+1)*n_par, :], ed[i*n_par:(i+1)*n_par, :],
                           rmatrix, logt, dlogt, glc, reg_tweak=reg_tweak, max_iter=max_iter,
                           rgt_fact=rgt_fact, dem_norm0=dem_norm0[i*n_par:(i+1)*n_par, :],
                           nmu=nmu, warn=warn, l_emd=l_emd) for i in np.arange(niter)]
                kwargs = {
                    'total': len(futures),
                    'unit': ' x10^2 DEM',
                    'unit_scale': True,
                    'leave': True
                }
                for f in tqdm(as_completed(futures), **kwargs):
                    pass
            for i, f in enumerate(futures):
                # store the outputs in arrays
                dem[i*n_par:(i+1)*n_par, :] = f.result()[0]
                edem[i*n_par:(i+1)*n_par, :] = f.result()[1]
                elogt[i*n_par:(i+1)*n_par, :] = f.result()[2]
                chisq[i*n_par:(i+1)*n_par] = f.result()[3]
                dn_reg[i*n_par:(i+1)*n_par, :] = f.result()[4]
            # if there are any remaining dems then execute remainder in serial
            if (np.mod(na, niter*n_par) != 0):
                i_start = niter*n_par
                for i in range(na-i_start):
                    result = dem_pix(dd[i_start+i, :], ed[i_start+i, :], rmatrix, logt, dlogt, glc,
                                     reg_tweak=reg_tweak, max_iter=max_iter, rgt_fact=rgt_fact,
                                     dem_norm0=dem_norm0[i_start+i, :],
                                     nmu=nmu, warn=warn, l_emd=l_emd)
                    dem[i_start+i, :] = result[0]
                    edem[i_start+i, :] = result[1]
                    elogt[i_start+i, :] = result[2]
                    chisq[i_start+i] = result[3]
                    dn_reg[i_start+i, :] = result[4]
    # else we execute in serial
    else:
        for i in range(na):
            result = dem_pix(dd[i, :], ed[i, :], rmatrix, logt, dlogt, glc,
                             reg_tweak=reg_tweak, max_iter=max_iter, rgt_fact=rgt_fact,
                             dem_norm0=dem_norm0[i, :], nmu=nmu, warn=warn, l_emd=l_emd)
            dem[i, :] = result[0]
            edem[i, :] = result[1]
            elogt[i, :] = result[2]
            chisq[i] = result[3]
            dn_reg[i, :] = result[4]
    return dem, edem, elogt, chisq, dn_reg


def dem_unwrap(
    dn, ed, rmatrix, logt, dlogt, glc, reg_tweak=1.0, max_iter=10,
    rgt_fact=1.5, dem_norm0=None, nmu=42, warn=False, l_emd=False
):
    """
    Run :func:`dem_pix` over a stack of observations in serial.

    Parameters
    ----------
    dn : ndarray
        Input counts with shape ``(ndem, nf)``.
    ed : ndarray
        Uncertainties on ``dn`` with the same shape.
    rmatrix : ndarray
        Response matrix with shape ``(nt, nf)``.
    logt : array_like
        Log temperature bins.
    dlogt : array_like
        Size of temperature bins.
    glc : array_like
        Length-``nf`` 0/1 mask selecting the filters used for EM loci
        weighting.
    reg_tweak : float, optional
        Initial Chisq target, by default 1.0
    max_iter : int, optional
        Max number of iterations to reach target chisq before giving up, by default 10
    rgt_fact : float, optional
        Factor to increase chisq by each iteration, by default 1.5
    dem_norm0 : array_like, optional
        Initial guess at the dem shape, by default 0
    nmu : int, optional
        number of reg param samples to use, by default 42
    warn : bool, optional
        Print warnings, by default False
    l_emd : bool, optional
        Remove sqrt from constraint matrix, by default False

    Returns
    -------
    dem : ndarray
        DEM values with shape ``(ndem, nt)``.
    edem : ndarray
        Vertical uncertainties on ``dem``.
    elogt : ndarray
        Horizontal temperature resolution estimates in log10(T).
    chisq : array_like
        Reduced chi-squared values.
    dn_reg : ndarray
        Reconstructed counts with shape ``(ndem, nf)``.
    """
    ndem = dn.shape[0]
    nt = logt.shape[0]
    nf = dn.shape[1]
    if dem_norm0 is None:
        dem_norm0 = np.ones((ndem, nt))
    elif np.isscalar(dem_norm0):
        raise ValueError("dem_norm0 must be an array matching (ndem, nt), not a scalar")
    dem = np.zeros([ndem, nt])
    edem = np.zeros([ndem, nt])
    elogt = np.zeros([ndem, nt])
    chisq = np.zeros([ndem])
    dn_reg = np.zeros([ndem, nf])
    for i in range(ndem):
        result = dem_pix(
            dn[i, :], ed[i, :], rmatrix, logt, dlogt, glc,
            reg_tweak=reg_tweak, max_iter=max_iter, rgt_fact=rgt_fact,
            dem_norm0=dem_norm0[i, :], nmu=nmu, warn=warn, l_emd=l_emd
        )
        dem[i, :] = result[0]
        edem[i, :] = result[1]
        elogt[i, :] = result[2]
        chisq[i] = result[3]
        dn_reg[i, :] = result[4]
    return dem, edem, elogt, chisq, dn_reg


def dem_pix(dnin, ednin, rmatrix, logt, dlogt, glc, reg_tweak=1.0, max_iter=10,
            rgt_fact=1.5, dem_norm0=None, nmu=42, warn=True, l_emd=False):
    """
    Recover a DEM for one observation vector.

    Parameters
    ----------
    dnin : array_like
        Input counts for one observation.
    ednin : array_like
        Uncertainties on ``dnin``.
    rmatrix : ndarray
        Temperature response of each channel.
    logt : array_like
        Log temperature bins.
    dlogt : array_like
        Size of temperature bins.
    glc : array_like
        Length-``nf`` 0/1 mask selecting the filters used for EM loci
        weighting.
    reg_tweak : float, optional
        Initial Chisq target, by default 1.0
    max_iter : int, optional
        Max number of iterations to reach target chisq before giving up, by default 10
    rgt_fact : float, optional
        Factor to increase chisq by each iteration, by default 1.5
    dem_norm0 : array_like, optional
        Initial guess at the dem shape, by default 0
    nmu : int, optional
        number of reg param samples to use, by default 42
    warn : bool, optional
        Print warnings, by default False
    l_emd : bool, optional
        Remove sqrt from constraint matrix, by default False

    Returns
    -------
    dem : ndarray
        Recovered DEM values.
    edem : ndarray
        Vertical uncertainties on ``dem``.
    elogt : ndarray
        Horizontal temperature resolution estimates in log10(T).
    chisq : array_like
        Reduced chi-squared value.
    dn_reg : ndarray
        Reconstructed counts for each filter.
    """
    nf = rmatrix.shape[1]
    nt = logt.shape[0]
    if not np.all(np.isfinite(dnin)):
        raise ValueError("dnin must contain only finite values")
    if np.any(dnin < 0):
        raise ValueError("dnin must be non-negative")
    user_supplied_dem_norm0 = dem_norm0 is not None
    if dem_norm0 is None:
        dem_norm0 = np.ones(nt)
    elif np.isscalar(dem_norm0):
        raise ValueError("dem_norm0 must be an array matching (nt,), not a scalar")
    if nt < 3:
        raise ValueError("logt/dlogt must define at least 3 DEM bins")
    ltt = np.min(logt) + 1e-8 + (np.max(logt) - np.min(logt)) * np.arange(51) / (52 - 1.0)
    dem = np.zeros(nt)
    edem = np.zeros(nt)
    elogt = np.zeros(nt)
    chisq = 0
    dn_reg = np.zeros(nf)
    rmatrixin = rmatrix / ednin[np.newaxis, :]
    dn = dnin/ednin
    edn = ednin/ednin
    ndem = 1
    piter = 0
    rgt = reg_tweak
    #  If you have supplied an initial guess/constraint normalized DEM then don't
    #  need to calculate one (either from L=1/sqrt(dLogT) or min of EM loci)
    # As the call to this now sets dem_norm to array of 1s if nothing provided by user can also test for that
    # Before calling this dem_norm0 is set to array of 1s if nothing provided by user
    # So we need to work out some weighting for L or is one provided as dem_norm0 (not 0 or array of 1s)?
    if ((not user_supplied_dem_norm0) or np.all(dem_norm0 == 1.0) or dem_norm0[0] == 0):
        # Need to work out a weighting here then, have two approaches:
        #         1. Do it via the min of em loci - chooses this if gloci, glc=1 from user
        if (np.sum(glc) > 0.0):
            gdglc = (glc > 0).nonzero()[0]
            emloci = np.zeros((nt, gdglc.shape[0]))
            # for each gloci take the minimum and work out the emission measure
            for ee in np.arange(gdglc.shape[0]):
                emloci[:, ee] = dnin[gdglc[ee]]/(rmatrix[:, gdglc[ee]])
            # for each temp we take the min of the loci curves as the estimate of the dem
            dem_model = np.zeros(nt)
            for ttt in np.arange(nt):
                nz = np.nonzero(emloci[ttt, :])[0]
                dem_model[ttt] = np.min(emloci[ttt, nz]) if nz.size > 0 else 0.0
            dem_reg_lwght = dem_model
        # ~~~~~~~~~~~~~~~~~
        # 2. Or if nothing selected will run reg once, and use solution as weighting (self norm approach)
        else:
            # Calculate the initial constraint matrix
            # Just a diagonal matrix scaled by dlogT
            ldiag = 1.0 / np.sqrt(dlogt[:])
            # solve once and use the result only to weight the real solve below
            basis = _standard_form_svd(rmatrixin.T, ldiag)
            lamb = _discrepancy_lambda(basis, dn, rgt * np.sum(edn**2))
            kdag = _regularised_inverse(basis, lamb)
            dr0 = (kdag@dn).squeeze()
            # only take the positive with certain amount (fcofmx) of max, then make rest small positive
            fcofmax = 1e-4
            mask = (dr0 > 0) & (dr0 > fcofmax * np.max(dr0))
            # fill the rest relative to dr0's peak, so the weighting doesn't depend on units
            fill = fcofmax * np.max(dr0) if np.any(mask) else 1.0
            dem_reg_lwght = np.full(nt, fill)
            dem_reg_lwght[mask] = dr0[mask]
        # ~~~~~~~~~~~~~~~~~
        # Just smooth these initial dem_reg_lwght and max sure no value is too small
        # dem_reg_lwght=(np.convolve(dem_reg_lwght,np.ones(3)/3))[1:-1]/np.max(dem_reg_lwght[:])
        dem_reg_lwght = (np.convolve(dem_reg_lwght[1:-1], np.ones(5)/5))[1:-1]/np.max(dem_reg_lwght[:])
        dem_reg_lwght[dem_reg_lwght <= 1e-8] = 1e-8
    else:
        # Otherwise just set dem_reg to inputted weight
        dem_reg_lwght = dem_norm0
    # Now actually do the dem regularisation using the L weighting from above
    # Faster to do this and the SVD on R and L before the pos loop
    if l_emd:
        # this works better with EMD calc, instead of DEM
        ldiag = 1 / abs(dem_reg_lwght)
    else:
        ldiag = np.sqrt(dlogt) / np.sqrt(abs(dem_reg_lwght))
    basis = _standard_form_svd(rmatrixin.T, ldiag)
    err_term = np.sum(edn**2)
    # Loop until the DEM is positive or max_iter is reached, loosening the chi-squared target each time
    while ((ndem > 0) and (piter < max_iter)):
        lamb = _discrepancy_lambda(basis, dn, rgt * err_term)
        kdag = _regularised_inverse(basis, lamb)

        dem_reg_out = (kdag@dn).squeeze()

        ndem = len(dem_reg_out[dem_reg_out < 0])
        rgt = rgt_fact*rgt
        piter += 1
    if (warn and (piter == max_iter)):
        print('Warning, positivity loop hit max iterations, so increase max_iter? Or rgt_fact too small?')
    dem = dem_reg_out
    # work out the theoretical dn and compare to the input dn
    dn_reg = (rmatrix.T @ dem_reg_out).squeeze()
    residuals = (dnin-dn_reg)/ednin
    # work out the chisquared
    chisq = np.sum(residuals**2)/(nf)
    # do error calculations on dem
    edem = np.sqrt(np.sum(kdag**2, axis=1))
    kdagk = kdag@rmatrixin.T
    kdagk_max = np.max(kdagk, axis=0)
    elogt = np.zeros(nt)
    for kk in np.arange(nt):
        rr = np.interp(ltt, logt, kdagk[:, kk])
        hm_mask = (rr >= kdagk_max[kk]/2.)
        elogt[kk] = dlogt[kk]
        if (np.sum(hm_mask) > 0):
            elogt[kk] = (ltt[hm_mask][-1]-ltt[hm_mask][0])/2
    return dem, edem, elogt, chisq, dn_reg


def _standard_form_svd(A, ldiag):
    """
    SVD of the response with the diagonal constraint divided out, ``A L^-1 = U diag(s) V^T``.

    The solution of min ||A x - d||^2 + lam ||L x||^2 is then
    ``x = L^-1 V diag(s / (s^2 + lam)) U^T d`` for any regularisation strength ``lam``, so
    trying many ``lam`` needs only this one decomposition.

    Returns a dict with ``U`` (nf, k), ``s`` (k,), ``V`` (nt, k) and ``linv`` (nt,).
    """
    ldiag = np.asarray(ldiag, dtype=float)
    linv = np.divide(1.0, ldiag, out=np.zeros_like(ldiag), where=ldiag != 0)
    U, s, Vt = svd(A * linv[np.newaxis, :], full_matrices=False)
    return {"U": U, "s": s, "V": Vt.T, "linv": linv}


def _misfit(basis, coef, outside, lam):
    """Misfit ||A x - d||^2 of the regularised solution at strength ``lam``."""
    s2 = basis["s"] ** 2
    return np.sum((lam / (s2 + lam) * coef) ** 2) + outside


def _discrepancy_lambda(basis, data, target):
    """
    Find the regularisation strength at which the misfit just reaches ``target`` (the
    discrepancy principle).

    Stronger regularisation always fits worse, so there is a single answer and it is found exactly.
    If ``target`` can't be reached, the nearer end of the search range is returned.
    """
    coef = basis["U"].T @ data
    # the part of the data the response can't produce at all, so no strength can fit it
    outside = max(float(np.sum(data**2) - np.sum(coef**2)), 0.0)
    s2 = basis["s"] ** 2
    s2pos = s2[s2 > 0]
    if s2pos.size == 0:
        return 1.0
    # lam far below min(s^2) means almost no smoothing, far above max(s^2) almost total smoothing
    lo, hi = np.log(s2pos.min() * 1e-8), np.log(s2pos.max() * 1e8)
    f = lambda loglam: _misfit(basis, coef, outside, np.exp(loglam)) - target  # noqa: E731
    flo, fhi = f(lo), f(hi)
    if flo >= 0:
        return float(np.exp(lo))
    if fhi <= 0:
        return float(np.exp(hi))
    return float(np.exp(brentq(f, lo, hi, xtol=1e-10)))


def _regularised_inverse(basis, lam):
    """The (nt, nf) matrix ``L^-1 V diag(s / (s^2 + lam)) U^T`` taking weighted data to the DEM."""
    s = basis["s"]
    filt = s / (s**2 + lam)
    return basis["linv"][:, None] * (basis["V"] @ (filt[:, None] * basis["U"].T))
