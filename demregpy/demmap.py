"""Lower-level DEM inversion routines used by :func:`demregpy.dn2dem`."""

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from numpy.linalg import pinv, svd
from threadpoolctl import threadpool_limits
from tqdm import tqdm

__all__ = [
    'dem_inv_gsvd',
    'dem_pix',
    'dem_reg_map',
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
            # run gsvd
            sva, svb, U, _V, W = dem_inv_gsvd_diag(rmatrixin.T, ldiag)
            # run reg map
            mu, misfit_curve, err_term = _dem_reg_map_curve(sva, svb, U, dn, edn, nmu)
            lamb = _dem_reg_map_select(mu, misfit_curve, err_term, rgt)
            # filt, diagonal matrix
            U_nf = U[:nf, :nf]
            W_nf = W[:, :nf]
            sva_nf = sva[:nf]
            svb_nf_sq = svb[:nf]**2
            filt_diag = sva_nf / (sva_nf**2 + svb_nf_sq*lamb)
            kdag = W_nf @ (U_nf * filt_diag[:, None])
            dr0 = (kdag@dn).squeeze()
            # only take the positive with certain amount (fcofmx) of max, then make rest small positive
            fcofmax = 1e-4
            mask = (dr0 > 0) & (dr0 > fcofmax * np.max(dr0))
            dem_reg_lwght = np.ones(nt)
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
    # Faster to do this and the GSVD on R and L before the pos loop
    if l_emd:
        # this works better with EMD calc, instead of DEM
        ldiag = 1 / abs(dem_reg_lwght)
    else:
        ldiag = np.sqrt(dlogt) / np.sqrt(abs(dem_reg_lwght))
    sva, svb, U, _V, W = dem_inv_gsvd_diag(rmatrixin.T, ldiag)
    mu, misfit_curve, err_term = _dem_reg_map_curve(sva, svb, U, dn, edn, nmu)
    U_nf = U[:nf, :nf]
    W_nf = W[:, :nf]
    sva_nf = sva[:nf]
    svb_nf_sq = svb[:nf]**2
    #  Now loop until positive solution or max_iter reached
    while ((ndem > 0) and (piter < max_iter)):
        # #make L from 1/dem reg scaled by dlogt and diagonalise
        # L=np.diag(np.sqrt(dlogt)/np.sqrt(abs(dem_reg_lwght)))
        # #call gsvd and reg map
        # sva,svb,U,V,W = dem_inv_gsvd(rmatrixin.T,L)
        lamb = _dem_reg_map_select(mu, misfit_curve, err_term, rgt)
        filt_diag = sva_nf / (sva_nf**2 + svb_nf_sq*lamb)
        kdag = W_nf @ (U_nf * filt_diag[:, None])

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


def _dem_reg_map_curve(sigmaa, sigmab, U, data, err, nmu):
    """Precompute the regularization misfit curve for a fixed GSVD."""
    nf = data.shape[0]
    sigs = sigmaa[:nf]/sigmab[:nf]
    maxx = np.max(sigs)
    minx = np.min(sigs)**2.0*1E-4
    minx = max(minx, np.finfo(float).tiny)
    denom = max(nmu - 1.0, 1.0)
    log_min = np.log(minx)
    log_max = np.log(maxx)
    log_max = min(log_max, np.log(np.finfo(float).max))
    step = (log_max - log_min) / denom
    log_mu = log_min + np.arange(nmu) * step
    log_mu = np.clip(log_mu, log_min, log_max)
    mu = np.exp(log_mu)
    coef = U[:nf, :] @ data
    sigmab_sq = sigmab[:nf, None]**2
    numerator = mu[None, :] * sigmab_sq * coef[:, None]
    denom = sigmaa[:nf, None]**2 + mu[None, :] * sigmab_sq
    misfit_curve = np.sum((numerator / denom)**2, axis=0)
    err_term = np.sum(err**2)
    return mu, misfit_curve, err_term


def _dem_reg_map_select(mu, misfit_curve, err_term, reg_tweak):
    """Select the regularization parameter for a given target misfit."""
    discr = misfit_curve - err_term * reg_tweak
    return mu[np.argmin(np.abs(discr))]


def dem_reg_map(sigmaa, sigmab, U, W, data, err, reg_tweak, nmu=500):
    """
    Select the regularization parameter from a GSVD solution.

    Parameters
    ----------
    sigmaa : array_like
        GSVD ``alpha`` singular values.
    sigmab : array_like
        GSVD ``beta`` singular values.
    U : ndarray
        GSVD ``U`` matrix.
    W : ndarray
        GSVD ``W`` matrix.
        Unused.
    data : array_like
        Data vector in the weighted space used by the inversion.
    err : array_like
        Uncertainty vector corresponding to ``data``.
    reg_tweak : float
        Target chi-squared scaling used when choosing the regularization
        parameter.
    nmu : int, optional
        Number of candidate regularization parameters to sample.

    Returns
    -------
    float
        Selected regularization parameter.
    """
    mu, misfit_curve, err_term = _dem_reg_map_curve(sigmaa, sigmab, U, data, err, nmu)
    return _dem_reg_map_select(mu, misfit_curve, err_term, reg_tweak)


def _dem_inv_gsvd_from_bdiag(A, bdiag):
    """GSVD helper for the common case where B is diagonal."""
    bdiag = np.asarray(bdiag)
    bdiag_inv = np.divide(
        1.0,
        bdiag,
        out=np.zeros_like(bdiag, dtype=np.result_type(A, bdiag, float)),
        where=bdiag != 0,
    )
    # For diagonal B, A @ pinv(B) is just column scaling by pinv(diag(B)).
    AB1 = A * bdiag_inv
    sze = AB1.shape
    p = max(sze)
    # Use the rectangular SVD directly, then pad the singular spectrum and U rows
    # to preserve the output shapes expected by the existing GSVD code.
    u, s, v = svd(AB1, full_matrices=True, compute_uv=True)
    s_pad = np.zeros(p, dtype=s.dtype)
    s_pad[:s.shape[0]] = s
    beta = 1.0 / np.sqrt(1 + s_pad**2)
    alpha = s_pad * beta
    U = np.zeros((p, sze[0]), dtype=u.dtype)
    U[:u.shape[1], :] = u.T
    vb = v / beta[:, None]
    w2 = pinv(vb * bdiag[np.newaxis, :])
    return alpha, beta, U, v.T, w2


def dem_inv_gsvd_diag(A, bdiag):
    """Perform the GSVD of A with diagonal B defined by its diagonal entries."""
    return _dem_inv_gsvd_from_bdiag(A, bdiag)


def dem_inv_gsvd(A, B):
    """
    Perform the generalised singular value decomposition of ``A`` and ``B``.

    Parameters
    ----------
    A : ndarray
        Response-like matrix.
    B : ndarray
        Regularisation matrix.

    Returns
    -------
    alpha : array_like
        Diagonal values of ``SA``.
    beta : array_like
        Diagonal values of ``SB``.
    U : ndarray
        Left GSVD factor for ``A``.
    V : ndarray
        Left GSVD factor for ``B``.
    w2 : ndarray
        Right GSVD factor.
    """
    bdiag = np.diagonal(B)
    if np.all(B == np.diag(bdiag)):
        return _dem_inv_gsvd_from_bdiag(A, bdiag)
    # calculate the matrix A*B^-1
    AB1 = A@pinv(B)
    sze = AB1.shape
    p = max(sze)
    # Keep the historical output shapes while avoiding explicit padding of the
    # matrix passed into SVD.
    u, s, v = svd(AB1, full_matrices=True, compute_uv=True)
    # from the svd products calculate the diagonal components form the gsvd
    s_pad = np.zeros(p, dtype=s.dtype)
    s_pad[:s.shape[0]] = s
    beta = 1./np.sqrt(1+s_pad**2)
    alpha = s_pad*beta
    U = np.zeros((p, sze[0]), dtype=u.dtype)
    U[:u.shape[1], :] = u.T
    # calculate the w matrix
    w2 = pinv((v / beta[:, None]) @ B)
    # return gsvd products, transposing v as we do.
    return alpha, beta, U, v.T, w2
