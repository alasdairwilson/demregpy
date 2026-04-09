"""
Utilities for building synthetic DEM observations from response matrices.
"""

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SyntheticObservation",
    "synthesize_counts",
]


@dataclass(slots=True)
class SyntheticObservation:
    """
    Synthetic observation returned by :func:`synthesize_counts`.

    The fields store the input DEM, the corresponding noise-free counts, the
    counts used as inversion input, and their uncertainties.
    """

    dem: np.ndarray
    dn_clean: np.ndarray
    dn_in: np.ndarray
    edn_in: np.ndarray


def _logt_bin_widths(logt):
    logt = np.asarray(logt, dtype=float)
    if logt.ndim != 1:
        raise ValueError("logt must be one-dimensional")
    if logt.size < 2:
        raise ValueError("logt must contain at least two temperature points")

    edges = np.empty(logt.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (logt[:-1] + logt[1:])
    edges[0] = logt[0] - 0.5 * (logt[1] - logt[0])
    edges[-1] = logt[-1] + 0.5 * (logt[-1] - logt[-2])
    return np.diff(edges)


def synthesize_counts(
    dem,
    logt,
    tresp,
    *,
    error_fraction=0.1,
    noise_fraction=None,
    min_fraction=0.1,
    random_state=None,
):
    """
    Fold a DEM through a temperature response matrix and build synthetic counts.

    Parameters
    ----------
    dem : array_like
        DEM values on the ``logt`` grid. The last axis must match ``logt``.
        Leading dimensions are preserved in the output.
    logt : array_like
        Log10 temperature grid corresponding to the first axis of ``tresp``.
    tresp : array_like
        Temperature response matrix with shape ``(nt, nf)``.
    error_fraction : float or array_like, optional
        Relative uncertainty applied to the clean counts when building
        ``edn_in``. Defaults to ``0.1``.
    noise_fraction : float or array_like, optional
        Relative Gaussian noise applied to the clean counts. If omitted, the
        returned ``dn_in`` is noise-free.
    min_fraction : float, optional
        Lower bound applied to noisy counts as a fraction of the clean counts.
    random_state : int or numpy.random.Generator, optional
        Random seed or generator used when ``noise_fraction`` is provided.

    Returns
    -------
    SyntheticObservation
        Synthetic DEM, clean counts, input counts, and uncertainties.
    """

    dem = np.asarray(dem, dtype=float)
    logt = np.asarray(logt, dtype=float)
    tresp = np.asarray(tresp, dtype=float)

    if tresp.ndim != 2:
        raise ValueError("tresp must have shape (nt, nf)")
    if tresp.shape[0] != logt.size:
        raise ValueError("The first axis of tresp must match the length of logt")
    if dem.shape[-1] != logt.size:
        raise ValueError("The last axis of dem must match the length of logt")

    dlogt = _logt_bin_widths(logt)
    kernel = tresp * (10 ** logt * np.log(10 ** dlogt))[:, np.newaxis]
    dn_clean = np.tensordot(dem, kernel, axes=([-1], [0]))

    if noise_fraction is None:
        dn_in = np.array(dn_clean, copy=True)
    else:
        rng = random_state
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(random_state)
        perturb = rng.normal(loc=0.0, scale=noise_fraction, size=dn_clean.shape)
        dn_in = dn_clean * (1.0 + perturb)
        dn_in = np.maximum(dn_in, min_fraction * dn_clean)

    edn_in = np.asarray(error_fraction, dtype=float) * dn_clean

    return SyntheticObservation(
        dem=dem,
        dn_clean=dn_clean,
        dn_in=dn_in,
        edn_in=edn_in,
    )
