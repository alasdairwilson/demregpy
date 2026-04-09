"""
Temperature response data.
"""
from pathlib import Path

import numpy as np

aia_tresp = Path(__file__).parent / 'data' / 'aia_tresp_en.dat'
aia_tresp_nb = Path(__file__).parent / 'data' / 'aia_tresp_en_nb.dat'
aia_tresp_v9 = Path(__file__).parent / 'data' / 'aia_trespv9_en.dat'


def load_aia_response(response_file=aia_tresp):
    """
    Load an AIA temperature response file into the form used by ``dn2dem``.

    Parameters
    ----------
    response_file : path-like, optional
        Path to an IDL ``.sav`` response file. Defaults to the bundled evenorm
        response.

    Returns
    -------
    channels : list[str]
        Channel names such as ``["A94", "A131", ...]``.
    tresp_logt : ndarray
        Log-temperature grid of the response file.
    trmatrix : ndarray
        Temperature response matrix with shape ``(nt, nf)``.
    """
    try:
        import scipy.io as io
    except ImportError as exc:
        raise ImportError(
            "load_aia_response requires scipy. Install demregpy with the "
            "'aia' or 'responses' extra to use bundled response files."
        ) from exc

    trin = io.readsav(response_file)
    channels = [ch.decode("utf-8") for ch in trin["channels"]]
    tresp_logt = np.array(trin["logt"])
    nf = len(trin["tr"][:])
    trmatrix = np.zeros((len(tresp_logt), nf))
    for i in range(nf):
        trmatrix[:, i] = trin["tr"][i]
    return channels, tresp_logt, trmatrix

__all__ = [
    'aia_tresp',
    'aia_tresp_nb',
    'aia_tresp_v9',
    'load_aia_response',
]
