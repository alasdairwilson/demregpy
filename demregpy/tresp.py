"""
Temperature response data.
"""
from pathlib import Path

aia_tresp = Path(__file__).parent / 'data' / 'aia_tresp_en.dat'
aia_tresp_nb = Path(__file__).parent / 'data' / 'aia_tresp_en_nb.dat'
aia_tresp_v9 = Path(__file__).parent / 'data' / 'aia_trespv9_en.dat'

__all__ = [
    'aia_tresp',
    'aia_tresp_nb',
    'aia_tresp_v9',
]
