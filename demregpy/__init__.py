#demregpy
from demregpy.dn2dem import dn2dem
def aia_tresp():
    import os
    aia_tresp = os.path.join(os.path.dirname(__file__),'tresp','aia_tresp_en.dat')
    return aia_tresp
