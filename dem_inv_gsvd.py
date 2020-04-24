import numpy as np
from numpy.linalg import inv,pinv
import pprint
def dem_inv_gsvd(A,B):
    """
    dem_inv_gsvd

    Performs the generalised singular value decomposition of two matrices A,B.

    Inputs

    A:
        cross section matrix
    B:
        regularisation matrix (square)

    Performs

    the decomposition of:

        A=U*SA*W^-1
        B=V*SB*W^-1

        with gsvd matrices u,v and the weight W and diagnoal matrics SA and SB

    Outputs

    U:
        decomposition product matrix
    V:
        decomposition prodyct matrix
    W:
        decomposition prodyct matrix
    alpha:
        the vector of the diagonal values of SA
    beta:
        the vector of the diagonal values of SB
  

    """  
    #calculate the matrix A*B^-1
    AB1=A@inv(B)

    #use np.linalg.svd to calculate the singular value decomposition
    u,s,v = np.linalg.svd(AB1,full_matrices=True,compute_uv=True)

    
    #from the svd products calculate the diagonal components form the gsvd
    beta=1./np.sqrt(1+s**2)
    alpha=s*beta
    #diagonalise alpha and beta into SA and SB
    onea=np.diag(alpha)
    oneb=np.diag(beta)
    print(u.shape,onea.shape,A.shape)
    #calculate the weighting matrix
    w=pinv(inv(onea)@inv(u)@A)
    print(np.isclose(u@onea@pinv(w),A))
    #return gsvd products, transposing v as we do.
    return alpha,beta,u,v.T,w
# x=np.random.randn(16,6)
# y=np.random.randn(16,16)

# a,b,u,v,wt=dem_inv_gsvd(x,y)

#testing gsvd prodcuts
# print(np.isclose(u@np.diag(a)@pinv(wt),x))
