import numpy as np
from numpy.linalg import inv
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
    #calculate the weighting matrix
    w2=inv(inv(onea)@u.T@A)
    #this verification step to check w and w2 recovered from u and v respectively match, turns out LAPACK 
    # svd is returns the hermitian transpsoe of v and not v itself
    #w2 and w should be (and are in tests) identical
    w=inv(inv(oneb)@v@B)
    #return gsvd products, transposing v as we do.
    return alpha,beta,u,v.T,w,w2

x=np.array([[1,3,5,9],[2,4,-1,2],[1,7,-3,9],[4,-1,1,14]])
y=np.array([[2,7,1,3],[12,3,-2,-1],[-2,-4,5,1],[-1,7,-1,-3]])


a,b,u,v,wt,w2=dem_inv_gsvd(x,y)

#testing gsvd prodcuts
print(np.isclose(u@np.diag(a)@inv(wt),x))
print(np.isclose(v@np.diag(b)@inv(wt),y))

