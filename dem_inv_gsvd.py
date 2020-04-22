import numpy as np
from numpy.linalg import inv
import pprint
def dem_inv_gsvdcsq(A,B):
    AB1=np.matmul(A,inv(B)) # need to check with IGH if this is correct
    u,s,v = np.linalg.svd(AB1,full_matrices=True,compute_uv=True)
#run gscvd from lapack uisng f2py here (Actually just using numpy now)
    beta=1./np.sqrt(1+s**2)
    alpha=s*beta
    #diagonalise alpha and beta
    onea=np.diag(alpha)
    oneb=np.diag(beta)

    w2=inv(inv(onea)@np.transpose(u)@A)
    #this verification step to check w and w2 recovered from u and v respectively match, turns out LAPACK 
    # svd is returns the hermitian transpsoe of v and not v itself
    w=inv(inv(oneb)@v@B)
    return alpha,beta,u,np.transpose(v),w,w2

x=np.array([[1,3,5,9],[2,4,-1,2],[1,7,-3,9],[4,-1,1,14]])
y=np.array([[2,7,1,3],[12,3,-2,-1],[-2,-4,5,1],[-1,7,-1,-3]])


a,b,u,v,wt,w2=dem_inv_gsvdcsq(x,y)

print(a)
print(b)
print(u)
print(v)
print(wt)
print(w2)




