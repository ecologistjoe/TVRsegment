import numpy as np
from scipy.signal import convolve2d as conv2

## Convolution Kernals used for the sum of
#  convolutions implementation of Jacobi solver
#  These are all nominally divided by 81, but due
#  boundaries, the outer two rows/columns of pixels
#  with need different normalization constants:
#
# N = [ 45    55    59    59  ...  59    59    55    45
#       55    73    77    77  ...  77    77    73    55
#       59    77    81    81  ...  81    81    77    59
#       59    77    81    81  ...  81    81    77    59
#               ...           ...           ...
#       59    77    81    81  ...  81    81    77    59
#       59    77    81    81  ...  81    81    77    59
#       55    73    77    77  ...  77    77    73    55
#       45    55    59    59  ...  59    59    55    45  ]

kU = np.array([[0, 0, 4, 0, 0],
               [0, 8, 0, 8, 0],
               [4, 0,16, 0, 4],
               [0, 8, 0, 8, 0],
               [0, 0, 4, 0, 0]])#/81
kF = np.array([[0, 2, 0],
               [2, 9, 2],
               [0, 2, 0]])#/81
kY = np.array([[0, -4, 4, 0],
               [-4,-14,14,4],
               [0, -4, 4, 0]])#/81
kX = kY.T


def TVR(F, mu, tol, method="aniso", solver="gs"):

    
    # Determine method, Either 'aniso' or 'iso'
    m = method[:3].lower()
    if m != 'ani' and m != 'iso':
        raise ValueError('Method "' + method + '" not recognized. Chose either "anisotropic" or "isotropic".')
    iso = (m=='iso')
    
    
    # Determine solver, Either 'gs' or 'jac'
    solver = solver.lower()
    if len(solver) > 2: solver=solver[:3]
    if solver != 'gs' and solver != 'jac' and solver != 'con':
        raise ValueError('Solver "' + method + '" not recognized. Chose either "gs" for Gauss-Seidel, "jac" for Jacobi, or "conv" for convolution implementation of Jacobi.')

    # Precalc
    rows, cols = F.shape
    alpha = 1.0 / (2*mu)
    tol = tol**2
    
    # Initialize
    X = np.zeros((rows-1, cols));
    Y = np.zeros((rows, cols-1));
    bX = np.zeros((rows-1, cols));
    bY = np.zeros((rows, cols-1));
    
    # Initial guess
    U = np.copy(F)
    U1 = np.ones((rows,cols))
    
    # Build matrix of normalization constants for Jacobi solvers
    # Because these are structured, there are more memory efficient
    # ways to do this rather than build another MxN matrix: see
    # the Cython implementation for one example.
    if solver=='con':
        N = np.ones(U.shape, dtype=np.uint8)
        N = conv2(N, kU, 'same') + conv2(N, kF, 'same')
    elif solver=='jac':
        N = np.ones((rows, cols), dtype=np.uint8)
        N[1:,:]  += 2
        N[:-1,:] += 2
        N[:,1:]  += 2
        N[:,:-1] += 2
    

    # Perform Split-Bregman iterations. Typically takes around 20.
    # The stopping criteria is used for whole images. Something better
    # may be needed for an Earth-Engine implementation.
    for iter in range(100):

        U1 = np.copy(U);

        # gsU
        if solver=='jac':
            # Solve the linear system using the Jacobi method
            # Jacobi is slower (taking two passes) and uses twice
            # the memory, but is parallelizable
            # An even number (two) Jacobi reps are needed
            # to ensure Split Bregman converges
            U = jacobiU(U, X, bX, Y, bY, F, N)
            U = jacobiU(U, X, bX, Y, bY, F, N)
        elif solver=='con':
            # Both Jacobi reps implemented as a sum of convolutions
            U = jacobiConvolutionU(U, X, bX, Y, bY, F, N)
        else:
            # Solve system using Gauss-Seidel
            U = gsU(U, X, bX, Y, bY, F)

        # Stopping Criteria
        if (iter > 5):
            diff = np.sum((U-U1)**2) / np.sum(U1**2)
            if diff < tol :
                break

        # Calc deltas
        Udx = np.diff(U,axis=0);
        Udy = np.diff(U,axis=1);

        if iso:
            # gsSpace
            X,Y = shrinkIso(Udx, Udy, bX, bY, alpha)
        else:
            # gsX, gsY
            X = shrink(Udx + bX, alpha)
            Y = shrink(Udy + bY, alpha)

        # bregmanX, bregmanY
        bX += Udx - X;
        bY += Udy - Y;
    
        
    return U
    
  
    

def jacobiConvolutionU(U0, X, bX, Y, bY, F, N):
    # Two iterations of Jacobi method implemented using
    # a sum of convolutions
    
    U = conv2(U0, kU, 'same')      + \
        conv2(F, kF, 'same')       + \
        conv2(X-bX, kX)[1:-1,1:-1] + \
        conv2(Y-bY, kY)[1:-1,1:-1]
    
    return U/N
    
    
def jacobiU(U0, X, bX, Y, bY, F, N):
    U = np.zeros(U0.shape)
    U[:-1,:] += -X + bX + U0[1:, :]
    U[1:,:]  +=  X - bX + U0[:-1,:]
    U[:,:-1] += -Y + bY + U0[:, 1:]
    U[:,1:]  +=  Y - bY + U0[:,:-1]
    U = (2*U + F) / N
    return U

    
# A more memory efficient method that doesn't
# require building N
#def jacU(U0, X, bX, Y, bY, F):
#    U = np.zeros(U.shape)
#    U[:-1,:] += -X + bX + U0[1:, :]
#    U[1:,:]  +=  X - bX + U0[:-1,:]
#    U[:,:-1] += -Y + bY + U0[:, 1:]
#    U[:,1:]  +=  Y - bY + U0[:,:-1]
#    U = 2*U + F
#    U[1:-1,1:-1] /= 9.0
#    U[0,1:-1] /= 7.0
#    U[1:-1,0] /= 7.0
#    U[-1,1:-1] /= 7.0
#    U[1:-1,-1] /= 7.0
#    U[0,0] /= 5.0
#    U[0,-1] /= 5.0
#    U[-1,0] /= 5.0
#    U[-1,-1] /= 5.0
#    return U
    
    
    
# This is a direct python replication of Goldstein's gsU C function.
# It's very slow in python, but is great when cythonized.  There's
# no particularly good method to do Gauss-Seidel in a vectorized way
# because it relies on updating the matrix in place and using
# just-computed values to improve convergence rate.
def gsU(U, X, bX, Y, bY, F):

    rows, cols = U.shape
    for i in xrange(1,rows-1):
        for j in xrange(1,cols-1):
            s = X[i-1,j] - X[i,j] + Y[i,j-1] - Y[i,j]  - bX[i-1,j] + bX[i,j] - bY[i,j-1] + bY[i,j]
            s += U[i+1,j] + U[i-1,j] + U[i,j+1] + U[i,j-1]
            U[i,j] = (2.0*s+F[i,j]) / 9.0;
            
    i = 0
    for j in xrange(1,cols-1):
        s = -X[i,j] + Y[i,j-1] - Y[i,j]  + bX[i,j] - bY[i,j-1] + bY[i,j]
        s += U[i+1,j]  + U[i,j+1] + U[i,j-1]
        U[i,j] = (2.0*s+F[i,j]) / 7.0;    
    
    i=rows-1
    for j in xrange(1,cols-1):
        s = X[i-1,j] + Y[i,j-1] - Y[i,j]  - bX[i-1,j]  - bY[i,j-1] + bY[i,j]
        s += U[i-1,j] + U[i,j+1] + U[i,j-1]
        U[i,j] = (2.0*s+F[i,j]) / 7.0;
        
    j=0
    for i in xrange(1,rows-1):
        s = X[i-1,j] - X[i,j]  - Y[i,j]  - bX[i-1,j] + bX[i,j]  + bY[i,j]
        s += U[i+1,j] + U[i-1,j] + U[i,j+1] 
        U[i,j] = (2.0*s+F[i,j]) / 7.0;
    
    
    j=cols-1
    for i in xrange(1,rows-1):    
        s = X[i-1,j] - X[i,j] + Y[i,j-1]   - bX[i-1,j] + bX[i,j] - bY[i,j-1] 
        s += U[i+1,j] + U[i-1,j] + U[i,j-1]
        U[i,j] = (2.0*s+F[i,j]) / 7.0;
    
    i=0; j=0
    s = - X[i,j] - Y[i,j] + bX[i,j] + bY[i,j]
    s += U[i+1,j] + U[i,j+1]
    U[i,j] = (2.0*s+F[i,j]) / 5.0;
    
    i=0; j=cols-1
    s = - X[i,j] + Y[i,j-1] + bX[i,j] - bY[i,j-1] 
    s += U[i+1,j] + U[i,j-1]
    U[i,j] = (2.0*s+F[i,j]) / 5.0;  

    i=rows-1; j=0
    s = X[i-1,j] - Y[i,j] - bX[i-1,j] + bY[i,j]
    s += + U[i-1,j] + U[i,j+1]
    U[i,j] = (2.0*s+F[i,j]) / 5.0;
    
    i=rows-1; j=cols-1
    s = X[i-1,j] + Y[i,j-1] - bX[i-1,j] - bY[i,j-1]
    s += + U[i-1,j] + U[i,j-1]
    U[i,j] = (2.0*s+F[i,j]) / 5.0;
    
    return U
    
#
#cdef double[:,:] shrink2d(double[:,:] B, double alpha):
#    cdef:
#        int rows = B.shape[0]
#        int cols = B.shape[1]
#        double[:,:] X = np.zeros((rows,cols))
#        int i, j
#        double b
#
#    for i in range(rows):
#        for j in range(cols):
#            b = B[i,j]
#            if b > alpha:
#                X[i,j] = b-alpha
#            elif b < -alpha:
#                X[i,j] = b+alpha
#
#    return X
#
    
# Shrink values of B with magnitude greater than alpha by alpha.
def shrink(B, alpha):
    W = np.zeros(B.shape);
    
    L = B > alpha;
    W[L] = B[L]-alpha;
    
    L = B < -alpha;
    W[L] = B[L]+alpha;
    
    return W

# This gives better denoising, but for our purposes, the
# anisometric method tends to give better discretization
def shrinkIso(Udx, Udy, bX, bY, alpha):
 
    a = Udx[:,:-1] + bX[:,:-1]
    b = Udy[:-1,:] + bY[:-1,:]
    S = a*a + b*b
    L = S > alpha**2
    S[~L] = 0
    S[L] = 1-alpha/np.sqrt(S[L])
    
    # Get edges
    x = shrink(Udx[:,-1]+bX[:,-1], alpha).reshape(-1,1)
    y = shrink(Udy[-1,:]+bY[-1,:], alpha).reshape(1,-1)
    
    X = np.concatenate(( S*a , x ), axis=1)
    Y = np.concatenate(( S*b , y ), axis=0)
       
    return X,Y