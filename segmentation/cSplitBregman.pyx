# encoding: utf-8
# #cython: profile=True, linetrace=True, binding=True

import numpy as np
import math
import sys
from time import time
cimport cython

cpdef TVR(double [:,:] F, double mu, double tol, method="aniso", solver="gs"):

    cdef:
        int rows=F.shape[0]
        int cols=F.shape[1]
        double alpha = 1.0 / (2*mu)
        double err
        double [:,:] U = np.copy(F)
        double [:,:] U1
        double [:,:] X = np.zeros((rows-1,cols))
        double [:,:] bX = np.zeros((rows-1,cols))
        double [:,:] Y = np.zeros((rows,cols-1))
        double [:,:] bY = np.zeros((rows,cols-1))

        int i,j
        int iter
    
    
    # Determine solver, Either 'gs' or 'jac'
    s = solver.lower()
    if s != 'gs' and s != 'jac':
        raise ValueError('Solver "' + method + '" not recognized. Chose either "gs" for Gauss-Seidel or "jac" for Jacobi.')
    jacobi = (s=='jac')

    # Determine method, Either 'aniso' or 'iso'
    m = method[:2].lower()
    if m != 'an' and m != 'is':
        raise ValueError('Method "' + method + '" not recognized. Chose either "anisotropic" or "isotropic".')
    iso = (m=='is')


    tol = tol*tol

    for iter in range(100):
        U1 = np.copy(U);

        # Update U 
        if jacobi:
            #Jacobi method
            # Have to apply an even number of iterations (2)
            # or Split-Bregman won't converge
            U = jacU(U, X, bX, Y, bY, F)
            U = jacU(U, X, bX, Y, bY, F)
        else:
            # Gauss-Seidel
            U = gsU(U, X, bX, Y, bY, F)
            
        
        # Break if below tolerance
        if iter > 4:
            err = stopfunc(U, U1)
            if err < tol:
                break
            
            
        # Get deltas
        Udx = np.diff(U,axis=0);
        Udy = np.diff(U,axis=1);

        if iso:
            # gsSpace
            X,Y = flexiso(Udx+bX, Udy+bY, alpha)
        else:
            # gsX, gsY
            X = flex2d(Udx + bX, alpha)
            Y = flex2d(Udy + bY, alpha)

        # bregmanX, bregmanY
        bX += Udx - X;
        bY += Udy - Y;
    
    return np.asarray(U)


cdef double stopfunc(double [:,:] new, double [:,:] old):
    #np.sum((U-U1)**2) / np.sum(U1*U1)
    cdef:
        double ssq=0
        double s=0
        double d
        
    for i in range(old.shape[0]):
        for j in range(old.shape[1]):
            d = new[i,j]-old[i,j]
            ssq += d*d
            s += old[i,j]*old[i,j]
            
            
    return ssq/s

    
#
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
    
@cython.boundscheck(False)
cdef double[:,:] jacU(double [:,:] U, double [:,:] X, double [:,:] bX, double [:,:] Y, double [:,:] bY, double [:,:] F):
    cdef:
        int i,j
        double s
        int rows = U.shape[0]
        int cols = U.shape[1]
        double [:,:] U1 = np.zeros((rows,cols))
        
    for i in xrange(1,rows-1):
        for j in xrange(1,cols-1):
            s = X[i-1,j] - X[i,j] + Y[i,j-1] - Y[i,j]  - bX[i-1,j] + bX[i,j] - bY[i,j-1] + bY[i,j]
            s += U[i+1,j] + U[i-1,j] + U[i,j+1] + U[i,j-1]
            U1[i,j] = (2.0*s+F[i,j]) / 9.0;

    i = 0
    for j in xrange(1,cols-1):
        s = -X[i,j] + Y[i,j-1] - Y[i,j]  + bX[i,j] - bY[i,j-1] + bY[i,j]
        s += U[i+1,j]  + U[i,j+1] + U[i,j-1]
        U1[i,j] = (2.0*s+F[i,j]) / 7.0;

    i=rows-1
    for j in xrange(1,cols-1):
        s = X[i-1,j] + Y[i,j-1] - Y[i,j]  - bX[i-1,j]  - bY[i,j-1] + bY[i,j]
        s += U[i-1,j] + U[i,j+1] + U[i,j-1]
        U1[i,j] = (2.0*s+F[i,j]) / 7.0;

    j=0
    for i in xrange(1,rows-1):
        s = X[i-1,j] - X[i,j]  - Y[i,j]  - bX[i-1,j] + bX[i,j]  + bY[i,j]
        s += U[i+1,j] + U[i-1,j] + U[i,j+1]
        U1[i,j] = (2.0*s+F[i,j]) / 7.0;


    j=cols-1
    for i in xrange(1,rows-1):
        s = X[i-1,j] - X[i,j] + Y[i,j-1]   - bX[i-1,j] + bX[i,j] - bY[i,j-1]
        s += U[i+1,j] + U[i-1,j] + U[i,j-1]
        U1[i,j] = (2.0*s+F[i,j]) / 7.0;

    i=0; j=0
    s = - X[i,j] - Y[i,j] + bX[i,j] + bY[i,j]
    s += U[i+1,j] + U[i,j+1]
    U1[i,j] = (2.0*s+F[i,j]) / 5.0;

    i=0; j=cols-1
    s = - X[i,j] + Y[i,j-1] + bX[i,j] - bY[i,j-1]
    s += U[i+1,j] + U[i,j-1]
    U1[i,j] = (2.0*s+F[i,j]) / 5.0;

    i=rows-1; j=0
    s = X[i-1,j] - Y[i,j] - bX[i-1,j] + bY[i,j]
    s += + U[i-1,j] + U[i,j+1]
    U1[i,j] = (2.0*s+F[i,j]) / 5.0;

    i=rows-1; j=cols-1
    s = X[i-1,j] + Y[i,j-1] - bX[i-1,j] - bY[i,j-1]
    s += + U[i-1,j] + U[i,j-1]
    U1[i,j] = (2.0*s+F[i,j]) / 5.0;

    return U1
    
    
@cython.boundscheck(False)
cdef double[:,:] gsU(double [:,:] U, double [:,:] X, double [:,:] bX, double [:,:] Y, double [:,:] bY, double [:,:] F) nogil:

    cdef:
        int i,j
        double s
        int rows = U.shape[0]
        int cols = U.shape[1]

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



cdef double[:,:] flex2d(double[:,:] B, double alpha):
    cdef:
        int rows = B.shape[0]
        int cols = B.shape[1]
        double[:,:] X = np.zeros((rows,cols))
        int i, j
        double b

    for i in range(rows):
        for j in range(cols):
            b = B[i,j]
            if b > alpha:
                X[i,j] = b-alpha
            elif b < -alpha:
                X[i,j] = b+alpha

    return X


cdef double[:] flex1d(double[:] B, double alpha):
    cdef:
        int rows = B.shape[0]
        double[:] X = np.zeros(rows)
        int i, j
        double b

    for i in range(rows):
        b = B[i]
        if b > alpha:
            X[i] = b-alpha
        elif b < -alpha:
            X[i] = b+alpha

    return X



cdef flexiso(double[:,:] dx, double[:,:] dy, double alpha):

    cdef:
        int rows = dx.shape[0]
        int cols = dy.shape[1]
        double [:,:] X =np.zeros((rows, cols+1))
        double [:,:] Y=np.zeros((rows+1, cols))
        double a, b, s
        double a2 = alpha*alpha

    for i in xrange(rows):
        for j in xrange(cols):
            a = dx[i,j]
            b = dy[i,j]
            s = a*a + b*b
            if s > a2:
                s = 1 - alpha/math.sqrt(s)
                X[i,j] = a * s
                Y[i,j] = b * s

    X[:,-1] = flex1d(dx[:,-1], alpha)
    Y[-1,:] = flex1d(dy[-1,:], alpha)


    return X,Y
