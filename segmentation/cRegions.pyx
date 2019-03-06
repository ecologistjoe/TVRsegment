# encoding: utf-8
# #cython: profile=True, linetrace=True, binding=True

import numpy as np
cimport cython


cdef class Region:

    cdef public:
        list PixelIdxList
        double Sum, SumSquares
        unsigned int Area


    def __init__(self, X=None):
        self.PixelIdxList = []
        self.Sum = 0
        self.SumSquares = 0
        self.Area = 0

        if X: self += X

    def __iadd__(self, x):
        if isinstance(x, Region):
            self.PixelIdxList += x.PixelIdxList
            self.Sum += x.Sum
            self.SumSquares += x.SumSquares
        elif isinstance(x, tuple):
            idx = x[0]
            val = x[1]
            if hasattr(val, '__iter__'):
                self.PixelIdxList += [idx]
                v = np.array(val)
                self.Sum += np.sum(v)
                self.SumSquares += np.sum(v*v)
            else:
                self.PixelIdxList.append(idx)
                self.Sum += val
                self.SumSquares += val*val

        self.Area = len(self.PixelIdxList)
        return self

    def __len__(self):
        return self.Area

    def __repr__(self):
        return "Sum={0:0.3f}, SumSquares={1:0.3f}, Area={2:0.3f}, PixelIdxList={3}".format(self.Sum, self.SumSquares, self.Area, self.PixelIdxList)

    @property
    def Mean(self):
        return self.Sum / self.Area

    @property
    def Var(self):
        return (self.SumSquares - (self.Sum**2)/self.Area)/self.Area

    # Returns the variance of the combined regions
    def combinedVariance(self, x):
        N = self.Area + x.Area
        SSq = self.SumSquares + x.SumSquares
        SS = self.Sum + x.Sum
        return (SSq - SS**2/N)/N

    # Remove all pixels from region
    def empty(self):
        self.PixelIdxList = []
        self.Sum = 0
        self.SumSquares = 0
        self.Area = 0

# Generate the properties of each labeled region (L) using data from B
cpdef list getProperties(unsigned int [:,:] L, double [:,:] B):

    cdef:
        int rows = L.shape[0]
        int cols = L.shape[1]
        unsigned int N = np.max(L)
        list S

    S = [Region() for n in xrange(N+1)]
    for i in xrange(rows):
        for j in xrange(cols):
            S[L[i,j]] += ((i,j), B[i,j])

    return S

# Give each same-valued region in X a unique label
def labelConnected(X, int conn):

    cdef:
        double [:,:] A = np.array(X).astype(np.double)

        unsigned int x, n, i, j, k, Ln, Lw, Lnw, Lh
        int rows = A.shape[0]
        int cols = A.shape[1]

        # A matrix to hold labels
        unsigned int [:,:] L = np.zeros((rows,cols), np.uint32)

        # A list of labels, with each value pointing either
        # to itself, or to a smaller-valued n equivalent label.
        unsigned int [:] Na
        unsigned int [:] U
        list N=[], s=[]


    ## Initialize Labels matrix
    #  Top-left corner
    n = 0
    L[0,0] = n
    N.append(n)

    # top row
    for i in xrange(1,rows):
        if A[i,0] == A[i-1,0]:
            L[i,0] = L[i-1,0]
        else:
            n += 1
            L[i,0] = n
            N.append(n)

    # left column
    for j in xrange(1,cols):
        if A[0,j] == A[0,j-1]:
            L[0,j] = L[0,j-1]
        else:
            n += 1
            L[0,j] = n
            N.append(n)

    ## Pass 1: Find locally connected regions and
    # record any equiv. classes in nodes
    for i in xrange(1,rows):
        for j in xrange(1,cols):

            West = (A[i,j] == A[i-1,j])
            North = (A[i,j] == A[i,j-1])
            Lw = L[i-1,j]
            Ln = L[i,j-1]

            if West and North and (Lw != Ln):
                # Record larger-valued node as eq. to smaller-valued node
                # Set current label to same
                if Lw < Ln:
                    N[Ln] = Lw
                    L[i,j] = Lw
                else:
                    N[Lw] = Ln
                    L[i,j] = Ln
            elif West:
                L[i,j] = Lw
            elif North:
                L[i,j] = Ln
            else:
                n += 1
                L[i,j] = n
                N.append(n)

            # Update equiv. classes of same-valued diagonals
            if conn==8:

                # Check the diagonal formed by the N and W pixels to this one
                if (A[i-1,j] == A[i,j-1]) and (Lw != Ln):
                    if Lw < Ln:
                        N[Ln] = N[Lw]
                    else:
                        N[Lw] = N[Ln]

                # Check this pixel and it's NW diagonal
                Lh = L[i,j]
                Lnw = L[i-1,j-1]
                if (A[i,j] == A[i-1,j-1]) and (Lh != Lnw):
                    if Lh < Lnw:
                        N[Lnw] = N[Lh]
                    else:
                        N[Lh] = N[Lnw]

    ## Pass 2: Relabel equiv. nodes
    Na = np.array(N).astype(np.uint32)
    for i in xrange(rows):
        for j in xrange(cols):
            k = L[i,j]
            # Search for lowest-valued equiv. node
            if Na[k]!=k:
                s = [k]
                while Na[k] != k:
                    k = Na[k]
                    s.append(k)
                for x in s:
                    Na[x] = k    # memoize search
                L[i,j] = k

    ## Pass 3: Compact label values into a consecutive list
    # Is there a way to fold this into the above?
    U = np.unique(Na)
    for k in xrange(len(U)):
        Na[U[k]] = k

    for i in xrange(rows):
        for j in xrange(cols):
           L[i,j] = Na[L[i,j]]

    return np.asarray(L)



cpdef set findNeighbors(unsigned int [:,:] L,  region, int conn):

    cdef:
        int rows = L.shape[0]
        int cols = L.shape[1]
        int R = rows-1
        int C = cols-1
        int i
        int num_pixels = len(region.PixelIdxList)
        long x,y
        unsigned int this
        unsigned int [:,:] A = np.zeros((num_pixels,conn), np.uint32)
        set N

    for i in xrange(num_pixels):

        x,y = region.PixelIdxList[i]
        this = L[x,y]

        if x > 0:  A[i,0] = (L[(x-1,y)])
        if y > 0:  A[i,1] = (L[(x,y-1)])
        if x < R:  A[i,2] = (L[(x+1,y)])
        if y < C:  A[i,3] = (L[(x,y+1)])

        if conn == 8:
            if x>0 and y>0:  A[i,4] = (L[(x-1,y-1)])
            if x>0 and y<C:  A[i,5] = (L[(x-1,y+1)])
            if x<R and y>0:  A[i,6] = (L[(x+1,y-1)])
            if x<R and y<C:  A[i,7] = (L[(x+1,y+1)])

    N = set()
    for i in xrange(num_pixels):
        for j in xrange(conn):
            if (A[i,j] != 0) and (A[i,j] != this):
                N.add(A[i,j])


    return N
