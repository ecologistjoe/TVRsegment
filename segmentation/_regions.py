import numpy as np

class Region:
    def __init__(self, X=([],[])):
    
        self.PixelIdxList = []
        self.Sum = 0
        self.SumSquares = 0

        if X: self += X
    
    def __iadd__(self, x):
        if isinstance(x, Region):
            self.PixelIdxList.extend(x.PixelIdxList)
            self.Sum += x.Sum
            self.SumSquares += x.SumSquares
        elif isinstance(x, tuple):
            idx = x[0]
            val = x[1]
            if hasattr(val, '__iter__'):
                self.PixelIdxList.extend(idx)
                self.Sum += np.sum(val)
                self.SumSquares += np.sum([v*v for v in val])
            else:
                self.PixelIdxList.append(idx)
                self.Sum += val
                self.SumSquares += val*val
        
        return self
    
    def __len__(self):
        return len(self.PixelIdxList)
    
    def __cmp__(self,x):
        return len(self.PixelIdxList)-x
    
    def __repr__(self):
        return "Sum={0:0.3f}, SumSquares={1:0.3f}, Area={2:0.3f}, PixelIdxList={3}".format(self.Sum, self.SumSquares, self.Area, self.PixelIdxList)
    
    @property
    def Area(self):
        return len(self.PixelIdxList)
    
    @property    
    def Mean(self):
        return self.Sum / self.Area
   
    @property
    def Var(self):
        return (self.SumSquares + (self.Sum**2)/self.Area)/self.Area
    
    # Returns the variance of the combined regions
    def combinedVariance(self, x):  
        N = self.Area + x.Area
        SSq = self.SumSquares + x.SumSquares
        SS = self.Sum + x.Sum
        return SSq/N  - (SS/N)**2
    
    # Remove all pixels from region
    def empty(self):
        self.PixelIdxList = []
        self.Sum = 0
        self.SumSquares = 0
        

def getProperties(O, B):

    N = np.max(O)
    rows, cols = O.shape
    S = [Region() for n in xrange(N+1)]
    for i in xrange(rows):
        for j in xrange(cols):
            S[O[i,j]] += ((i,j), B[i,j])

    return S

    
def labelConnected(A, conn):

    rows, cols = A.shape
    
    # A matrix to hold labels
    L = np.zeros((rows,cols), np.uint32)
    
    # A list of labels, with each value pointing either
    # to itself, or to a smaller-valued n equivillent label.
    N = []
    
    
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
                        N[Lw] = [Ln]

                # Check this pixel and it's NW diagonal
                Lh = L[i,j]
                Lnw = L[i-1,j-1]
                if (A[i,j] == A[i-1,j-1]) and (Lh != Lnw):
                    if Lh < Lnw:
                        N[Lnw] = N[Lh]
                    else:
                        N[Lh] = N[Lnw]
                        

    ## Pass 2: Relabel equiv. nodes
    N = np.array(N)
    for i in xrange(rows):
        for j in xrange(cols):
            k = L[i,j]
            # Search for lowest-valued equiv. node
            if N[k]!=k:
                s = [k]
                while N[k] != k:
                    k = N[k]
                    s.append(k) 
                N[s] = k    # memoize search
            
                L[i,j] = k
    

    return L
    
    
def findNeighbors(L, region, conn):

    rows, cols = L.shape
    R = rows-1
    C = cols-1
    
    N = set()
    for x,y in region.PixelIdxList:
    
        if x > 0:  N.add(L[(x-1,y)])
        if y > 0:  N.add(L[(x,y-1)])
        if x < R:  N.add(L[(x+1,y)])
        if y < C:  N.add(L[(x,y+1)])
        
        if conn == 8:
            if x>0 and y>0:  N.add(L[(x-1,y-1)])
            if x>0 and y<C:  N.add(L[(x-1,y+1)])
            if x<R and y>0:  N.add(L[(x+1,y-1)])
            if x<R and y<C:  N.add(L[(x+1,y+1)])
    
    i = L[region.PixelIdxList[0]]
    if 0 in N: N.remove(0)
    if i in N: N.remove(i)
    return N

        