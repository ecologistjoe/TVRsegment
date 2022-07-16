import pyximport; pyximport.install()
import numpy as np
import segmentation.cregions as regions
import segmentation.cSplitBregman as cSplitBregman
import math


def segment(A, MMU=10, alpha=10, conn=4, Mask=None, nodata=None):

    # Make a constant for minimum search
    INF = float('inf')

    # Use given mask or make one from nodata values
    if not Mask:
        Mask = np.not_equal(A, nodata)

    # Get an initial guess from TVR using the SplitBregman algorithm
    B = renorm(A, 2, Mask)
    L = cSplitBregman.TVR(B, 100/alpha, 1e-4);
    L = np.round(1+L, 3)
    L[~Mask] = 0

    # Assign unique labels to connected, same-valued regions
    L = regions.labelConnected(L, conn)
    S = regions.getProperties(L, B)

    # Merge regions
    for iter in xrange(100):
        noMerges = True

        order = np.argsort([x.Area for x in S])
        for i in order:

            if not S[i]: continue

            # Get set of all neighbors
            N = regions.findNeighbors(L, S[i], conn)
            if not N: continue

            # Find neighbor with least distance from this region
            mu_i = S[i].Mean
            minDiff = INF
            for n in N:
                d = (S[n].Mean - mu_i)**2
                if d < minDiff:
                    minDiff = d
                    nearest = n

            # Merge? Always for small patches, sometimes for similar ones
            merge = False
            if S[i].Area < MMU:
                merge = True
            else:
                # Only do these calculations if needed:
                minCVar = S[nearest].combinedVariance(S[i])
                maxVar = max(S[nearest].Var, S[i].Var)
                if (minCVar < maxVar) and (minDiff < minCVar):
                    merge = True

            # Merge regions
            if merge:
                p = np.transpose(S[i].PixelIdxList)
                L[p[0],p[1]] = nearest
                S[nearest] += S[i]
                S[i].empty()
                noMerges = False

        if noMerges: break

    # Remove 0-size regions
    L = regions.labelConnected(L, conn)
    return L



def renorm(A, strength, Mask):

    B = np.empty(A.shape)
    W = np.copy(A[Mask]);
    W = (W-np.mean(W)) / (strength*np.std(W));

    erf = np.vectorize(math.erf)
    B[Mask] = (1 + erf(W))/2
    B[~Mask] = -100;

    return B
