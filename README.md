# TVRsegment

A python library to identify and label patches of similarly-valued regions
in a single-band source image using total variation regularization followed
by merging to a minimum mapping unit.

```
    from segmentation import segment
    segment(image, mmu, alpha, conn)

    image: A 2d numpy array of intensity values to segment
    mmu: The minimum mapping unit; patches will not be created with fewer
        pixels than this size (suggested: 10)
    alpha: The intensity of de-noising to pass to TVR. Larger values remove
        more fine-scale detail and result in larger patches. (suggested: 10)
    conn: Either 4 or 8; the linkage for determining if patches are connected
```


Two additional Cython modules are included: cSplitBregman and cRegions.
Python modulees with similar names should not be trusted.

## cSplitBregman:
```
    from segmentation import TVR
    TVR(image, alpha, tol)

    image: The 2d numpy array of intensity values to denoise.
    alpha: The intensity of de-noising; larger values remove more fine-scale
        detail resulting in larger patches. Try 10 for intensities scaled
        between 0 and 1.
    tol: An iterative stopping criterion, specifying relative change.
        Try 1e-4 for intensities scaled between 0 and 1
```
TVR stands for Total Variation Regularization (aka ROF: Rudin Osher Fatemi
regularization). This module implements the Split Bregman algorithm
developed by Tom Goldstein described in "The Split Bregman Method for L1
Regularized Problems" ( Goldstein, 2009):
ftp://ftp.math.ucla.edu/pub/camreport/cam08-29.pdf

TVR performs image denoising while nominally preserving object edges. This
keeps the patches' shapes while homogenizing internal variance.



## cRegions:
This is a collection of functions for labeling and manipulating connected
same-valued regions in images.

```
    cRegions.labelConnected(X, int conn)
    Will create a label image giving each same-valued region in X a unique label

    cpdef list getProperties(unsigned int [:,:] L, double [:,:] B):
    Generates a list of properties (Count, Sum, Sum of Squares) of each labeled
    region specified in L using data from B. For example, if B is a biomass maps
    you can get the sum and variance of each labeled region in a list.
```

## segment.py:

This is a CLI script that uses the segmentation library to allow a user
to specify a gray-level image from which to create patches, 4 or 8 connectedness,
and the two patch-size parameters, alpha and MMU. It can also take a datafile
(either a different one or the same one used to make labels) and calculate stats
within each set of labels for each band within it. Note, the stats data file and
the file used to make the labels should have the same bounds and pixel size.

```
usage: segment.py [-h] [-a ALPHA] [-m MMU] [-n NODATA] [-c {4,8}]
                  [-s DATA_FILE OUTPUT_FILE]
                  infile [outfile]

positional arguments:
  infile                The file to create patches from. If infile is a multi-
                        band file, patches are created from the first band.
  outfile               GeoTiff Filename where patch labels will be saved.

optional arguments:
  -h, --help            show this help message and exit
  -a ALPHA, --alpha ALPHA
                        The level of denoising in initial patch-creation.
                        Higher values make larger patches
  -m MMU, --mmu MMU     The minimum mapping unit; defines the smallest size in
                        pixels patches can be.
  -n NODATA, --nodata NODATA
                        Specify nodata value. If not given, the value in the
                        INFILE metadata is used.
  -c {4,8}, --conn {4,8}
                        Use 4- or 8- connection when determining patch
                        linkage.
  -s DATA_FILE OUTPUT_FILE, --stats DATA_FILE OUTPUT_FILE
                        When specified, the mean and variance of the data
                        specified in DATA_FILE will be calculated within each
                        labeled region for each band and written as a comma
                        separated list to OUTPUT_FILE.
```