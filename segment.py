#!/usr/bin/env python

from segmentation import segment, regions
from argparse import ArgumentParser
from osgeo import gdal, osr
import numpy as np

def write_geotiff(fn, data, model_ds, PixelType=gdal.GDT_UInt32, CO=[]):
    bands = 1
    Ysize, Xsize = data.shape

    driver = gdal.GetDriverByName('GTiff');
    map_ds = driver.Create( fn, Xsize, Ysize, bands, PixelType, ["BIGTIFF=YES","TILED=YES"]+CO)
    map_ds.SetGeoTransform(model_ds.GetGeoTransform() )
    projection = osr.SpatialReference()
    projection.ImportFromWkt(model_ds.GetProjectionRef())
    map_ds.SetProjection(projection.ExportToWkt())
    band = map_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.WriteArray(data)


if __name__ == "__main__":

    p = ArgumentParser()
    p.add_argument('-a','--alpha', default=10, type=int, help="The level of denoising in initial patch-creation. Higher values make larger patches")
    p.add_argument('-m','--mmu', default=10, type=int, help="The minimum mapping unit; defines the smallest size in pixels patches can be.")
    p.add_argument('-n', '--nodata', default=None, type=float, help="Specify nodata value. If not given, the value in the INFILE metadata is used.")
    p.add_argument('-c', '--conn', choices=[4,8], type=int, default=4, help="Use 4- or 8- connection when determining patch linkage.")
    p.add_argument('-s', '--stats', default=None, nargs=2, metavar=('DATA_FILE', 'OUTPUT_FILE'), help="When specified, the mean and variance of the data specified in DATA_FILE will be calculated within each labeled region for each band and written as a comma separated list to OUTPUT_FILE.")
    p.add_argument('infile', help="The file to create patches from. If infile is a multi-band file, patches are created from the first band.")
    p.add_argument('outfile', nargs='?', default='', help="GeoTiff Filename where patch labels will be saved.")
    args = p.parse_args()


    if not args.outfile:
        args.outfile = '.'.join(args.infile.split('.')[:-1]) + '_labels.tif'

    ds = gdal.Open(args.infile)
    band = ds.GetRasterBand(1)

    if args.nodata is None:
        nodata = band.GetNoDataValue()
    else:
        nodata = args.nodata

    A = band.ReadAsArray()
    L = segment(A, args.mmu, args.alpha, args.conn, nodata=nodata)
    write_geotiff(args.outfile, L, ds)

    if args.stats:
        with open(args.stats[1], 'w+') as outfile:
            outfile.write('LABEL,BAND,COUNT,SUM,MEAN,VARIANCE\n')

            ds = gdal.Open(args.stats[0])
            for b in range(ds.RasterCount):
                B = ds.GetRasterBand(b+1).ReadAsArray()
                R = regions.getProperties(L, B.astype(np.double))

                for i,r in enumerate(R):
                    if r.Area==0: continue
                    outfile.write("{label},{band},{count},{sum},{mean:0.6g},{var:0.6g}\n".format(
                        label=i,
                        band=b+1,
                        count=r.Area,
                        sum=r.Sum,
                        mean=r.Mean,
                        var=r.Var))
