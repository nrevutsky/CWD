import argparse
import Image
import ImageDraw
import numpy
import subprocess

from osgeo import gdal
from osgeo.gdalconst import *
from scipy.interpolate import interp1d
from struct import unpack

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class Dataset():
    def __init__(self, Filename):
        self.Filename = Filename
        self.dataset = gdal.Open(Filename, GA_ReadOnly)
        self.Projection = self.dataset.GetProjection()
        self.Geotransform = self.dataset.GetGeoTransform()
        self.ListofBands = [self.dataset.GetRasterBand(i) for i in range(1, self.dataset.RasterCount + 1)]
        self.BandsArrays = [numpy.array(i.ReadAsArray(), numpy.int16) for i in self.ListofBands]
        self.xsize = self.ListofBands[0].XSize
        self.ysize = self.ListofBands[0].YSize
        self.datatype = self.ListofBands[0].DataType
        self.PanBanList = []

    def pansharpen(self, panname, SaveAllPANBands=True):
        pan = gdal.Open(panname, GA_ReadOnly)
        bigPan = pan.ReadAsArray()
        driver = gdal.GetDriverByName("GTiff")
        driver.Register()
        bandNums = [''.join('Band_%s.TIF' % (str(arg))) for arg in range(1, len(self.ListofBands) + 1)]  # NameofBands
        self.BandsArrays = [numpy.float64(Band.ReadAsArray()) for Band in self.ListofBands]  # BandsArrays as float64
        smooth = numpy.kron(sum(self.BandsArrays) / len(self.BandsArrays), numpy.ones([4, 4])) + 1e-9
        if SaveAllPANBands == False:
            self.PanBanList = [(numpy.nan_to_num(bigPan / (smooth * numpy.kron(img, numpy.ones([4, 4]))))) for img in
                               self.BandsArrays]
            return self.PanBanList
        if SaveAllPANBands == True:
            PanLayers = [driver.CreateCopy(i, pan, 0) for i in
                         bandNums]  # List of created files whith pan parameters using for WriteArray()
            for img, dest in zip(self.BandsArrays, PanLayers):
                newimg = bigPan / (smooth * numpy.kron(img, numpy.ones([4, 4])))
                newimg[newimg > 255] = 255
                band = dest.GetRasterBand(1)
                band.WriteArray(numpy.nan_to_num(newimg))
                band.SetProjection(self.Projection)
                band.SetGeoTransform(self.Geotransform)

    def SaveAllBands(self):
        # Save all Bands in another file named Band_()
        driver = gdal.GetDriverByName('GTiff')
        gdal_data_types = {'Byte': GDT_Byte,
                           'UInt16': GDT_UInt16,
                           'Int16': GDT_Int16,
                           'UInt32': GDT_UInt32,
                           'Int32': GDT_Int32,
                           'Float32': GDT_Float32,
                           'Float64': GDT_Float64}
        bandNAMES = []
        for i in self.ListofBands:
            xsize = i.XSize
            ysize = i.YSize
            values = i.ReadAsArray()
            DATATYPE = gdal_data_types[gdal.GetDataTypeName(self.ListofBands[0].DataType)]
            nameholder = ''.join('Band_%s.TIF' % (str(self.ListofBands.index(i) + 1)))
            output_dataset = driver.Create(str(nameholder), xsize, ysize, 1, DATATYPE)
            output_dataset.GetRasterBand(1).WriteArray(values)
            output_dataset.SetProjection(self.Projection)
            output_dataset.SetGeoTransform(self.Geotransform)

    def histequalization(self, im, bins=256):
        NoDatVal = im.GetNoDataValue()
        im = im.ReadAsArray()
        im = numpy.ma.masked_equal(im, NoDatVal, copy=False)
        shape = im.shape
        im = im.ravel()
        hist, bins = numpy.histogram(im.compressed(), bins=256, normed=True)
        cdf = hist.cumsum()
        cdf = 255 * cdf / cdf[-1]
        im_func = interp1d(bins[:-1], cdf, bounds_error=False)
        im = im_func(im)
        return im.reshape(shape)

    def SaveBAND(self, BAND, OUTPUT_NAME, DATATYPE=gdal.GDT_Byte, DRIVER="GTiff"):
        # Save a specific band or array
        if str(type(BAND)) != "<type 'numpy.ndarray'>":
            values = numpy.nan_to_num(BAND.ReadAsArray())
            XSize = BAND.XSize
            YSize = BAND.YSize
        elif str(type(BAND)) == "<type 'numpy.ndarray'>":
            values = numpy.nan_to_num(BAND)
            XSize = len(BAND[0])
            YSize = len(BAND)
        output_file = '%s' % (OUTPUT_NAME)
        driver = gdal.GetDriverByName(DRIVER)
        new_file = driver.Create(output_file, XSize, YSize, 1, DATATYPE)
        new_file.GetRasterBand(1).WriteArray(values)
        new_file.SetProjection(self.Projection)
        new_file.SetGeoTransform(self.Geotransform)
        return new_file

    def Info(self):
        UseGdalScript = subprocess.Popen('gdalinfo %s' % self.Filename, stdout=subprocess.PIPE, shell=True)
        out, err = UseGdalScript.communicate()
        return out

    def CompositeBands(self, R, G, B, alpha=None, DATATYPE=gdal.GDT_Byte, JPEG=False):
        # R,G,B,alpha - Arrays
        xsize = len(R[0])
        ysize = len(R)
        driver_GTiff = gdal.GetDriverByName("GTiff")
        TIF_dataset = driver_GTiff.Create("TMP.tiff", xsize, ysize, 4, DATATYPE)
        TIF_dataset.GetRasterBand(1).WriteArray(R)
        TIF_dataset.GetRasterBand(2).WriteArray(G)
        TIF_dataset.GetRasterBand(3).WriteArray(B)
        if alpha != None:
            TIF_dataset.GetRasterBand(4).WriteArray(alpha)
        TIF_dataset.SetProjection(self.Projection)
        TIF_dataset.SetGeoTransform(self.Geotransform)
        if JPEG == True:
            driver_JPEG = gdal.GetDriverByName("JPEG")
            JPEG_dataset = driver_JPEG.CreateCopy("TMP.JPEG", TIF_dataset)
            JPEG_dataset.SetProjection(self.Projection)
            JPEG_dataset.SetGeoTransform(self.Geotransform)

    def NDVI(self, RED, NIR, Numpy_array_datatype=numpy.float64, save=True, Gdal_image_datatype=gdal.GDT_Byte):
        # Normalized Difference Vegetation Index
        RED = RED.astype(Numpy_array_datatype)
        NIR = NIR.astype(Numpy_array_datatype)
        NDVI_numpy_array = ((NIR - RED) / (NIR + RED))
        NDVI_numpy_array = numpy.nan_to_num(NDVI_numpy_array)
        NDVI_numpy_array = (NDVI_numpy_array * 255).astype(numpy.int32)
        if save == True:
            self.SaveBAND(NDVI_numpy_array, 'TMP.tiff', Gdal_image_datatype)
        return NDVI_numpy_array

    def ARVI(self, RED, BLUE, NIR, gamma=0.69, Numpy_array_datatype=numpy.float64, save=True,
             Gdal_image_datatype=gdal.GDT_Byte):
        # Atmospherically Resistant Vegetation Index
        RED = RED.astype(Numpy_array_datatype)
        NIR = NIR.astype(Numpy_array_datatype)
        BLUE = BLUE.astype(Numpy_array_datatype)
        RE_BL = RED - (gamma * (BLUE - RED))
        ARVI_numpy_array = ((NIR - RE_BL) / (NIR + RE_BL))
        ARVI_numpy_array = (numpy.nan_to_num(ARVI_numpy_array) * 255)
        ARVI_numpy_array = ARVI_numpy_array.astype(numpy.int32)
        if save == True:
            self.SaveBAND(ARVI_numpy_array, 'TMP.tiff', Gdal_image_datatype)
        return ARVI_numpy_array

    def IOR(self, BLUE, GREEN, YELLOW, Numpy_array_datatype=numpy.float64, save=True,
            Gdal_image_datatype=gdal.GDT_Byte):
        # Iron Oxide Ratio
        BLUE = BLUE.astype(Numpy_array_datatype)
        GREEN = GREEN.astype(Numpy_array_datatype)
        YELLOW = YELLOW.astype(Numpy_array_datatype)
        IOR_numpy_array = ((GREEN * YELLOW) / (BLUE * 1000))
        IOR_numpy_array = (numpy.nan_to_num(IOR_numpy_array) * 255)
        IOR_numpy_array = IOR_numpy_array.astype(numpy.int32)
        if save == True:
            self.SaveBAND(IOR_numpy_array, 'TMP.tiff', Gdal_image_datatype)
        return IOR_numpy_array
