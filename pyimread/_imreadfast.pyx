#!python
#cython: initializedcheck=False, boundscheck=False, overflowcheck=False

import cython
cimport cython
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t
from cpython cimport bool
import numpy as _np
cimport numpy as _np
_np.import_array()
import multiprocessing as _mpc

cdef extern from "imagesread.h":
    cdef cppclass images_read:
        images_read() except +
        images_read(const char*, const char*, IMAGE_PREDICTORS_TYPE) except +
        int multiThreadReadImages(void*, uint16_t)
        IMAGE_DATA_TYPE getImageType()
        uint64_t getWidth()
        uint64_t getHeight()
        uint64_t getDepth()
        uint64_t getFilesNum()
        int getDiffImages(void* , uint16_t)
        void multiThreadImagesWirte(void* , const char* , const char* , uint16_t )

    cdef enum IMAGE_DATA_TYPE:
        UINT8_TYPE = 0
        UINT16_TYPE = 1
        UINT32_TYPE = 2
        UINT64_TYPE = 3
        INT8_TYPE = 4
        INT16_TYPE = 5
        INT32_TYPE = 6
        INT64_TYPE = 7
        FLOAT32_TYPE = 8
        FLOAT64_TYPE = 9

    cdef enum IMAGE_PREDICTORS_TYPE:
        NO_PREIDICTORS = 0,
        PREIDCTORS_A = 1,
        PREIDCTORS_B = 2,
        PREIDCTORS_C = 3,
        PREIDCTORS_APB_DC = 4,
        PREIDCTORS_A_BDC_Div2 = 5,
        PREIDCTORS_B_ADC_Div2 = 6,
        PREIDCTORS_APB_Div2 = 7,
        PREIDCTORS_APB_Div2_Exten = 8

#cdef int readImages(const char* path, const char* suffix, void* im, int numThreads)
cdef class pyImagesRead:
    cdef images_read pyread

    def __cinit__(self, basestring imagespath, basestring suffix, IMAGE_PREDICTORS_TYPE ptype = PREIDCTORS_APB_Div2):
        self.pyread = images_read(_cstr(imagespath), _cstr(suffix), ptype)

    def getImageData(self, threads = _mpc.cpu_count()):
        cdef _np.ndarray imageData
        width = self.pyread.getWidth()
        height = self.pyread.getHeight()
        depth = self.pyread.getDepth()
        filesNum = self.pyread.getFilesNum()
        print(width, height, depth, filesNum, self.pyread.getImageType())
        if self.pyread.getImageType() == UINT8_TYPE:
            imageData = _np.empty([filesNum,depth,height,width], _np.dtype(_np.uint8))
            self.pyread.multiThreadReadImages(imageData.data, threads)
            return imageData
        elif self.pyread.getImageType() == UINT16_TYPE:
            imageData = _np.empty([filesNum,depth,height,width], _np.dtype(_np.uint16))
            self.pyread.multiThreadReadImages(imageData.data, threads)
            return imageData

    def getDiffImage(self, threads = _mpc.cpu_count()):
        cdef _np.ndarray imageData
        width = self.pyread.getWidth()
        height = self.pyread.getHeight()
        depth = self.pyread.getDepth()
        filesNum = self.pyread.getFilesNum()
        print(width, height, depth, filesNum, self.pyread.getImageType())
        if self.pyread.getImageType() == UINT8_TYPE:
            imageData = _np.empty([filesNum,depth,height,width], _np.dtype(_np.uint8))
            self.pyread.getDiffImages(imageData.data, threads)
            return imageData
        elif self.pyread.getImageType() == UINT16_TYPE:
            imageData = _np.empty([filesNum,depth,height,width], _np.dtype(_np.uint16))
            self.pyread.getDiffImages(imageData.data, threads)
            return imageData

    def writeImageData(self, _np.ndarray imageData, basestring path, basestring suffix, threads = _mpc.cpu_count()):

        if self.pyread.getImageType() == UINT8_TYPE:
            self.pyread.multiThreadImagesWirte(imageData.data, _cstr(path), _cstr(suffix), threads)
        elif self.pyread.getImageType() == UINT16_TYPE:
            self.pyread.multiThreadImagesWirte(imageData.data,  _cstr(path), _cstr(suffix), threads)


cdef inline const char* _cstr(basestring pstr):
    if isinstance(pstr, unicode):
        return (<unicode>pstr).encode('utf8')
    else:
        return pstr