################################################################################
#
# Library: pydstk
#
# Copyright 2010 Kitware Inc. 28 Corporate Drive,
# Clifton Park, NY, 12065, USA.
#
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 ( the "License" );
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################


"""pydstk's I/O and utility routines.
"""


__license__ = "Apache License, Version 2.0"
__author__  = "Roland Kwitt, Kitware Inc., 2013"
__email__   = "E-Mail: roland.kwitt@kitware.com"
__status__  = "Development"


import time
import cv2
import cv2.cv as cv
import numpy as np

# import pyinfo module
import dsinfo

# import ErrorDS class from dsexcp module in dscore package
from dscore.dsexcp import ErrorDS


class Timer(object):
    """Class to make timing as simple as possible.
    """
    
    def __init__(self, name=None):
        self._name = name

    def __enter__(self):
        self._tstart = time.time()

    def __exit__(self, type, value, traceback):
        str = ''
        if self._name:
            str += 't(%s)=' % self._name
        str += '%s [sec]' % (time.time() - self._tstart)
        dsinfo.time(str)


def orth(A):
    """Creates an orthonormal basis for the range of the input matrix.
    
    Parameters
    ----------
    A : numpy.ndarray, shape (N, D)
        Input matrix.
    
    Returns
    -------
    Q : numpy.ndarray, shape (N, E)
        Orthonormal basis for range(A), i.e., Q'Q = I; E = rank(Q)
    """
    
    U,S,_ = np.linalg.svd(A, full_matrices=False)
    m,n = A.shape
    if m > 1:
        pass
    elif m == 1:
        S = S(1)
    else:
        S = 0
    tol = np.max((m,n))*np.max(S)*np.spacing(1)
    Q = U[:,0:len(np.where(S > tol)[0])]
    return Q
    

def renormalize(data, (newmin, newmax), oldrange=None):
    """Normalize data into requested range.
    
    Parameters
    ----------
    data : numpy array, shape = (N, D)
        Input data.
        
    (newmin, newmax) : tuple of min. and max. value
        The range we want the values to be in.
    
    oldrange : tupe of user-specified input range
        Input range to use (Note: no clipping is done)
    
    Returns
    -------
    out : numpy array, shape = (N, D)
        Scaled output data.
    """
    
    data = data.astype('float32')
    if oldrange is None:
        (oldmin, oldmax) = (np.min(data), np.max(data))
    else:
        (oldmin, oldmax) = oldrange
    slope = (newmin-newmax+0.)/(oldmin-oldmax)
    out = slope*(data-oldmin) + newmin
    return out    
    
    
def showMovie(frames, movSz, fps=20, transpose=False):
    """Show a movie using OpenCV.
    
    Takes a numpy matrix (with images as columns) and shows the images in 
    a video at a specified frame rate.
    
    Parameters
    ----------
    frames : numpy array, shape = (N, D)
        Input data with N images as D-dimensional column vectors.
        
    movSz : tupe of (height, width, nFrames)
        Size of the movie to show. This is used to reshape the column vectors
        in the data matrix.

    fps : int (default: 20)
        Show video at specific frames/second (FPS) rate.

    transpose : boolean (defaukt : False)
        Transpose each frame.
    """

    if fps < 0:
        raise Exception("FPS < 0")
    
    video = frames.reshape(movSz)
    nImgs = frames.shape[1]
    tWait = int(1000.0/fps);

    for i in range(nImgs):
        if transpose:
            cv2.imshow("video", renormalize(video[:,:,i].T, (0, 1)))
        else:
            cv2.imshow("video", renormalize(video[:,:,i], (0, 1)))
            
        key = cv2.waitKey(tWait)
        if key == 27: 
            break
    
    
def loadDataFromVideoFile(inFile):
    """Read an AVI video into a data matrix.
    
    Parameters
    ----------
    inFile : string
        Name of the AVI input video file (might be color - if so, it will be 
        converted to grayscale).
        
    Returns
    -------
    dataMat : numpy array, shape = (N, D)
        Output data matrix, where N is the number of pixel in each of the D
        frames.
        
    dataSiz : tuple of (height, width, D)    
        The video dimensions.
    """

    capture = cv2.VideoCapture(inFile)

    flag, frame = capture.read()
    if flag == 0:
        raise ErrorDS("Could not read %s!" % inFile)
    
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    N = np.prod(frame.shape)
    D = int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT))
    (height, width) = frame.shape

    #dataMat = np.zeros((N, D), dtype=np.float32)
    dataMat = np.zeros((64*64,D), dtype=np.float32)
    newF = cv2.resize(frame, (64,64))
    
    dataMat[:,0] = newF.reshape(-1) #frame.reshape(-1)
    
    cnt = 1
    while True:
        flag, frame = capture.read()
        if flag == 0:
            break
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        newF = cv2.resize(frame, (64,64))
        dataMat[:,cnt] = newF.reshape(-1) #frame.reshape(-1)
        cnt += 1  

    return (dataMat, (64,64,D))
    #return (dataMat, (height, width, D))
    

def loadDataFromASCIIFile(inFile):
    """Read an ASCII file into a data matrix.
    
    Reads a data matrix from a file. The first line of the ASCII file needs
    to contain a header of the form (HEIGHT WIDTH FRAMES), e.g.,
    
    2 2 3
    1.2 3.4 5.5
    1.1 1.1 2.2
    4.4 1.1 2.2
    3.3 3.3 1.1
    
    The header information will be returned to the user so that the data
    matrix can later be reshaped into an actual video.
    
    Parameters
    ----------
    inFile : string
        Input file name.
    
    Returns
    -------
    dataMat : numpy array, shape = (N, D)
        Output data matrix, where N = (width x height)
        
    dataSiz : tuple of (height, width, D)    
        The video dimensions.
    """

    with open(inFile) as fid:
        header = fid.readline()
    
    dataSiz = [int(x) for x in header.split(' ')]
    if len(dataSiz) != 3:
        raise ErrorDS("invalid header when reading %s!" % inFile)
    
    dataMat = np.genfromtxt(inFile, dtype=np.float32, skip_header=1)    
    return (dataMat, dataSiz)
    
   
def loadDataFromVolumeFile(inFile):
    """Load a volumetric image as a video.
    
    Reads a volumetric image and creates a video, where the z-axis is defined
    as the time-axis. 
    
    Note: The spatial dimensions have to be equal!
    
    Paramters
    ---------
    inFile : string
        Filename of input image (e.g., test.mha)
    
    Returns
    -------   
    dataMat : numpy.ndarray, shape = (N, #images)
        Output data matrix, where N = (width x height)
    
    dataSiz : tuple of (height, width, #images)    
      The video dimensions.
    """
    
    import SimpleITK as sitk
    data = sitk.GetArrayFromImage(sitk.ReadImage(inFile))

    zDim = data.shape[0]
    xDim = data.shape[1]
    yDim = data.shape[2]
    
    if not (xDim == yDim):
        raise ErrorDS("spatial dimensions of video ")
        
    dataMat = data.reshape((zDim,xDim*yDim)).T
    dataSiz = (xDim, yDim, zDim)
    return (dataMat, dataSiz)

    
def loadDataFromIListFile(inFile):
    """Read list of image files into a data matrix.
    
    Reads a file with absolute image filenames, e.g.,
    
    /tmp/im0.png
    /tmp/im1.png
    ...
    
    into a data matrix with images as column vectors. 
    
    Parameters
    ----------
    inFile : string
        Input file name.
        
    Returns
    -------
    dataMat : numpy array, shape = (N, #images)
        Output data matrix, where N = (width x height)
        
    dataSiz : tuple of (height, width, #images)    
        The video dimensions.
    """
    import SimpleITK as sitk
    
    with open(inFile) as fid:
        fileNames = fid.readlines()
    
    dataMat = None
    for i, imFile in enumerate(fileNames):
        img = sitk.ReadImage(imFile.rstrip())        
        mat = sitk.GetArrayFromImage(img)
        
        if len(mat.shape) > 2:
            raise ErrorDS("Only grayscale images are supported!")
        
        if dataMat is None:
            dataMat = np.zeros((np.prod(mat.shape),len(fileNames)))
        dataMat[:,i] = dat.reshape(-1)
    
    return (dataMat, (mat.shape[0], mat.shapep[1], len(fileNames)))    
        
        
        
        
        
