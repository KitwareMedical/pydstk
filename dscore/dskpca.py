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

"""Customized kernel PCA (KPCA) for NLDS.
"""


__license__ = "Apache License, Version 2.0"
__author__  = "Roland Kwitt, Kitware Inc., 2013"
__email__   = "E-Mail: roland.kwitt@kitware.com"
__status__  = "Development"


import sys
import time
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import euclidean_distances

# import dsinfo module from dsinfo package
import dsutil.dsinfo as dsinfo

# import ErrorDS and Timer class
from dscore.dsexcp import ErrorDS
from dsutil.dsutil import Timer


class RBFParam:
    """Class for RBF kernel parametes.
    """
    
    def __init__(self):
        self._kCen = None
        self._kMat = None
        self._sig2 = None
        self._trS0 = None
        self._trS1 = None
        self._teS0 = None


class KPCAParam:
    """Class for KPCA parameters.
    
    Member variables are:
    
        _A : numpy.array, shape = (N, k) - KPCA weight matrix
        _l : numpy.array, shape = (k, )  - Eigenvalues of kernel matrix
        _kPar : Kernel parameters (depends on kernel)
        _kFun : Kernel function (depends on kernel)
    """

    def __init__(self):
        self._A = None
        self._l = None
        self._kPar = None
        self._kFun = None
        self._data = None


def rbfK(X, Y, params):
    """RBF kernel.
    
    Compute a centered RBF kernel K_ij = exp(-||x_i - y_j||^2/sigma2), 
    where 'x_i' and 'y_j' are D-dimensional signals (i.e., the i-th and
    j-th column of the input data matrices X and Y) and 'sigma2' is the 
    kernel width.
    
    In case the setting for sigma2 is empty, we compute sigma2 as 
    sigma2 = median {||x_i - y_j||^2}_{ij}).
    
    Parameters:
    -----------
    X : numpy array, shape = (N, D)
        D N-dimensional input vectors.

    Y : numpy array, shape = (M, D)
        D M-dimensional input vectors.
        
    params : RBFParam instance
        The parameters that will be updated when calling the kernel function. 
        The following fields will always be set:
        
            _kMat : numpy.array, shape = (D, D)
                The kernel matrix.

        The following fields need to be set by the user:
            
            _kCen : boolean
                Do we need to center the kernel.

        If _kCen is set to True, then the following fields will be updated.       

            _trS0 : numpy.array, shape = (D,) - Sum over kernel columns
            _trS1 : float - Sum over all kernel values

        In case we are computing a testing kernel, the following fields will
        also be updated when _kCen is True:
        
            _teS0 : numpy.array, shape = (D,) - Sum over kernel rows
            
        If the field _sig2 is set, it will be used as the RBF kernel width;
        If it is not set (None), it will be computed (see above) and the 
        field will be updated.
    """
    
    if params._kCen is None:
        raise ErrorDS('centering parameter invalid!')
    
    if not isinstance(params, RBFParam):
        raise ErrorDS('wrong parameters for RBF kernel!')
    
    # checks if we compute a TRAINING kernel
    isTrain = False
    if X is Y:
        isTrain = True
     
    # compute pairwise (squared) Eucl. distances   
    dMat = euclidean_distances(X.T, Y.T, squared=True)
    
    if params._sig2 is None:
        params._sig2 = np.median(dMat.ravel())

    # computes RBF kernel
    kMat = np.exp(-1.0*dMat/params._sig2)
    
    # do we need centering?
    if params._kCen:
        n, m = kMat.shape

        if isTrain: # X == Y
            trS0 = np.sum(kMat, axis=1)/n
            trS1 = np.sum(kMat.ravel())/n**2
            
            kMat = (kMat - np.tile(trS0, (n, 1)) - 
                    np.tile(np.asmatrix(trS0).T, (1, n)) + 
                    trS1)
            params._trS0 = trS0
            params._trS1 = trS1
            
        else: # X != Y
            if params._trS0 is None or params._trS1 is None:
                raise ErrorDS('some training kernel values are unavailable!')
                
            trS0 = params._trS0
            trS1 = params._trS1 
            teS0 = (1/m)*np.sum(kMat, axis=1)
            kMat = kMat - np.tile(trS0, (n, 1)) - np.tile(teS0, (1, m)) + trS1
            params._teS0 = teS0
    
    params._kMat = kMat


def normalize(A, l, tol=1e-6):
    """Normalize KPCA weight vectors.
    
    Given that A = [a_0, ..., a_K] is a matrix of K D-dimensional 
    eigenvectors (columns of A) and l_0, ..., l_K are the corresponding 
    eigenvalues obtained from an eigenanalysis of the kernel matrix K, 
    then each weight vector a_j is normalized by \hat{a}_j = a_{j} * 
    1/sqrt(l_j). 
    
    NOTE: The routine UPDATES the columns of the A matrix, i.e., the
    entries are updated in place!
    
    Parameters:
    -----------
    A : numpy.array, shape = (N, K)
        Matrix of eigenvectors (alphas).
        
    l : numpy.array, shape = (K, )
        Vector of eigenvalues (lambdas).
    """
    n, m = A.shape

    ltTol = np.where(np.abs(l)<tol)[0]
    if len(ltTol) > 0:
        geTol = np.where(np.abs(l)>=tol)[0]
        A[:,geTol] /= np.tile(np.sqrt(l[geTol]), (n, 1))
    else:    
        A /= np.tile(np.sqrt(l), (n, 1))
    
    
def kpca(Y, k, params):
    """KPCA driver.
    
    Runs KPCA on the input data matrix and UPDATES the KPCA parameters given
    by the user. See 
    
    [1] B. Schoelkopf, A. Smola, and K. R. Muller. "Nonlinear component analysis
        as a kernel eigenvalue problem", Neural Computation, vol. 10, 
        pp. 1299-1319, 1998
    
    for technical details on KPCA.
    
    Parameters:
    -----------
    Y : numpy array, shape = (N, D)
        Input matrix of D N-dimensional signals.
    
    k : int
        Compute k KPCA components.
    
    params : KPCAParam instance
        KPCA parameters. 
        
        Upon completion, the params is updated. The following fields
        are set:
        
            _data : numpy.array, shape = (N, D) - Original data
            _A : numpy.array, shape = (N, k)    - KPCA weight matrix
            _l : numpy.array, shape = (k,)      - Eigenvalues of kernel matrix
    
        The following fields need to be set already:
        
            _kPar : Kernel parameters (depends on the kernel)
            _kFun : Kernel function   (depends on the kernel)
        
        Since the kernel will be called interally, the kernel parameters
        will also be updated (see kernel documentation).
        
    Returns:
    --------
    Xhat : numpy array, shape (k, D)
        NLDS state parameters.
    """
    
    if not isinstance(params, KPCAParam):
        raise ErrorDS('wrong KCPA parameters!')
    
    if (params._kPar is None or params._kFun is None):
        raise ErrorDS('KPCA not properly configured!')
    
    # save data
    params._data = Y
    
    # calls kernel fun
    params._kFun(Y, Y, params._kPar)
    kpcaObj = KernelPCA(kernel="precomputed")
    kpcaObj.fit(params._kPar._kMat)

    params._A = kpcaObj.alphas_[:,0:k]
    params._l = kpcaObj.lambdas_[0:k]

    if np.any(np.where(kpcaObj.lambdas_ <= 0)[0]):
        dsinfo.warn("some unselected eigenvalues are negative!")
    if np.any(np.where(params._l < 0)[0]):
        dsinfo.warn("some eigenvalues are negative!")

    # normalize KPCA weight vectors
    normalize(params._A, params._l)   
    return params._A.T*params._kPar._kMat
