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


"""This module implements similarity measures between two (linear/non-linear) 
dynamical systems.
"""


__license__ = "Apache License, Version 2.0"
__author__  = "Roland Kwitt, Kitware Inc., 2013"
__email__   = "E-Mail: roland.kwitt@kitware.com"
__status__  = "Development"


import copy
import pickle
import numpy as np
import scipy.linalg
from termcolor import colored
from sklearn.utils.extmath import randomized_svd

# import module dsinfo from package dsutil
import dsutil.dsinfo as dsinfo

# import ErrorDS class
from dscore.dsexcp import ErrorDS


def nldsIP(nlds1, nlds2):
    """Inner product between NLDS feature spaces.
    
    Parameters:
    -----------
    nlds1 : nlds instance
        First NLDS model.
        
    nlds2 : nlds instance
        Second NLDS model.
        
    Returns:
    --------
    F : numpy.array, shape = (D, D)
        Inner product between KCPA components.
    """
    
    if not type(nlds1._kpcaParams._kPar) is type(nlds2._kpcaParams._kPar):
        raise ErrorDS('kernel types are incompatible!')
    
    # KPCA weight matrices
    A1 = nlds1._kpcaParams._A
    A2 = nlds2._kpcaParams._A
    
    N1 = A1.shape[0]
    N2 = A2.shape[0]
            
    if nlds1._kpcaParams._kPar._kCen:
        A1n = copy.copy(A1)
        A2n = copy.copy(A2)
        A1n -= np.tile(np.mean(A1, axis=0),(N1, 1))
        A2n -= np.tile(np.mean(A2, axis=0),(N2, 1))
  
    # copy kernel params of nlds1 and enforce NO centering/scaling
    kPar = copy.copy(nlds1._kpcaParams._kPar)
    kPar._kCen = False
    kPar._sig2 = 1
    
    # data of both NLDS's
    Y1 = nlds1._kpcaParams._data
    Y2 = nlds2._kpcaParams._data
    
    sig1s = np.sqrt(nlds1._kpcaParams._kPar._sig2)
    sig2s = np.sqrt(nlds2._kpcaParams._kPar._sig2)
        
    # inner-product (in feature space) between Gaussian kernels (updates kPar)
    nlds1._kpcaParams._kFun(Y1/sig1s, Y2/sig2s, kPar)
    F = np.asmatrix(A1).T*kPar._kMat*A2
    return F
    
    
def nldsMartinDistance(nlds1, nlds2, N=20):
    """Martin distance between two NLDS's.
    
    The implemented algorithm computes an iterative solution to the 
    Martin distance between two NLDS's.
    
    Algorithmic outline:
    --------------------
    
    The idea of using subspace angles is based on the work of
    
    [1] K. De Cock and B. De Moor, "Subspace angles between ARMA models", In:
        Systen & Control Letters, vol 46, pp. 265-270, 2002

    and was extended to a specific kind of non-linear dynamical
    system in
    
    [2] A. Chan and N. Vasconcelos. "Classifying Video with Kernel Dynamic 
        Textures", In: CVPR (2007)
    
    Parameters:
    -----------
    nlds1: core.nlds instance
        First NLDS model.
    
    nlds2: core.lds instance
        Second NLDS model.
    
    N : int (default: 20)
        Number of iterations to compute the "infinite sum" that is the 
        solution to the Lyapunov equation.
    
    Returns:
    --------
    D : np.float32
        Martin distance between nlds1 and nlds2.
    """
    
    Ahat1 = nlds1._Ahat
    Ahat2 = nlds2._Ahat
    
    dx1 = len(nlds1._initX0)
    dx2 = len(nlds2._initX0)
    
    C1C2 = nldsIP(nlds1, nlds2)
    C1C1 = np.eye(dx1)
    C2C2 = np.eye(dx2)
    
    K = np.zeros((dx1+dx2, dx1+dx2))
    L = np.zeros((dx1+dx2, dx1+dx2))
    
    # N summation terms
    for i in range(N+1):
        if i == 0:
            O1O2 = C1C2
            O1O1 = C1C1
            O2O2 = C2C2
            a1t = Ahat1
            a2t = Ahat2
        else:
            O1O2 = O1O2 + a1t.T*C1C2*a2t
            O1O1 = O1O1 + a1t.T*C1C1*a1t
            O2O2 = O2O2 + a2t.T*C2C2*a2t
            if i != N-1:
                a1t = a1t*Ahat1
                a2t = a2t*Ahat2
                
        # we are at the end
        if i == N-1:
            K[0:dx1,dx1:] = O1O2
            K[dx1:,0:dx1] = O1O2.T
            L[0:dx1,0:dx1] = O1O1
            L[dx1:,dx1:] = O2O2
            ev = np.flipud(np.sort(np.real(scipy.linalg.eigvals(K, L))))
            if len(np.nonzero(ev)[0]) != len(ev):
                return np.inf
            else:
                return -2*np.sum(np.log(ev[0:dx1]))
                

def ldsMartinDistance(lds1, lds2, N=20):
    """Martin distance between two LDS's.
    
    Algorithmic outline:
    --------------------
    
    This code implements the subspace angle approach of
    
    [1] K. De Cock and B. De Moor, "Subspace angles between ARMA models", In:
        Systen & Control Letters, vol 46, pp. 265-270, 2002

    that is used in many works which implement the Martin distance as a 
    similarity measure linear dynamical systems.
    
    Parameters:
    -----------
    lds1: core.lds instance
        First LDS model.
    
    lds2: core.lds instance
        Second LDS model.
    
    N : int (default: 20)
        Number of iterations to compute the "infinite sum" that is the 
        solution to the Lyapunov equation (see code.)
    
    Returns:
    --------
    D : np.float32
        Martin distance between lds1 and lds2.
    """
    
    if not lds1.check() or not lds2.check():
        raise Exception("Models are incomplete!")
    
    # get relevant params
    C1 = lds1._Chat
    C2 = lds2._Chat
    A1 = lds1._Ahat
    A2 = lds2._Ahat
    
    C1C1 = np.asmatrix(C1).T*C1
    C2C2 = np.asmatrix(C2).T*C2
    C1C2 = np.asmatrix(C1).T*C2
    
    dx1 = len(lds1._initM0)
    dx2 = len(lds2._initM0)
    
    # matrices that are used for the GEP
    K = np.zeros((dx1+dx2, dx1+dx2))
    L = np.zeros((dx1+dx2, dx1+dx2))
    
    # N summation terms
    for i in range(N+1):
        if i == 0:
            O1O2 = C1C2
            O1O1 = C1C1
            O2O2 = C2C2
            a1t = A1
            a2t = A2
        else:
            O1O2 = O1O2 + a1t.T*C1C2*a2t
            O1O1 = O1O1 + a1t.T*C1C1*a1t
            O2O2 = O2O2 + a2t.T*C2C2*a2t
            if i != N-1:
                a1t = a1t*A1
                a2t = a2t*A2
                
        # we are at the end
        if i == N-1:
            K[0:dx1,dx1:] = O1O2
            K[dx1:,0:dx1] = O1O2.T
            L[0:dx1,0:dx1] = O1O1
            L[dx1:,dx1:] = O2O2
            ev = np.flipud(np.sort(np.real(scipy.linalg.eigvals(K,L))))
            if len(np.nonzero(ev)[0]) != len(ev):
                return np.inf
            else:
                return -2*np.sum(np.log(ev[0:dx1]))
            
            
            
    
    