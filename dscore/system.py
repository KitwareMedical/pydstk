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


"""Contains the core dynamical system implementations.
"""


__license__ = "Apache License, Version 2.0"
__author__  = "Roland Kwitt, Kitware Inc., 2013"
__email__   = "E-Mail: roland.kwitt@kitware.com"
__status__  = "Development"


import copy
import time
import pickle
import numpy as np
from collections import deque
from termcolor import colored
from sklearn.utils.extmath import randomized_svd

# import pyds package contents
import dsutil.dsinfo as dsinfo
import dsutil.dsutil as dsutil

# import pyds classes
from dsutil.dsutil import Timer
from dscore.dsexcp import ErrorDS
from dscore.dskpca import kpca, KPCAParam, rbfK, RBFParam


class NonLinearDS(object):
    """Non-linear dynamical system class.
    
    Implements parameter estimation for non-linear dynamical systems of the 
    form: 

        x_{t+1} = Ax_{t} + v_{t}
        y_{t}   = C(x_{t}) + w_{t}

    This code implements the non-linear dynamical systems approach to video 
    classification (referred to as "Kernel Dynamic Textures") proposed in:

    [1] A. Chan and N. Vasconcelos. "Classifying Video with Kernel Dynamic 
        Textures", In: CVPR (2007)
    """
    
    def __init__(self, nStates, kpcaParams, verbose=False):
        """Initialize nlds instance.
        
        
        nStates : int 
            Number of KDT states.
        
        kpcaParams : KPCAParam instance
            Configured KPCA parameters.
            
        verbose : boolean (default : False)
            Do we want verbose output ?
        """
        
        self._Ahat = None
        self._Rhat = None
        self._Qhat = None
        self._Xhat = None
        self._initX0 = None
        self._initM0 = None
        self._initS0 = None

        self._kpcaParams = kpcaParams
        self._nStates = nStates
        self._verbose = verbose

        self._ready = False

    @staticmethod
    def naiveCompare(nlds1, nlds2):
        """Compare NLDS parameters (in a naive Frobenius norm manner).
        
        Parameters:
        -----------
        nlds1 : NonLineDS instance
            Target LDS
        
        nlds2: NonLinearDS instance
            Source LDS
        
        Returns:
        --------
        err : float
            Sum of the Frobenius norms of the difference matrices.
        """
        
        err = (np.linalg.norm(nlds1._Ahat - nlds2._Ahat, 'fro') +
               np.linalg.norm(nlds1._Qhat - nlds2._Qhat, 'fro') +
               np.linalg.norm(nlds1._Xhat - nlds2._Xhat, 'fro') +
               np.linalg.norm(nlds1._initX0 - nlds2._initX0) +
               np.linalg.norm(nlds1._initM0 - nlds2._initM0) +
               np.linalg.norm(nlds1._initS0 - nlds2._initS0))
        return err
        
    
    def check(self):
        """Check validity of LDS parameters.
     
        Currently, this routine only checks if the parameters are set, but not
        if they are actually valid parameters!
        
        Returns:
        --------
        validity : boolean
            True if parameters are valid, False otherwise.
        """
        
        for key in self.__dict__:
           if self.__dict__[key] is None: 
               return False
        return True

    
    def suboptimalSysID(self, Y):
        """System identification using KPCA.
    
        Updates the NLDS parameters.
        
        Parameters:
        -----------
        Y : numpy array, shape = (N, D)
            Input data.
        """

        nStates = self._nStates
                 
        # call KPCA to get state estimate
        if self._verbose:
            with Timer('kpca'):
                Xhat = kpca(Y, nStates, self._kpcaParams)
        else:
            Xhat = kpca(Y, nStates, self._kpcaParams)
            
        # estimate rest of parameters
        _, tau = Y.shape
        
        Ahat = Xhat[:,1:tau]*np.linalg.pinv(Xhat[:,0:tau-1])
        Vhat = Xhat[:,1:tau]-Ahat*Xhat[:,0:tau-1]
        Qhat = (Vhat*Vhat.T)/(tau-1)
        initX0 = Xhat[:,0]
        initM0 = np.mean(Xhat, axis=1)
        initS0 = np.diag(np.cov(Xhat))
   
        self._Rhat = 0
        self._Ahat = Ahat
        self._Xhat = Xhat
        self._Vhat = Vhat
        self._Qhat = Qhat
        self._initX0 = initX0
        self._initM0 = initM0
        self._initS0 = initS0
        
        
class LinearDS(object):
    """Implements a linear dynamical system (LDS) of the form:
    
    x_{t+1} = A*x_{t} + w_{t}
    y_{t}   = C*x_{t} + v_{t}
    
    Parameter details (in terms of matrix dimensions):
    
        x_{t} : [k x 1] - State vector at time t
        y_{t} : [N x 1] - Obervation vector at time t
        w_{t} : [k x 1] - State noise at time t
        v_{t} : [N x 1] - Observation noise at time t
        A     : [k x k] - State transition matrix
        C     : [N x k] - Observation matrix
    """
    
    def __init__(self, nStates, approx=False, verbose=False):
        """Initialization.
        
        Parameters:
        -----------
        nStates : int
            Number of LDS states.
            
        approx : boolean (default : False)
            Use randomized SVD.
        
        verbose : boolean (default : False)
            Verbose output.
        """
    
        self._Ahat = None
        self._Chat = None
        self._Rhat = None
        self._Qhat = None
        self._Xhat = None
        self._Yavg = None
        self._initM0 = None
        self._initS0 = None
        
        self._approx = approx
        self._verbose = verbose
        self._nStates = nStates
        
        if self._nStates < 0:
            raise ErrorDS("#states < 0!")

        self._ready = False
        
    
    def check(self):
        """Check validity of LDS parameters.
     
        Currently, this routine only checks if the parameters are set, but not
        if they are actually valid parameters!
        
        Returns:
        --------
        validity : boolean
            True if parameters are valid, False otherwise.
        """
        
        for key in self.__dict__:
           if self.__dict__[key] is None: 
               return False
        return True
      
   
    def synthesize(self, tau=50, mode=None):
        """Synthesize obervations.
        
        Parameters
        ----------
        tau : int (default = 50)
            Synthesize tau frames. 
            
        mode : Combination of ['s','q','r']
            's' - Use the original states
            'q' - Do NOT add state noise
            'r' - Add observations noise

            In case 's' is specified, 'tau' is ignored and the number of 
            frames equals the number of state time points.
            
        Returns
        -------
        I : numpy array, shape = (D, tau)
            Matrix with N D-dimensional column vectors as observations.
            
        X : numpy array, shape = (N, tau) 
            Matrix with N tau-dimensional state vectors.        
        """
        
        if not self._ready:
            raise ErrorDS("LDS not ready for synthesis!")
        
        Bhat = None
        Xhat = self._Xhat
        Qhat = self._Qhat
        Ahat = self._Ahat
        Chat = self._Chat
        Rhat = self._Rhat
        Yavg = self._Yavg
        initM0 = self._initM0
        initS0 = self._initS0
        nStates = self._nStates
        
        if mode is None:
            raise ErrorDS("No synthesis mode specified!")
        
        # use original states -> tau is restricted
        if mode.find('s') >= 0:
            tau = Xhat.shape[1]
        
        # data to be filled and returned     
        I = np.zeros((len(Yavg), tau))
        X = np.zeros((nStates, tau))
        
        if mode.find('r') >= 0:
            stdR = np.sqrt(Rhat)
        
        # add state noise, unless user explicitly decides against
        if not mode.find('q') >= 0:
            stdS = np.sqrt(initS0)
            (U, S, V) = np.linalg.svd(Qhat, full_matrices=False)
            Bhat = U*np.diag(np.sqrt(S)) 
    
        t = 0 
        Xt = np.zeros((nStates, 1))
        while (tau<0) or (t<tau):  
            # uses the original states
            if mode.find('s') >= 0:
                Xt1 = Xhat[:,t]
            # first state
            elif t == 0:
                Xt1 = initM0;
                if mode.find('q') < 0:
                    Xt1 += stdS*np.rand(nStates)
            # any further states (if mode != 's')
            else:
                Xt1 = Ahat*Xt
                if not mode.find('q') >= 0:
                    Xt1 = Xt1 + Bhat*np.rand(nStates)
            
            # synthesizes image
            It = Chat*Xt1 + np.reshape(Yavg,(len(Yavg),1))
         
            # adds observation noise
            if mode.find('r') >= 0:
                It += stdR*np.randn(length(Yavg))
            
            # save ...
            Xt = Xt1;
            I[:,t] = It.reshape(-1)
            X[:,t] = Xt.reshape(-1)
            t += 1
            
        return (I, X)
    
    
    def suboptimalSysID(self, Y):
        """Suboptimal system identification using SVD.
        
        Suboptimal system identification based on SVD, as proposed in the 
        original work of Doretto et al. [1]. 
        
        Parameters
        ----------
        Y : numpy array, shape = (N, D)
            Input data with D observations as N-dimensional column vectors.
        """
        
        nStates = self._nStates
        
        if self._verbose:
            dsinfo.info("using suboptimal SVD-based estimation!")

        (N, tau) = Y.shape
        Yavg = np.mean(Y, axis=1)
        Y = Y - Yavg[:,np.newaxis]
        
        if self._approx:
            if self._verbose:
                with Timer('randomized_svd'):
                    (U, S, V) = randomized_svd(Y, nStates)
            else:
                (U, S, V) = randomized_svd(Y, nStates)
        else:
            if self._verbose:
                with Timer('np.linalg.svd'):
                    (U, S, V) = np.linalg.svd(Y, full_matrices=0)
            else:
                (U, S, V) = np.linalg.svd(Y, full_matrices=0)
                
        Chat = U[:,0:nStates]
        Xhat = (np.diag(S)[0:nStates,0:nStates] * np.asmatrix(V[0:nStates,:]))
    
        initM0 = np.mean(Xhat[:,0], axis=1)
        initS0 = np.zeros((nStates, 1))

        pind = range(tau-1);

        phi1 = Xhat[:,pind]
        phi2 = Xhat[:,[i+1 for i in pind]]
        
        Ahat = phi2*np.linalg.pinv(phi1)
        Vhat = phi2-Ahat*phi1;
        Qhat = 1.0/Vhat.shape[1] * Vhat*Vhat.T 
         
        errorY = Y - Chat*Xhat
        Rhat = np.var(errorY.ravel())
        
        # save parameters
        self._initS0 = initS0
        self._initM0 = initM0
        self._Yavg = Yavg
        self._Ahat = Ahat
        self._Chat = Chat
        self._Xhat = Xhat
        self._Qhat = Qhat
        self._Rhat = Rhat
                
        if self.check():
            self._ready = True
 
 
    @staticmethod
    def stateSpaceMap(lds1, lds2):
        """
        Map parameters from lds1 into space of lds2 (state-space).
        
        Parameters:
        -----------
        lds1 : lds instance
            Target LDS
        
        lds2: lds instance
            Source LDS
        
        Returns:
        --------
        lds : lds instance
            New instance of lds2 (with UPDADED parameters)
            
        err : float
            Absolute difference between the vectorized parameter sets before
            the state-space mapping.
        """
        
        # make a shallow copy (no compound object -> no problem)
        lds = copy.copy(lds2)

        Chat1 = lds1._Chat
        Chat2 = lds2._Chat
       
        F = np.asmatrix(np.linalg.pinv(Chat2))*Chat1
     
        # compute TRANSFORMED params (rest should be kept the same)
        lds._Chat = lds2._Chat*F
        lds._Ahat = F.T*lds2._Ahat*F
        lds._Qhat = F.T*lds2._Qhat*F
        lds._Rhat = lds2._Rhat
        lds._initM0 = F.T*lds2._initM0
        lds._initS0 = np.diag(F.T*np.diag(lds._initS0.ravel())*F)
        
        err = 0
        err += np.sum(np.abs(lds2._Chat.ravel() - lds1._Chat.ravel()))
        err += np.sum(np.abs(lds2._Ahat.ravel() - lds1._Ahat.ravel()))
        err += np.sum(np.abs(lds2._Qhat.ravel() - lds1._Qhat.ravel()))
        err += np.sum(np.abs(lds2._Rhat.ravel() - lds1._Rhat.ravel()))
        err += np.sum(np.abs(lds2._initM0.ravel() - lds1._initM0.ravel()))                        
        err += np.sum(np.abs(lds2._initS0.ravel() - lds1._initS0.ravel()))                        
        err += np.sum(np.abs(lds2._Yavg.ravel() - lds1._Yavg.ravel()))                        
        return (lds, err)


class OnlineNonLinearDS(NonLinearDS):
    """Online version of non-linear DS (for real-time use).
    """

    def __init__(self, nStates, kpcaParam, bufLen, nShift=1, verbose=False):
        """ Initialization.
        
        Parameters:
        -----------
        nStates : int
            Number of NLDS states.
            
        kpcaParam: instance of KPCAParam
            KPCA parameters.
        
        bufLen : int
            Length of circular buffer to hold data vectors.
            
        nShift : int (default : 1)
            Shift window by N vectors forward.
            
        verbose : boolean (default : False)
            Verbose output.
        """
    
        if nShift == 0:
            raise ErrorDS('nShift == 0!')
        NonLinearDS.__init__(self, nStates, kpcaParam, verbose)
        
        self._buf = deque(maxlen = bufLen)
        [self._buf.append(None) for i in range(bufLen)]
            
        self._nShift = nShift
        self._cnt = nShift - 1
        
        
    def update(self, x):
        """Update NLDS model (i.e., re-estimate if required)
        
        Parameters:
        -----------
        x : numpy.array, shape = (N, )
            New data vector.
        """
        
        self._buf.append(x)
            
        if self._buf.count(None) > 0:
            return
        self._cnt -= 1
        
        if self._cnt == 0 or self._nShift == 1:
            self.suboptimalSysID(np.asarray(self._buf).T)
            self._cnt = self._nShift
        

class OnlineLinearDS(LinearDS):
    """Online version of a linear DS (for real-time use).
    """
    
    def __init__(self, nStates, bufLen, nShift=1, approx=False, verbose=False):
        """ Initialization.
        
        Parameters:
        -----------
        nStates : int
            Number of LDS states.
            
        bufLen : int
            Length of circular buffer to hold data vectors.
            
        nShift : int (default : 1)
            Shift window by N vectors forward.
            
        approx : boolean (default : False)
            Use randomized SVD.
            
        verbose : boolean (default : False)
            Verbose output.
        """
            
        if nShift == 0:
            raise ErrorDS('nShift == 0!')
                
        # call base class init
        LinearDS.__init__(self, nStates, approx, verbose)
            
        # initialize buffer and fill with None's
        self._buf = deque(maxlen = bufLen)
        [self._buf.append(None) for i in range(bufLen)]
            
        self._nShift = nShift
        self._cnt = nShift - 1
            
            
    def update(self, x):
        """Update LDS model (i.e., re-estimate if required)
        
        Parameters:
        -----------
        x : numpy.array, shape = (N, )
            New data vector.
        """
            
        self._buf.append(x)
            
        # rampup time ... do nothin
        if self._buf.count(None) > 0:
            return
        
        self._cnt -= 1
        
        if self._cnt == 0 or self._nShift == 1:
            self.suboptimalSysID(np.asarray(self._buf).T)
            self._cnt = self._nShift
