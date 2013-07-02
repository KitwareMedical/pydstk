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


"""Testing for dscore/system.py
"""


import os
import sys
import json
import pickle
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dscore.system import LinearDS
from dsutil.dsutil import loadDataFromASCIIFile
from dscore.system import NonLinearDS
from dscore.dskpca import KPCAParam, rbfK, RBFParam


TESTBASE = os.path.dirname(__file__) 


def test_LinearDS_check():
    lds = LinearDS(5, False, False)
    assert lds.check() is False


def test_NonLinearDS_check():
    kpcaP = KPCAParam()
    kpcaP._kPar = RBFParam()
    kpcaP._kPar._kCen = True
    kpcaP._kFun = rbfK
    
    nlds = NonLinearDS(5, kpcaP, False)
    assert nlds.check() is False


def test_LinearDS_suboptimalSysID(): 
    dataFile = os.path.join(TESTBASE, "data/data1.txt")
    data, _ = loadDataFromASCIIFile(dataFile)
     
    lds = LinearDS(5, False, False)
    lds.suboptimalSysID(data)
     
    baseLDSFile = os.path.join(TESTBASE, "data/data1-dt-5c-center.pkl")
    baseLDS = pickle.load(open(baseLDSFile))
     
    _, err = LinearDS.stateSpaceMap(baseLDS, lds)
    assert np.allclose(err, 0.0) == True


def test_NonLinearDS_suboptimalSysID(): 
    dataFile = os.path.join(TESTBASE, "data/data1.txt")
    data, _ = loadDataFromASCIIFile(dataFile)
    
    kpcaP = KPCAParam()
    kpcaP._kPar = RBFParam()
    kpcaP._kPar._kCen = True
    kpcaP._kFun = rbfK
         
    nlds = NonLinearDS(5, kpcaP, False)
    nlds.suboptimalSysID(data)

    baseNLDSFile = os.path.join(TESTBASE, "data/data1-rbf-kdt-5c-center.pkl")
    baseNLDS = pickle.load(open(baseNLDSFile))
    
    err = NonLinearDS.naiveCompare(baseNLDS, nlds)
    assert np.allclose(err, 0.0) == True