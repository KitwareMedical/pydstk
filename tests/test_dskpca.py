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


"""Testing for dscore/dskpca.py
"""


import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dscore.dskpca import KPCAParam, rbfK, RBFParam, kpca
from dsutil.dsutil import loadDataFromASCIIFile


TESTBASE = os.path.dirname(__file__)


def test_rbfK_centered():
    par = RBFParam()
    par._kCen = True

    dataFile = os.path.join(TESTBASE, "data/random.txt")
    data = np.genfromtxt(dataFile, dtype=np.double)

    rbfK(data, data, par)

    kernelFile = os.path.join(TESTBASE, "data/random-rbf-center.txt")
    kernel = np.genfromtxt(kernelFile, dtype=np.double)

    err = np.linalg.norm(kernel - par._kMat, 'fro')
    np.testing.assert_almost_equal(err, 0, 2)


def test_rbfK_noncentered():
    par = RBFParam()
    par._kCen = False

    dataFile = os.path.join(TESTBASE, "data/random.txt")
    data = np.genfromtxt(dataFile, dtype=np.double)

    rbfK(data, data, par)

    kernelFile = os.path.join(TESTBASE, "data/random-rbf-nocenter.txt")
    kernel = np.genfromtxt(kernelFile, dtype=np.double)

    err = np.linalg.norm(kernel - par._kMat, 'fro')
    np.testing.assert_almost_equal(err, 0, 2)


def test_kpca():
    dataFile = os.path.join(TESTBASE, "data/data1.txt")
    data, _ = loadDataFromASCIIFile(dataFile)

    kpcaP = KPCAParam()
    kpcaP._kPar = RBFParam()
    kpcaP._kPar._kCen = True
    kpcaP._kFun = rbfK

    X = kpca(data, 5, kpcaP)

    baseKPCACoeffFile = os.path.join(TESTBASE, "data/data1-rbf-kpca-5c-center.txt")
    baseKPCACoeff = np.genfromtxt(baseKPCACoeffFile, dtype=np.double)

    # don't care about the sign
    err = np.linalg.norm(np.abs(baseKPCACoeff)-np.abs(X), 'fro')
    np.testing.assert_almost_equal(err, 0, 2)


if __name__ == "__main__":
    test_kpca()
