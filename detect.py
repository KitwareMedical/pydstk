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


"""Template detector.
"""

import sys
import time
import json
import numpy as np

import dsutil.dsutil as dsutil
import dsutil.dsinfo as dsinfo

from dsutil.dsutil import Timer
from dscore.system import LinearDS
from dscore.system import NonLinearDS
from dscore.system import OnlineLinearDS


def loadDB(f):
    """Load database of templates.
    """
    
    models, labels = [], []
    dbinfo = json.load(open(sys.argv[1]))
    for entry in dbinfo["data"]:
        models.append(pickle.load(open(entry["model"])))
        labels.append(entry["label"])
        

if __name__ == '__main__':
    
    (vid, size) = dsutil.loadDataFromVideoFile(sys.argv[1])
    #dsutil.showMovie(vid, size, fps=20)
    print vid.shape
    
    #models, labels = loadDB(sys.argv[1])
    
    # video loading
    #Y, _ = dsutil.loadDataFromASCIIFile(sys.argv[1])
    #T = Y.shape[1]

    # create an online LDS
    #odt = OnlineLinearDS(nStates=3, bufLen=10, nShift=1, approx=False, verbose=True)
    
    #for f in range(Y.shape[1]):
    #    odt.update(Y[:,f])
    #    
    #    d = [] 
    #    for tpl in models:
    #        d.append(dsdist.ldsMartinDistance(tpl, win, 20))
        
    #    p = np.argmin(np.asarray(d))
    #    print labels[p]