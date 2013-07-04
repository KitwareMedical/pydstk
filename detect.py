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
import pickle
import numpy as np
from optparse import OptionParser

import dsutil.dsutil as dsutil
import dsutil.dsinfo as dsinfo
import dscore.dsdist as dsdist

from dsutil.dsutil import Timer
from dscore.system import LinearDS
from dscore.system import NonLinearDS
from dscore.system import OnlineLinearDS
from dscore.system import OnlineNonLinearDS
from dscore.dskpca import kpca, KPCAParam, rbfK, RBFParam


def usage():
    """Print usage information"""
    print("""
Continuous (temporal) template recognition in videos.

USAGE:
    {0} [OPTIONS]
    {0} -h

OPTIONS (Overview):

    -s ARG -- Source video file (AVI)
    -d ARG -- JSON database file
    -t ARG -- DS type ('dt', 'kdt') 
    [-n ARG] -- #DS states (int, default=5)
    [-w ARG] -- Sliding window size (int, default=20)
    [-s ARG] -- Sliding window shift (int, default=2)
    [-i ARG] -- Lyapunov solver iterations (int, default=20)
    [-v] -- Verbose output

NOTE: See 'db.example.json' for an example database file!
        
AUTHOR: Roland Kwitt, Kitware Inc., 2013
        roland.kwitt@kitware.com
""".format(sys.argv[0]))
    sys.exit(-1)

       
def main(argv=None):
    if argv is None: 
        argv = sys.argv

    parser = OptionParser(add_help_option=False)

    # input video, database, dynamical system type
    parser.add_option("-s", dest="inFile")
    parser.add_option("-d", dest="dbFile") 
    parser.add_option("-t", dest="dsType")
    
    # DS states, sliding window size and sliding window shift
    parser.add_option("-n", dest="nStates", type="int", default=5)
    parser.add_option("-w", dest="winSize", type="int", default=40)
    parser.add_option("-m", dest="shiftMe", type="int", default=2)

    # iterations for solving Lyapunov equation
    parser.add_option("-i", dest="numIter", type="int", default=20)

    parser.add_option("-h", dest="doUsage", action="store_true", default=False)
    parser.add_option("-v", dest="verbose", action="store_true", default=False) 
    options, args = parser.parse_args()
    
    if options.doUsage: 
        usage()
    
    nStates = options.nStates
    winSize = options.winSize
    shiftMe = options.shiftMe
    nStates = options.nStates
    verbose = options.verbose
    numIter = options.numIter
    
    #TODO: sanity checks
    
    (vid, size) = dsutil.loadDataFromVideoFile(options.inFile)
    dbinfo = json.load(open(options.dbFile))
    
    db = []
    for entry in dbinfo['data']:
        db.append({ 'model' : pickle.load(open(entry['model'])), 
                    'label' : int(entry["label"]) })
    
    if options.dsType == 'dt':
        ds = OnlineLinearDS(nStates, winSize, shiftMe, False, verbose)
    elif options.dsType == 'kdt':
        kpcaP = KPCAParam()
        kpcaP._kPar = RBFParam()
        kpcaP._kPar._kCen = True
        kpcaP._kFun = rbfK
        ds = OnlineNonLinearDS(nStates, kpcaP, winSize, shiftMe, verbose)
    else:
        dsinfo.fail('DS type %s not supported!' % options.dsType)        
        return -1

    dList = []
    for f in range(vid.shape[1]):
        ds.update(vid[:,f])
        if ds.check():
            dists = np.zeros((len(db), 1))
            for j, dbentry in enumerate(db):
                dists[j] = { 'dt' : dsdist.ldsMartinDistance,
                             'kdt': dsdist.nldsMartinDistance
                }[options.dsType](ds, dbentry['model'], numIter)
                dList.append(dists)
                               
    
if __name__ == '__main__':
    sys.exit(main())    