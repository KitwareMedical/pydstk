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


import os
import sys
import time
import json
import glob
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

    -s ARG -- Source video file (only AVI videos are supported!)
    -d ARG -- Database file in JSON format
    -c ARG -- Config file in JSON format
    -v ARG -- Base directory of template videos
    -m ARG -- Base directory of template models
    [-o ARG] -- Write distance matrix to file
    [-x] -- Verbose output

AUTHOR: Roland Kwitt, Kitware Inc., 2013
        roland.kwitt@kitware.com
""".format(sys.argv[0]))
    sys.exit(-1)


def loadDB(videoDir, modelDir, dbFile):
    """Load database information.
    
    Parameters
    ----------
    videoDir : string
        Base directory where the videos reside.
    modelDir: string
        Base directory where the models reside.
    dbFile: string
        Filename of the database file in JOSN format.

    Returns
    -------
    db : list
        List of N dictionaries with keys:
            "video" -- Name of DS file
            "label" -- Name of corresponding AVI video file
            "model" -- Loaded DS model
    winSize : int 
        Length (#frames) of the templates
    nStates : int
        Number of DS models for templates
    dynType : string
        DS type (result of type(...))
    """
    
    dbinfo = json.load(open(dbFile))
    
    db, labels = [], []
    winSize = set() 
    nStates = set()
    dynType = set()
    
    for entry in dbinfo:
        tplEntry = entry["ks"] # Key sequence name
        labEntry = entry["cl"] # Class label of key sequence
        
        res = glob.glob(os.path.join(videoDir, '%s*.avi' % tplEntry))
        for entry in res:
            videoFile = os.path.basename(entry)
            modelFile = os.path.splitext(videoFile)[0]+".pkl"
            modelFile = os.path.join(modelDir, modelFile)
                                     
            if not os.path.exists(modelFile):                         
                dsinfo.fail("%s does not exist!" % modelFile)
                raise Exception()
                                     
            db.append({ "model" : pickle.load(open(modelFile)),
                        "video" : videoFile,
                        "label" : labEntry })                     
            winSize.add(db[-1]["model"]._Xhat.shape[1])
            nStates.add(db[-1]["model"]._nStates)
            dynType.add(type(db[-1]["model"]))
      
    if not (len(winSize) == 1 and len(nStates) == 1 and len(dynType) == 1):
        dsinfo.fail("Incompatible template configuration!")
        raise Exception()
    
    return (db, 
            iter(winSize).next(), # common sliding window size
            iter(nStates).next(), # common number of DS states
            iter(dynType).next()) # common DS type
   
          
def main(argv=None):
    if argv is None: 
        argv = sys.argv

    parser = OptionParser(add_help_option=False)

    parser.add_option("-s", dest="inFile")
    parser.add_option("-d", dest="dbFile") 
    parser.add_option("-m", dest="models")
    parser.add_option("-v", dest="videos")
    parser.add_option("-c", dest="config")
    parser.add_option("-o", dest="mdFile")

    parser.add_option("-h", dest="doUsage", action="store_true", default=False)
    parser.add_option("-x", dest="verbose", action="store_true", default=False) 
    options, args = parser.parse_args()
    
    if options.doUsage: 
        usage()

    # read config file
    config = json.load(open(options.config))
    
    # get DS config settings
    dynType = config["dynType"]
    shiftMe = config["shiftMe"]
    numIter = config["numIter"]

    verbose = options.verbose
    
    # I/O configuration
    inFile = options.inFile
    dbFile = options.dbFile
    models = options.models
    videos = options.videos
    mdFile = options.mdFile
    
    # check if the required options are present
    if (inFile is None or dbFile is None or
        models is None or videos is None):
        dsinfo.warn('Options missing!')
        usage()
    
    inVideo, inVideoSize = dsutil.loadDataFromVideoFile(inFile)
    if verbose:
        dsinfo.info("Loaded source video with %d frames!" % inVideo.shape[1])
    
    (db, winSize, nStates, dynType) = loadDB(videos, models, dbFile)
    
    if verbose:
        dsinfo.info("#Templates: %d #States: %d, WinSize: %d, Shift: %d" % 
                     (len(db), nStates, winSize, shiftMe))
    
    if dynType.__name__ == "LinearDS":
        # create online version of LinearDS
        ds = OnlineLinearDS(nStates, winSize, shiftMe, False, verbose)
    elif dynType.__name__ == "NonLinearDS":
        kpcaP = KPCAParam()
       
        # select kernel
        if config["kdtKern"] == "rbf":
            kpcaP._kPar = RBFParam()
            kpcaP._kFun = rbfK
        else:
            dsinfo.fail("Kernel %s not supported!" % kdtKern)
            return -1
        
        # configure kernel
        if config["kCenter"] == 1:
            kpcaP._kPar._kCen = True
        else:
            kpcaP._kPar._kCen = False
            
        # create online version of KDT
        ds = OnlineNonLinearDS(nStates, kpcaP, winSize, shiftMe, verbose)
    else:
        dsinfo.fail('System type %s not supported!' % options.dsType)        
        return -1

    dList = []
    for f in range(inVideo.shape[1]):
        ds.update(inVideo[:,f])
        if ds.check() and ds.hasChanged():
            dists = np.zeros((len(db),))
            for j, dbentry in enumerate(db):
                dists[j] = { "LinearDS" : dsdist.ldsMartinDistance,
                             "NonLinearDS": dsdist.nldsMartinDistance
                }[dynType.__name__](ds, dbentry["model"], numIter)
            dList.append(dists)
    
    # write distance matrix
    if not mdFile is None:
        np.savetxt(mdFile, np.asmatrix(dList), fmt='%.5f', delimiter=' ')
    
    
if __name__ == '__main__':
    sys.exit(main())    