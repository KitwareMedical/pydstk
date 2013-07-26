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
import cv2
import sys
import time
import copy
import pickle
import numpy as np
import cv2.cv as cv
from optparse import OptionParser

import dscore.dsdist as dsdist
import dsutil.dsutil as dsutil
import dsutil.dsinfo as dsinfo

from dscore.system import LinearDS


"""Application #3: Dynamic Texture Clustering
"""


__license__ = "Apache License, Version 2.0"
__author__  = "Roland Kwitt, Kitware Inc., 2013"
__email__   = "E-Mail: roland.kwitt@kitware.com"
__status__  = "Development"


def usage():
    """Print usage information"""
    print("""
Dynamic Texture Clusering (using MDS).

USAGE:
    {0} [OPTIONS]
    {0} -h

OPTIONS (Overview):

    -i ARG -- Input file (list of pickled DT's)
    -k ARG -- Number of K cluster centers
    -b ARG -- Base directory of input DT's
    -o ARG -- Output list of K represenatives

    [-v] -- Verbose output (default: False)
    
AUTHOR: Roland Kwitt, Kitware Inc., 2013
        roland.kwitt@kitware.com
""".format(sys.argv[0]))
    sys.exit(-1)


def main(argv=None):
    if argv is None: 
        argv = sys.argv

    parser = OptionParser(add_help_option=False)
    parser.add_option("-i", dest="iListFile") 
    parser.add_option("-o", dest="oListFile")
    parser.add_option("-b", dest="iBase")
    parser.add_option("-k", dest="kCenter", type="int", default=5)
    parser.add_option("-h", dest="shoHelp", action="store_true", default=False)
    parser.add_option("-v", dest="verbose", action="store_true", default=False) 
    options, args = parser.parse_args()
    
    if options.shoHelp: 
        usage()
    
    iBase = options.iBase
    iListFile = options.iListFile
    oListFile = options.oListFile
    kCenter = options.kCenter
    verbose = options.verbose

    assert kCenter > 0, "Oops (kCenter < 1) ..."
 
    iList = pickle.load(open(iListFile))
 
    if verbose:
        dsinfo.info("Loaded list with %d DT models!" % len(iList))
 
    # load DT's
    dts = []
    for dtFile in iList:
        dts.append(pickle.load(open(os.path.join(iBase, dtFile))))
    
    # run clustering
    if verbose:
        dsinfo.info("Running DT clustering with %d clusters ..." % kCenter)
    
    ids = LinearDS.cluster(dts, kCenter, verbose)
    ids = list(ids)
    
    #write list of DT representatives
    oList = []
    for j, dtFile in enumerate(iList):
            if j in ids:
                oList.append(dtFile)
            else:
                oList.append("Dummy")
    pickle.dump(oList, open(oListFile, "w"))

    if verbose:
        dsinfo.info("Wrote list of representative DS's to %s!" % oListFile)


if __name__ == "__main__":
    sys.exit(main())




