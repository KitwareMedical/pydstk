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


"""Application #1: Dynamic Textures
"""


__license__ = "Apache License, Version 2.0"
__author__  = "Roland Kwitt, Kitware Inc., 2013"
__email__   = "E-Mail: roland.kwitt@kitware.com"
__status__  = "Development"


import os
import sys
import pickle
from optparse import OptionParser

# import classes from modules in dsutil/dscore package
from dsutil.dsutil import Timer
from dscore.system import LinearDS
from dscore.dsexcp import ErrorDS

# import modules from dsutil package
import dsutil.dsutil as dsutil
import dsutil.dsinfo as dsinfo


def usage():
    """Print usage information"""
    print("""
Dynamic Texture estimation / synthesis using Linear Dynamical Systems (LDS).

USAGE:
    {0} [OPTIONS]
    {0} -h

OPTIONS (Overview):

    -i ARG -- Input file
    -t ARG -- Input file type
    
        'vFile' - AVI video file
        'aFile' - ASCII data file
        'lFile' - Image list file 
        
    [-n ARG] -- LDS states (default: 5)
    [-o ARG] -- Save DT parameters -> ARG 
    [-p ARG] -- Load DT parameters <- ARG
    [-m ARG] -- FPS for synthesis movie (default: 20)
    [-e] -- Run estimation (default: False)
    [-s] -- Run synthesis  (default: False)
    [-v] -- Verbose output (default: False)
    [-a] -- Use randomized SVD for estimation
    
AUTHOR: Roland Kwitt, Kitware Inc., 2013
        roland.kwitt@kitware.com
""".format(sys.argv[0]))
    sys.exit(-1)


def main(argv=None):
    if argv is None: 
        argv = sys.argv

    parser = OptionParser(add_help_option=False)
    parser.add_option("-p", dest="pFile")
    parser.add_option("-i", dest="iFile") 
    parser.add_option("-t", dest="iType")
    parser.add_option("-o", dest="oFile")
    parser.add_option("-n", dest="nStates", type="int", default=+5)
    parser.add_option("-m", dest="doMovie", type="int", default=-1)
    parser.add_option("-a", dest="svdRand", action="store_true", default=False)
    parser.add_option("-e", dest="doEstim", action="store_true", default=False)
    parser.add_option("-s", dest="doSynth", action="store_true", default=False)
    parser.add_option("-h", dest="shoHelp", action="store_true", default=False)
    parser.add_option("-v", dest="verbose", action="store_true", default=False) 
    opt, args = parser.parse_args()
    
    if opt.shoHelp: 
        usage()
    
    dataMat = None
    dataSiz = None
    try:
        if opt.iType == 'vFile':
            (dataMat, dataSiz) = dsutil.loadDataFromVideoFile(opt.iFile)
        elif opt.iType == 'aFile':
            (dataMat, dataSiz) = dsutil.loadDataFromASCIIFile(opt.iFile)
        elif opt.iType == 'lFile':
            (dataMat, dataSiz) = dsutil.loadDataFromIListFile(opt.iFile)
        else:
            msg.fail("Unsupported file type : %s", opt.iType)    
            return -1
    except Exception as e:
        msg.fail(e)
        return -1

    try:
        # try loading the DT model
        if not opt.pFile is None:
            with open(opt.pFile) as fid:
                dsinfo.info('trying to load model %s' % opt.pFile)
                dt = pickle.load(fid)

        # run estimation
        if opt.doEstim:
            if not opt.pFile is None:
                dsinfo.fail('re-estimation attempt detected!')
                return -1
            dt = LinearDS(opt.nStates, approx=opt.svdRand, verbose=opt.verbose)
            dt.suboptimalSysID(dataMat)

        # synthesize output
        if opt.doSynth:
           dataSyn, _ = dt.synthesize(tau=50, mode='s')

        # show a movie of the synthesis result
        if opt.doMovie > 0:
            if opt.doSynth:
                dsutil.showMovie(dataSyn, dataSiz, fps=opt.doMovie)

        # write DT model to file
        if not opt.oFile is None:
            dsinfo.info('writing model to %s' % opt.oFile)
            with open(opt.oFile, 'w') as fid:
                pickle.dump(dt, fid)
     
    # catch pyds exceptions
    except ErrorDS as e:
        msg.fail(e)
        return -1
            
if __name__ == '__main__':
    sys.exit(main())
