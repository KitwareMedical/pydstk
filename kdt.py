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


"""Application #2: Kernel Dynamic Textures (KDT)
"""


__license__ = "Apache License, Version 2.0"
__author__  = "Roland Kwitt, Kitware Inc., 2013"
__email__   = "E-Mail: roland.kwitt@kitware.com"
__status__  = "Development"


# generic imports
import os
import sys
import pickle
import numpy as np
from optparse import OptionParser

# import supp. pyds packages
import dsutil.dsutil as dsutil
import dsutil.dsinfo as dsinfo
import dscore.dsdist as dsdist

# import pyds classes
from dscore.dsexcp import ErrorDS
from dscore.system import NonLinearDS
from dscore.dskpca import KPCAParam, rbfK, RBFParam


def usage():
    """Print usage information"""
    print("""
Kernel Dynamic Texture estimation using Non-Linear Dynamical Systems (NLDS).

USAGE:
    {0} [OPTIONS]
    {0} -h

OPTIONS (Overview):

    -i ARG -- Input file
    -t ARG -- Input file type
    
        'vFile' - AVI video file
        'aFile' - ASCII data file
        'lFile' - Image list file 
        
    [-n ARG] -- NLDS states (default: 5)
    [-o ARG] -- Save KDT parameters to ARG 
    [-v] -- Verbose output (default: False)
        
AUTHOR: Roland Kwitt, Kitware Inc., 2013
        roland.kwitt@kitware.com
""".format(sys.argv[0]))
    sys.exit(-1)
    
    
def main(argv=None):
    if argv is None: 
        argv = sys.argv

    parser = OptionParser(add_help_option=False)
    parser.add_option("-i", dest="iFile") 
    parser.add_option("-t", dest="iType")
    parser.add_option("-o", dest="oFile")
    parser.add_option("-n", dest="nStates", type="int", default=5)
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
            dsinfo.fail("Unsupported file type : %s" % opt.iType)    
            return -1
    
    # catch pyds exceptions
    except ErrorDS as e:
        msg.fail(e)
        return -1
    
    try:
        
        kpcaP = KPCAParam()
        kpcaP._kPar = RBFParam()
        kpcaP._kPar._kCen = True
        kpcaP._kFun = rbfK
        
        kdt = NonLinearDS(opt.nStates, kpcaP, opt.verbose)
        kdt.suboptimalSysID(dataMat)
       
        if not opt.oFile is None:
            if not kdt.check():
                dsinfo.fail('cannot write invalid model!')
                return -1
            dsinfo.info('writing model to %s' % opt.oFile)
            with open(opt.oFile, 'w') as fid:
                pickle.dump(kdt, fid)

        print kdt._Vhat


    except ErrorDS as e:
        dsinfo.fail(e)
        return -1

    
if __name__ == '__main__':
    sys.exit(main())
    
    
    



