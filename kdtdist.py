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


"""Distance measurements between KDT's.
"""


__license__ = "Apache License, Version 2.0"
__author__  = "Roland Kwitt, Kitware Inc., 2013"
__email__   = "E-Mail: roland.kwitt@kitware.com"
__status__  = "Development"


# generic imports
import os
import cv2
import sys
import pickle
import numpy as np
import cv2.cv as cv
from optparse import OptionParser

# import pyds package content
import dscore.dsdist as dsdist
import dsutil.dsutil as dsutil
import dsutil.dsinfo as dsinfo

# import LinearDS class from dscore package
from dscore.system import NonLinearDS


def usage():
    """Print usage information"""
    print("""
Martin distance computation between Kernel Dynamic Texture (KDT) models.

USAGE:
    {0} [OPTIONS]
    {0} -h

OPTIONS (Overview):

    -s ARG -- KDT1 model file
    -r ARG -- KDT2 model file
        
AUTHOR: Roland Kwitt, Kitware Inc., 2013
        roland.kwitt@kitware.com
""".format(sys.argv[0]))
    sys.exit(-1)


def main(argv=None):
    if argv is None: 
        argv = sys.argv

    parser = OptionParser(add_help_option=False)
    parser.add_option("-s", dest="model1File")
    parser.add_option("-r", dest="model2File") 
    parser.add_option("-n", dest="iterations", type="int", default=20)
    parser.add_option("-h", dest="shoHelp", action="store_true", default=False)
    parser.add_option("-v", dest="verbose", action="store_true", default=False) 
    opt, args = parser.parse_args()
    
    if opt.shoHelp: 
        usage()
    
    with open(opt.model1File, 'r') as fid:
        kdt1 = pickle.load(fid)
    with open(opt.model2File, 'r') as fid:
        kdt2 = pickle.load(fid)

    martinD = dsdist.nldsMartinDistance(kdt1, kdt2, opt.iterations)
    dsinfo.info('D(%s,%s) = %.4f' % (opt.model1File, opt.model2File, martinD))
        
            
if __name__ == '__main__':
    sys.exit(main())