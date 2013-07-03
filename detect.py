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

import dsutil.dsutil as dsutil
import dsutil.dsinfo as dsinfo
import dscore.dsdist as dsdist

from dsutil.dsutil import Timer
from dscore.system import LinearDS
from dscore.system import NonLinearDS
from dscore.system import OnlineLinearDS


def loadDB(f):
    """Load database of templates.
    """
    return json.load(open(f))        
    
    
if __name__ == '__main__':
    
    (vid, size) = dsutil.loadDataFromVideoFile(sys.argv[1])
    dbinfo = loadDB(sys.argv[2])
    
    db = []
    for entry in dbinfo['data']:
        db.append({ 'model' : pickle.load(open(entry['model'])), 
                    'label' : int(entry["label"]) })
    
    dsinfo.info("loaded %d template model(s)!" % len(db))

    # create online DS with 5 states, no-skipping
    odt = OnlineLinearDS(5, 40, 2, False, False)
    
    t0 = time.clock()

    for f in range(vid.shape[1]):
        odt.update(vid[:,f])
        if odt.check():
            for dbentry in db:
                md = dsdist.ldsMartinDistance(odt, dbentry['model'], 20)            
    
    t1 = time.clock()
    print 'processing time : %.2g [sec]' % (t1-t0)