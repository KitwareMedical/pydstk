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


""" Download REUS database from MIDAS.
"""


import pydas
import os
import sys
import json


if __name__ == "__main__":
    
    configFile = sys.argv[1]
    downloadTo = sys.argv[2]
    findFolder = sys.argv[3]

    fid = open(configFile).read()
    config = json.loads(fid)

    id = '10255'
    pydas.login(email=config['email'], 
                api_key=config['api_key'], 
                url=config['url'])

    if not os.path.exists(downloadTo):
        os.makedirs(downloadTo)

    _, id = pydas.api._search_folder_for_item_or_folder(findFolder, id)
        
    if id > 0:
        dataDir = os.path.join(downloadTo, findFolder)
        pydas.api._download_folder_recursive(id, downloadTo)