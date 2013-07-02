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

"""pydstk's CLI messaging (with color).
"""


__license__ = "Apache License, Version 2.0"
__author__  = "Roland Kwitt, Kitware Inc., 2013"
__email__   = "E-Mail: roland.kwitt@kitware.com"
__status__  = "Development"


from termcolor import colored


def time(msg):
    """For timing results.
    """
    print colored("[time]: %s" % msg, 'blue')


def info(msg):
    """For simple info messages.
    """
    print colored("[info]: %s" % msg, 'green')


def warn(msg):
    """For warning messages.
    """
    print colored("[warn]: %s" % msg, 'magenta')

    
def fail(msg):
    """For failure messages.
    """
    print colored("[fail]: %s" % msg, 'red')