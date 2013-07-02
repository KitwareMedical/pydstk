pydstk
======

Python (Linear / Non-Linear) Dynamical Systems Toolkit (pydstk). 
This package implements two dynamical system variants that are commonly used in computer vision: 
**Dynamic Textures** (i.e., linear DS) and **Kernel Dynamic Textures** (i.e., non-linear DS). In addition, 
several approaches to measure the similarity between dynamical systems are implemented. 

Requirements
------------

1. [**numpy**](http://www.numpy.org)
2. [**scipy**](http://www.scipy.org)
3. [**OpenCV**](http://opencv.willowgarage.com/wiki/) (Python wrapping, tested with v2.4.5)
4. [**SimpleITK**](http://www.simpleitk.org) (Python wrapping)

To check if those packages are available on your system, try
```python
import cv2
import scipy
import numpy
import SimpleITK
```
in a Python console. If no error occurs, you are all set! 

References
----------

For the seminal work on **Dynamic Textures**, see:

```bibtex
@article{Doretto01a,
  author = {G.~Doretto and A.~Chiuso and Y.~N.~Wu and S.~Soatto},
  title = {Dynamic Textures},
  journal = {Int. J. Comput. Vision},
  year = 2001,
  pages = {91--109},
  volume = 51,
  number = 2} 
```

Similarity measurement between two linear dynamical systems by means of subspace-angles is discussed in: 

```bibtex
@inproceedings{DeCock00a,
  author = {K.~{De Cock} and B.~D.~Moore},
  title = {Subspace angles between linear stochastic models},
  booktitle = {CDC},
  pages = {1561-1566},
  year = 2000}
```

**Kernel Dynamic Textures** (as well as the non-linear extension of the subspace-angle based similarity measure) were introduced in:

```bibtex
@inproceedings{Chan07a,
  author = {A.~B.~Chan and N.~Vasconcelos},
  title = {Classifying Video with Kernel Dynamic Textures},
  booktitle = {CVPR},
  pages = {1-6},
  year =  2007}
```


Supported I/O File Formats
--------------------------
tbd.


Example Applications
--------------------

pydstk implements a set of example applications that can be useful for starting to work
with dynamical system models. These example applications include command-line tools to 
estimate dynamic texture and non-linear dynamic texture models from videos represented 
as AVI files, ASCII files or a set of frames. These applications are named `dt.py` and 
`kdt.py`. Calling 

```
python dt.py -h
python kdt.py -h
```

shows some help on how to use these applications. Further, pydstk provides two supplementary
applications `dtdist.py` and `kdtdist.py` that enable to compute a similarity measure between
two DT/KDT models. This can be usefull for recognition experiments for instance. The 
applications are also a good starting point to get familiar with the API. 

To get started, pydstk contains some example videos under `tests/data/`, such as `ultrasound.avi`
or `data1.txt`. These videos are rather small and are also used during unit testing. In case
of `data1.txt`, it is worth taking a closer look at the header of this ASCII file, since it 
identifies the video dimensions (here 48x48x48 pixel).

We will use the file `ultasound.avi` in this example. Is represents an Ultrasound (US) video 
(64x64 pixel) x 40 frames that was acquired on an Ultrasound phantom. We first estimate a DT
model with 5 states as follows:

```bash
python dt.py -i tests/data/ultrasound.avi -n 5 -t vFile -e -o /tmp/us-dt-model.pkl
```

This write a file `/tmp/us-dt-model.pkl` using Python's pickle functionality. In case we want to 
synthesize the video from the model and show the synthesis result, we can use

```bash
python dt.py -i tests/data/ultrasound.avi -n 5 -t vFile -e -s -o /tmp/us-dt-model.pkl -m 20
```

which shows the synthesis result (at 20 FPS). Similar to that, we can estimate a 
KDT model with 5 states using a RBF kernel as follows:

```
python kdt.py -i tests/data/ultrasound.avi -n 5 -t vFile -o /tmp/us-kdt-model.pkl
```

**Note**: We do not support synthesis for KDT models at that point!. After we 
have estimated either DT or KDT models for our videos, we can measure similarity
between these models using the example applications `dtdist.py` and `kdtdist.py`
which rely on the **Martin distance**. For simplicity, we will measure the Martin
distance between a model and itself (which should be 0 of course). In the case 
of DT's, this can be done as follows:

```bash
python dtdist.py -s /tmp/us-dt-model.pkl /tmp/us-dt-model.pkl -n 50
```

Similar to that, we use

```bash
python kdtdist.py -s /tmp/us-kdt-model.pkl -r /tmp/us-kdt-model.pkl -n 50
````

in case of the KDT models. In both cases `-s` specifies the *source* model
and `-r` specifies the reference model. Note that the Martin distance is not
symmetric! The parameter `-n ARG` sets the number of summation terms that we
use to solve the discrete Lyapunov equation that appears in the formulation
of the Martin distance (see references).





















---
```
Author: Roland Kwitt
E-Mail: roland.kwitt@kitware.com
```
