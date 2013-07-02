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

Supported I/O file formats
--------------------------
The package `dsutil` contains a set of I/O routines to load data from harddisk (tbd.)


Running unit-tests
------------------
Unit-testing in pydstk is done using `nose`. All tests reside in the `tests` directory. To run, for instance, 
the tests for `systems.py` module, use:
```bash
$ nosetests tests/test_system.py -v
```

Some example applications
-------------------------

**Estimating a dynamic texture model (DT)**
- DT states: 5
- Input data: video file `tests/ultrasound.avi`
- Output data: DT model file `/tmp/us-dt-model.pkl`

```bash
$ python dt.py -i tests/data/ultrasound.avi \ 
               -n 5 \
               -t vFile \
               -e \ 
               -o /tmp/us-dt-model.pkl
```
**Estimate and synthesize a video from a DT model**
- DT states: 5
- Input data: video file `tests/data/ultrasound.avi`
- Output data: `/tmp/us-dt-model.pkl`
- Frame rate: 20 FPS

```bash
$ python dt.py -i tests/data/ultrasound.avi \
               -n 5 \
               -t vFile \
               -e \
               -s \
               -o /tmp/us-dt-model.pkl \
               -m 20
```
**Estimating a kernel dynamic texture model (KDT)**
- KDT states: 5
- Input data: video file `tests/data/ultrasound.avi`
- Output data: `/tmp/us-kdt-model.pkl`
- Kernel: RBF (default)

```bash
$ python kdt.py -i tests/data/ultrasound.avi -n 5 -t vFile -o /tmp/us-kdt-model.pkl
```
**Similarity Measurement between two DT models**
- Source model: `/tmp/us-dt-model.pkl`
- Reference model: `/tmp/us-dt-model.pkl`
- Nr. of summation terms (for Lyapunov eq.): 50

```bash
python dtdist.py -s /tmp/us-dt-model.pkl /tmp/us-dt-model.pkl -n 50
```
**Similarity Measurement between two KDT models**
- Source model: `/tmp/us-kdt-model.pkl`
- Reference model: `/tmp/us-kdt-model.pkl`
- Nr. of summation terms (for Lyapunov eq.): 50

```bash
python kdtdist.py -s /tmp/us-kdt-model.pkl -r /tmp/us-kdt-model.pkl -n 50
````

---
```
Author: Roland Kwitt
E-Mail: roland.kwitt@kitware.com
Personal website: http://rkwitt.org
```
