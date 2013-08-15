pydstk
======

[![Build Status](https://travis-ci.org/TubeTK/pydstk.png?branch=master)](https://travis-ci.org/TubeTK/pydstk)

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
5. [**nose**](https://nose.readthedocs.org/en/latest/)
6. [**termcolor**](https://pypi.python.org/pypi/termcolor)
7. [**scikit-learn**](http://scikit-learn.org/stable/)

To check if those packages are available on your system, try
```python
import cv2
import scipy
import numpy
import sklearn
import termcolor
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
The package `dsutil` contains a set of I/O routines to load data from harddisk. Three
common ways of loading video data are: 
- load an actual video file (via `loadDataFromVideoFIle`)
- load a video represented as a collection of frames (via `loadDataFromIListFile`)
- load a video as a large data matrix (via `loadDataFromASCIIFile`)

Type
```python
import dsutil.dsutil as dsutil
help(dsutil.loadDataFromASCIIFile)
```
in a Python console to get more information about the format of the input file(s) and 
the function parameters (here for function `loadDataFromASCIIFile`).

Running the unit-tests
----------------------
Unit-testing in pydstk is done using `nose`. All tests reside in the `tests` directory. To run, for instance, 
the tests for `systems.py` module, use:
```bash
$ nosetests tests/test_system.py -v
```

Where can I get data material ?
-------------------------------
Several resources for getting dynamic texture data can be found on the internet. An extensive database of dynamic texture is available in the [**Dyntex**](http://projects.cwi.nl/dyntex/) 
created by *R. Peteri et al.* Another interesting set of videos (e.g., for recognition experiments) is the [**Traffic**](http://www.svcl.ucsd.edu/projects/traffic/) database created
by *A. Chan and N. Vasconcelos* that was used in

```bibtex
@inproceedings{Chan05a,
  author = {A.~B.~Chan and N.~Vasconcelos},
  title = {Probabilistic Kernels for the Classification of Auto-regressive Visual Processes},
  booktitle = {CVPR},
  year = {2005}}
```

Another dataset, from the field of medical Ultrasound imaging is available from [**MIDAS**](http://midas3.kitware.com/midas/folder/10255).
This dataset contains a collection of Ultrasound videos acquired on a (hand-made) phantom. The videos (in AVI format) are split into *key* videos and *search*
videos and can be used to experiment with approaches that try to recognize the *key* videos in the *search* videos for instance. 
The `scripts` directory of `pydstk` contains a `download.py` script that can automatically download this database. You only need 
to adjust the file `scripts/pydas.config.example` to your MIDAS account settings. This database was used in

```bibtex
@article{Kwitt13b,
  author = {R. Kwitt and N. Vasconcelos and S. Razzaque and S. Aylward},
  title = {Localizing Target Structures in Ultrasound Video - A Phantom Study},
  journal = {Medical Image Analysis},
  volume ={17},
  number = {7},
  pages = {712-722},
  year = 2013}
```
and
```bibtex
@inproceedings{Kwitt12d,
  author = {R. Kwitt and N. Vasconcelos and S. Razzaque and S. Alyward},
  title = {Recognition in Ultrasound Videos: Where Am I?},
  booktitle = {MICCAI},
  year = 2012}
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
**Similarity measurement between two DT models**
- Source model: `/tmp/us-dt-model.pkl`
- Reference model: `/tmp/us-dt-model.pkl`
- Nr. of summation terms (for Lyapunov eq.): 50

```bash
python dtdist.py -s /tmp/us-dt-model.pkl /tmp/us-dt-model.pkl -n 50
```
**Similarity measurement between two KDT models**
- Source model: `/tmp/us-kdt-model.pkl`
- Reference model: `/tmp/us-kdt-model.pkl`
- Nr. of summation terms (for Lyapunov eq.): 50

```bash
python kdtdist.py -s /tmp/us-kdt-model.pkl -r /tmp/us-kdt-model.pkl -n 50
````
**Online template detection in videos**
- coming soon!

---
```
Author: Roland Kwitt
E-Mail: roland.kwitt@kitware.com
Personal website: http://rkwitt.org
```
