pydstk
======

Python (Linear / Non-Linear) Dynamical Systems Toolkit (pydstk). This package implements two dynamical system variants that are commonly used in computer vision: **Dynamic Textures** (i.e., linear DS) and **Kernel Dynamic Textures** (i.e., non-linear DS). In addition, several approaches to measure the similarity
between dynamical systems are implemented. 

Requirements
------------

1. [**numpy**](http://www.numpy.org)
2. [**scipy**](http://www.scipy.org)
3. [**OpenCV**](http://opencv.willowgarage.com/wiki/) (Python wrapping, tested with v2.4.5)
4. [**SimpleITK**](http://www.simpleitk.org) (Python wrapping)

To test if those packages are available on your system, try
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

---
```
Author: Roland Kwitt
E-Mail: roland.kwitt@kitware.com
```
