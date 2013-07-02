import time

import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy as np
import scikits.cuda.linalg as cla
import numpy.linalg as la
import scikits.cuda.cula as cula

cla.init()

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)


x = np.random.rand(128**2, 50).astype(np.float32)

with Timer('Push results'):
    gpux = gpuarray.to_gpu(x)

with Timer('GPU COV'):
    gcov = cla.dot(gpux, gpux, transa='C')
    ge_g, gh_g = np.linalg.eigh(gcov.get() / (x.shape[1] - 1))
    I = np.argsort(ge_g)[::-1]
    ge_g, gh_g = ge_g[I], gh_g[:,I]
    # push the matrix back out
    gpueigs = gpuarray.to_gpu(gh_g)
    W_g = cla.dot(gpux, gpueigs)
    #np.dot(x, gh_g)
    
with Timer('CPU COV'):
    hcov = np.dot(x.T, x)
    ge_c, gh_c = np.linalg.eigh(hcov / (x.shape[1] - 1))
    I = np.argsort(ge_c)[::-1]
    ge_c, gh_c = ge_c[I], gh_c[:,I]
    W_c = np.dot(x, gh_c)

with Timer('Fetch results'):
    W_g = W_g.get()

# Compute SVD
U, s, V = np.linalg.svd(x, full_matrices=False)

print np.max(np.abs(W_g - W_c))
nW = W_c / np.sqrt(np.sum(W_c**2, axis=0))[np.newaxis,:]

for i in range(5):
    print np.dot(nW[:,i], U[:,i])
