import time
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy as np
import scikits.cuda.linalg as cla
import numpy.linalg as la
import scikits.cuda.cula as cula

cla.init()

def testForSize(x):
    print 'Image Size %dx%d' % (x,x)

    x = np.random.rand(x**2, 40).astype(np.float32)

    def svdoverwrite(a_gpu, u_gpu, s_gpu, v_gpu, m, n, lda, ldu, ldvt):
        data_type = a_gpu.dtype.type
        real_type = np.float32
        cula_func = cula._libcula.culaDeviceSgesvd
        jobu = 'S'
        jobvt = 'S'

        status = cula_func(jobu, jobvt, m, n, int(a_gpu.gpudata),
                           lda, int(s_gpu.gpudata), int(u_gpu.gpudata),
                           ldu, int(v_gpu.gpudata), ldvt)

        cula.culaCheckStatus(status)

        # Free internal CULA memory:
        cula.culaFreeBuffers()

    t = time.time()
    gpux = gpuarray.to_gpu(x)
    pushtime = time.time() - t
    print '[Push results]',  pushtime

    t = time.time()
    u_g, s_g, v_g = cla.svd(gpux, 'S', 'S')   
    gpusvdtime = time.time() - t
    print '[GPU results]',  gpusvdtime

    t = time.time()
    u_g = u_g.get()
    s_g = s_g.get()
    v_g = v_g.get()
    fetchtime = time.time() - t
    print '[Fetch time]',  fetchtime

    t = time.time()
    u_c, s_c, v_c = la.svd(x, full_matrices=False)
    cputime = time.time() - t
    print '[CPU time]',  cputime

    print '[GPU time]', pushtime + gpusvdtime + fetchtime
    # Result on desktop Quadro FX1800
    print

if __name__ == '__main__':
    testForSize(128)
    testForSize(256)
    testForSize(512)
    testForSize(1024)
