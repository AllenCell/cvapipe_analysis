#cython: language_level=3

cimport cython
cimport numpy as np
np.import_array()

# cython e bool parecem não trabalhar bem juntos
DTYPE = int
ctypedef np.uint8_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def calculate(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y, int n):
    cdef double tp, tn, fp, fn, xs, ys, xy
    cdef int xi, yi, i
    xs = 0
    ys = 0
    xy = 0
    for i in range(n):
        xs += x[i]
        ys += y[i]
        xy += x[i]*y[i]

    tn = xy
    tp = n - xs - ys + xy
    fp = xs - xy
    fn = n - (tp + tn + fp)
    return (tp*tn-fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5