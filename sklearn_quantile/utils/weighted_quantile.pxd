cdef enum Interpolation:
    linear, lower, higher, midpoint, nearest

cdef int _weighted_quantile_presorted_1D(float[:] a,
                                         float[:] q,
                                         float[:] weights,
                                         float[:] quantiles,
                                         Interpolation interpolation) except -1 nogil

cdef int _weighted_quantile_unchecked_1D(float[:] a,
                                         float[:] q,
                                         float[:] weights,
                                         float[:] quantiles,
                                         Interpolation interpolation) except -1 nogil
