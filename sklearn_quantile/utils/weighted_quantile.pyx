# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# distutils: language=c

cimport numpy as cnp
import numpy as np
from libc.math cimport isnan
from libc.stdlib cimport free, realloc, calloc, qsort
from numpy.lib.function_base import _quantile_is_valid


cdef int _compare(const void *a, const void *b) noexcept nogil:
    cdef:
        float v, a_val = (<WeightedValue *> a).value, b_val = (<WeightedValue *> b).value

    if isnan(a_val) and isnan(b_val):
        return 0
    if isnan(a_val):
        return 1
    if isnan(b_val):
        return -1

    v = a_val - b_val
    if v < 0:
        return -1
    if v >= 0:
        return 1


cdef class WeightedQuantileCalculator:
    cdef:
        WeightedValue* data
        size_t size
        size_t capacity
        Interpolation interpolation
        float total_weights
        bint sorted

    def __cinit__(self, size_t initial_capacity = 1, Interpolation interpolation = linear):
        self.capacity = initial_capacity
        self.data = <WeightedValue *> calloc(initial_capacity, sizeof(WeightedValue))
        self.interpolation = interpolation
        self.size = 0
        self.total_weights = 0
        self.sorted = False

        if self.data == NULL:
            raise MemoryError

    def __dealloc__(self):
        if self.data is not NULL:
            free(self.data)

    cdef void reset(self) noexcept nogil:
        self.size = 0
        self.total_weights = 0
        self.sorted = False

    cdef void push_data_entry(self, float a, float weight) noexcept nogil:
        """This function assumes enough space is available in self.data"""
        if isnan(a) or isnan(weight):
            return
        if weight == 0:
            return

        self.data[self.size].value = a
        self.data[self.size].weight = weight
        self.total_weights += weight
        self.size += 1

    cdef int increase_capacity(self, size_t n_samples) except -1 nogil:
        cdef:
            WeightedValue *tmp_ptr

        if n_samples > self.capacity:
            while n_samples > self.capacity:
                self.capacity *= 2
            tmp_ptr = <WeightedValue *> realloc(self.data, self.capacity * sizeof(WeightedValue))
            if tmp_ptr == NULL:
                raise MemoryError
            self.data = tmp_ptr
        return 0

    cdef void sort(self) noexcept nogil:
        qsort(<void *> self.data, self.size, sizeof(WeightedValue), _compare)
        self.sorted = True

    cdef int insert_data(self, float[:] a, float[:] weights) except -1 nogil:
        cdef:
            size_t i, n_samples = a.shape[0]

        self.reset()
        self.increase_capacity(n_samples)

        for i in range(n_samples):
            self.push_data_entry(a[i], weights[i])
        return 0

    cdef void weighted_quantile(self, float[:] q, float[:] output) noexcept nogil:
        cdef:
            float previous_val, previous_weight, weights_cum = 0,
            float frac
            size_t i, iq, q_idx = 0, n_q = q.shape[0]

        if n_q == 0:
            return

        if not self.sorted:
            self.sort()

        previous_val = self.data[0].value
        previous_weight = self.data[0].weight

        for i in range(self.size):
            if (q_idx + 1) > n_q:
                break

            weights_cum += self.data[i].weight / self.total_weights

            for iq in range(q_idx, n_q):
                if weights_cum < q[iq]:
                    continue

                if self.interpolation == linear:
                    frac = (q[iq] - previous_weight) / (weights_cum - previous_weight)
                elif self.interpolation == lower:
                    frac = 0
                elif self.interpolation == higher:
                    frac = 1
                elif self.interpolation == midpoint:
                    frac = 0.5
                elif self.interpolation == nearest:
                    frac = (q[iq] - previous_weight) / (weights_cum - previous_weight)
                    if frac < 0.5:
                        frac = 0
                    else:
                        frac = 1

                output[iq] = previous_val + frac * (self.data[i].value - previous_val)
                q_idx += 1

            previous_val = self.data[i].value
            previous_weight = weights_cum

    def py_weighted_quantile(self, a, weights, q):
        a = np.asarray(a).flatten().astype(np.float32)
        weights = np.asarray(weights).flatten().astype(np.float32)
        q = np.asarray(q).flatten().astype(np.float32)

        if not np.allclose(q, np.sort(q)):
            raise IndexError("Quantiles need to be sorted")

        if not _quantile_is_valid(q):
            raise ValueError(f"Not all quantiles are valid {q=}")

        if a.size != weights.size:
            raise IndexError(f"Values and weights need to be of same length: {a.shape=} {weights.shape=}")

        output = np.empty(q.size, dtype=np.float32)

        self.insert_data(a, weights)
        self.weighted_quantile(q, output)

        return output


def get_c_interpolation(interpolation: str):
    return {
        'linear': linear,
        'lower': lower,
        'higher': higher,
        'midpoint': midpoint,
        'nearest': nearest,
    }.get(interpolation)


cdef void _weighted_quantile(float[:, :] a,
                             float[:, :] weights,
                             float[:] q,
                             Interpolation interpolation,
                             float[:, :] output):
    cdef:
        size_t i
        size_t n = a.shape[0]
        size_t n_samples = a.shape[1]
        WeightedQuantileCalculator wqc

    wqc = WeightedQuantileCalculator(n_samples)

    with nogil:
        for i in range(n):
            wqc.insert_data(a[i], weights[i])
            wqc.weighted_quantile(q, output[:, i])


def weighted_quantile(a, q, weights=None, axis=None, overwrite_input=False, interpolation='linear',
                      keepdims=False):
    """
    Compute the q-th weighted quantile of the data along the specified axis.

    Parameters
    ----------
    a : array-like
        Input array or object that can be converted to an array.
    q : array-like of float
        Quantile or sequence of quantiles to compute, which must be between
        0 and 1 inclusive.
    weights: array-like, optional
        Weights corresponding to a.
    axis : int, optional
        Axis along which the quantiles are computed. The default is to compute
        the quantile(s) along a flattened version of the array.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by intermediate
        calculations, to save memory. In this case, the contents of the input
        `a` after this function completes is undefined.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to
        use when the desired quantile lies between two data points
        ``i < j``:

            * linear: ``i + (j - i) * fraction``, where ``fraction``
              is the fractional part of the index surrounded by ``i``
              and ``j``.
            * lower: ``i``.
            * higher: ``j``.
            * nearest: ``i`` or ``j``, whichever is nearest.
            * midpoint: ``(i + j) / 2``.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

    Returns
    -------
    quantile : scalar or ndarray
        If `q` is a single quantile and `axis=None`, then the result
        is a scalar. If multiple quantiles are given, first axis of
        the result corresponds to the quantiles. The other axes are
        the axes that remain after the reduction of `a`. The output
        dtype is ``float32``.

    References
    ----------
    1. https://en.wikipedia.org/wiki/Percentile#The_Weighted_Percentile_method
    """
    if weights is None:
        return np.quantile(a, q, axis=axis, keepdims=keepdims, overwrite_input=overwrite_input,
                           interpolation=interpolation)

    q = np.asarray(q, dtype=np.float32).ravel()

    if not _quantile_is_valid(q):
        raise ValueError("Quantiles must be in the range [0, 1]")

    if a.shape != weights.shape:
        raise IndexError("the data and weights need to be of the same shape")
    if isinstance(axis, (tuple, list)):
        raise NotImplementedError("Several axes are currently not supported.")

    if axis is None or a.ndim == 1:
        wqc = WeightedQuantileCalculator(a.size)
        quantiles = wqc.py_weighted_quantile(a, weights, q)

        if keepdims:
            quantiles = quantiles.reshape(q.size, *([1] * a.ndim))

    else:
        n_samples = a.shape[axis]

        a_t = np.moveaxis(a, axis, -1)
        a_flat = a_t.reshape(-1, n_samples)

        weights_flat = np.moveaxis(a, axis, -1).reshape(-1, n_samples)

        quantile_output_shape = (q.size,) + a_flat.shape[:-1]
        quantiles = np.empty(quantile_output_shape, dtype=np.float32)

        _weighted_quantile(
            a_flat.astype(np.float32),
            weights_flat.astype(np.float32),
            q,
            get_c_interpolation(interpolation),
            quantiles
        )

        a_shape = list(a.shape)
        a_shape[axis] = 1

        quantiles = quantiles.reshape(q.size, *a_shape)

        if not keepdims:
            quantiles = np.take(quantiles, 0, axis=axis + 1)

    if q.size == 1:
        quantiles = quantiles[0]

    return quantiles
