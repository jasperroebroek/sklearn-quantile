cdef struct WeightedValue:
    double value
    double weight

cdef enum Interpolation:
    linear, lower, higher, midpoint, nearest

cdef class WeightedQuantileCalculator:
    cdef:
        WeightedValue *data
        size_t size
        size_t capacity
        Interpolation interpolation
        float total_weights
        bint sorted

    cdef void reset(self) noexcept nogil
    cdef void push_data_entry(self, float a, float weight) noexcept nogil
    cdef int increase_capacity(self, size_t n_samples) except -1 nogil
    cdef void sort(self) noexcept nogil
    cdef int insert_data(self, float[:] a, float[:] weights) except -1 nogil
    cdef void weighted_quantile(self, float[:] q, float[:] output) noexcept nogil
