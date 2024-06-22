cdef struct WeightedValue:
    double value
    double weight

cdef enum Interpolation:
    linear, lower, higher, midpoint, nearest
