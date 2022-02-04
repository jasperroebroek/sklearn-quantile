# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=True

# Authors: Jasper Roebroek <roebroek.jasper@gmail.com>
# License: BSD 3 clause

"""
Module providing the special case for quantile regression: predicting the maximum.
It only has a single implementation called RandomForestMaxRegression, which has the
same parameters as the regular RandomForestRegressor
"""
import threading

import joblib
from cython.parallel cimport prange
cimport openmp
cimport numpy as np
from joblib import Parallel

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.fixes import delayed, _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._forest import _generate_sample_indices

from sklearn_quantile.ensemble.quantile import BaseForestQuantileRegressor


ctypedef np.npy_intp SIZE_t              # Type for indices and counters


__all__ = ["RandomForestMaximumRegressor"]


cdef void _maximum_per_leaf(SIZE_t[::1] leaves,
                            SIZE_t[::1] unique_leaves,
                            float[::1] values,
                            SIZE_t[::1] idx,
                            float[::1] sampled_values,
                            SIZE_t n_jobs):
    """
    Index of maximum value for each unique leaf

    Parameters
    ----------
    leaves : array, shape = (n_samples)
        Leaves of a Regression tree, corresponding to weights and indices (idx)
    unique_leaves : array, shape = (n_unique_leaves)
    values : array, shape = (n_samples)
    idx : array, shape = (n_samples)
        Indices of original observations. The output will drawn from this.
    sampled_idx : shape = (n_unique_leaves)
    n_jobs : number of threads, similar to joblib
    """
    cdef:
        int i, j
        double c_leaf
        float c_max

        int n_unique_leaves = unique_leaves.shape[0]
        int n_samples = leaves.shape[0]

        int num_threads = joblib.effective_n_jobs(n_jobs)

    for i in prange(n_samples, num_threads=num_threads, nogil=True):
        c_leaf = leaves[i]
        for j in range(n_unique_leaves):
            if unique_leaves[j] == c_leaf:
                break

        if values[i] > sampled_values[j]:
            sampled_values[j] = values[i]


def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel. Based on the version in sklearn.ensemble._forest
    """
    prediction = predict(X, check_input=False)
    with lock:
        out[:] = np.maximum(prediction, out)


class RandomForestMaximumRegressor(BaseForestQuantileRegressor):
    """
    A random forest regressor predicting conditional maxima

    Implementation is equivalent to Random Forest Quantile Regressor,
    but calculation is much faster. For other quantiles revert to the
    original predictor.
    """
    def __init__(self,
                 n_estimators=10,
                 criterion='mse',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(BaseForestQuantileRegressor, self).__init__(
            base_estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.q = 1

    def fit(self, X, y, sample_weight=None):
        super(RandomForestMaximumRegressor, self).fit(X, y, sample_weight)

        for i, est in enumerate(self.estimators_):
            if self.verbose:
                print(f"Sampling tree {i+1} of {self.n_estimators}")

            mask = est.y_weights_ > 0

            leaves = est.y_train_leaves_[mask]
            idx = np.arange(self.n_samples_, dtype=np.intp)[mask]

            unique_leaves = np.unique(leaves)

            sampled_values = np.full(len(unique_leaves), -np.inf, dtype=np.float32)
            _maximum_per_leaf(leaves, unique_leaves, est.y_train_[mask, 0], idx, sampled_values, self.n_jobs)

            est.tree_.value[unique_leaves, 0, 0] = sampled_values

        return self

    def predict(self, X):
        """
        predition, based on the predict function of ForestRegressor
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        y_hat = np.zeros(X.shape[0], dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction)(e.predict, X, y_hat, lock)
            for e in self.estimators_)

        return y_hat
